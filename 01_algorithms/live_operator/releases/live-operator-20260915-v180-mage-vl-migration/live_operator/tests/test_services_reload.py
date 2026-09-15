"""Services cutover tests: real file/state transaction, fake external processes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_operator import services_reload as reload
from live_operator.cli import LifecycleError
from live_operator.processes import ProcessIdentity, StateStore


class FakeHooks:
    def __init__(self, live_config):
        self.live_config = live_config
        self.live = {11, 12, 13}
        self.calls = []
        self.next_pid = 20
        self.fail = None
        self.active = True
        self.kill_mode = "process"
        self.exec_stop = ""

    def watchdog_state(self, unit):
        return {"KillMode": self.kill_mode, "ExecStop": self.exec_stop,
                "ActiveState": "active" if self.active else "inactive"}

    def suspend_watchdog(self, unit):
        self.calls.append("suspend")
        self.active = False

    def restore_watchdog(self, unit, was_active):
        self.calls.append("restore")
        self.active = was_active

    def alive(self, identity):
        return identity.pid in self.live

    def launch_context(self, identity, config, run_dir):
        return SimpleNamespace(environment={"JIAN_KONG_VLM_REVIEW_CONFIG": str(self.live_config)},
                               python=sys.executable, cwd=str(run_dir))

    def block_new_events(self, owner):
        self.calls.append(("block", owner))
        if self.fail == "block":
            raise OSError("block write failed")
        return 100.0

    def drain_clips(self, timeout, owner, requested_at):
        self.calls.append(("drain", timeout, owner, requested_at))
        if self.fail == "drain":
            raise OSError("metadata unavailable")

    def stop_services(self, identity):
        self.calls.append(("stop", identity.pid))
        self.live.discard(identity.pid)

    def wait_port(self, verify):
        verify()
        self.calls.append("port")

    def start_services(self, release, run_dir, config, context):
        self.calls.append(("start", release.name))
        if self.fail == "start" and release.name == "candidate":
            raise OSError("candidate failed")
        self.next_pid += 1
        self.live.add(self.next_pid)
        return ProcessIdentity(self.next_pid, "new", self.next_pid, "a" * 32)

    def http_self_check(self):
        self.calls.append("http")
        if self.fail == "http" and self.next_pid == 21:
            raise LifecycleError("candidate not ready")


@pytest.fixture
def rig(tmp_path):
    releases = tmp_path / "releases"
    for name in ("candidate", "rollback"):
        root = releases / name
        (root / "live_operator").mkdir(parents=True)
        (root / "live_operator" / "cli.py").write_text("# fixture")
    current = tmp_path / "current"
    current.symlink_to(releases / "rollback")
    ca = tmp_path / "rollback-ca"
    ca.write_bytes(b"old ca")
    live = tmp_path / "vlm.json"
    base = {"schema_version": 1, "tls_ca_file": str(ca), "shared_secret_file": str(tmp_path / "secret"),
            "expected_model_version": "model-v1", "expected_prompt_revision": "prompt-v1"}
    live.write_text(json.dumps(dict(base, endpoint="https://old/v1/review")))
    live.chmod(0o600)
    for name, data in (("candidate-vlm", json.dumps(dict(base, endpoint="https://new/v1/review", tls_ca_file=str(tmp_path / "candidate-ca"))).encode()),
                       ("rollback-vlm", live.read_bytes()),
                       ("candidate-ca", b"new ca"), ("rollback-ca", b"old ca")):
        (tmp_path / name).write_bytes(data)
        (tmp_path / name).chmod(0o600)
    config = tmp_path / "live.json"
    config.write_text("{}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = StateStore(tmp_path / "state.json")
    original = {"state": "running", "run_dir": str(run_dir), "generation_id": "keep",
                "processes": {name: ProcessIdentity(pid, "original", pid, str(pid % 10) * 32).to_dict()
                              for name, pid in (("mediamtx", 11), ("deepstream", 12), ("services", 13))}}
    store.save(original)
    paths = reload.ReloadPaths(releases / "candidate", releases / "rollback", config,
                               tmp_path / "candidate-vlm", tmp_path / "rollback-vlm",
                               tmp_path / "candidate-ca", tmp_path / "rollback-ca",
                               "jiankong-live-watchdog.service")
    hooks = FakeHooks(live)
    operation = reload.ServicesReload(store, paths, hooks)
    return SimpleNamespace(**locals())


def test_cutover_changes_only_services_identity_and_atomic_release_files(rig):
    receipt = rig.operation.execute()
    assert receipt["ok"] is True
    expected = dict(rig.original, processes=dict(rig.original["processes"],
                    services=ProcessIdentity(21, "new", 21, "a" * 32).to_dict()))
    assert rig.store.load() == expected
    assert rig.current.resolve() == rig.paths.candidate_release
    assert json.loads(rig.live.read_text())["endpoint"] == "https://new/v1/review"
    assert rig.ca.read_bytes() == b"old ca"
    assert rig.paths.candidate_ca.read_bytes() == b"new ca"
    assert ("stop", 13) in rig.hooks.calls
    assert not any(call in rig.hooks.calls for call in (("stop", 11), ("stop", 12)))
    assert rig.hooks.calls.index(("drain", 8.0, "3" * 32, 100.0)) < rig.hooks.calls.index(("stop", 13))
    assert rig.hooks.active and rig.hooks.calls[-1] == "restore"


@pytest.mark.parametrize("invalid", ["state", "dead", "owner", "symlink", "watchdog", "execstop", "rollback", "ca_path"])
def test_preflight_refuses_unsafe_cutover_without_stopping_anything(rig, invalid):
    if invalid == "state":
        rig.store.save(dict(rig.original, state="stopped"))
    elif invalid == "dead":
        rig.hooks.live.remove(12)
    elif invalid == "owner":
        state = rig.store.load()
        state["processes"]["services"]["owner_token"] = None
        rig.store.save(state)
    elif invalid == "symlink":
        candidate = rig.paths.candidate_release
        candidate.rename(candidate.with_name("real"))
        candidate.symlink_to(candidate.with_name("real"))
    elif invalid == "watchdog":
        rig.hooks.kill_mode = "control-group"
    elif invalid == "execstop":
        rig.hooks.exec_stop = "/bin/stop-everything"
    elif invalid == "rollback":
        rig.paths.rollback_vlm_config.write_text("{}")
    else:
        rig.paths.candidate_vlm_config.write_text(json.dumps({"tls_ca_file": "/different/ca.pem"}))
    receipt = rig.operation.execute()
    assert receipt["ok"] is False
    assert not any(isinstance(call, tuple) and call[0] in ("stop", "start", "block") for call in rig.hooks.calls)
    assert rig.hooks.active


@pytest.mark.parametrize("failure", ["block", "drain", "start", "http"])
def test_failure_after_block_restores_entire_domain_and_readies_old_services(rig, failure):
    rig.hooks.fail = failure
    receipt = rig.operation.execute()
    assert receipt["ok"] is False
    assert receipt["rolled_back"] is True
    assert rig.current.resolve() == rig.paths.rollback_release
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()
    assert rig.ca.read_bytes() == b"old ca"
    state = rig.store.load()
    assert state["processes"]["mediamtx"] == rig.original["processes"]["mediamtx"]
    assert state["processes"]["deepstream"] == rig.original["processes"]["deepstream"]
    assert state["processes"]["services"]["pid"] != 13
    assert rig.hooks.calls[-2:] == ["http", "restore"]
    assert rig.hooks.active


def test_inactive_watchdog_remains_inactive(rig):
    rig.hooks.active = False
    assert rig.operation.execute()["ok"]
    assert not rig.hooks.active


def test_watchdog_restored_on_keyboard_interrupt_after_block(rig):
    rig.hooks.drain_clips = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    result = rig.operation.execute()
    assert not result["ok"] and result["rolled_back"]
    assert rig.hooks.active


def test_port_reuse_wait_is_bounded_and_keeps_checking_preserved_identities(monkeypatch):
    now = [0.0]
    checks = []
    monkeypatch.setattr(reload.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(reload.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(reload, "probe_port", lambda port: False)
    with pytest.raises(LifecycleError, match="8767"):
        reload.ReloadHooks().wait_port(lambda: checks.append(now[0]))
    assert now[0] == 75.0 and len(checks) > 1


def test_owned_popen_exit_cannot_be_mistaken_for_live_identity(monkeypatch):
    hooks = reload.ReloadHooks()
    child = SimpleNamespace(poll=lambda: 1, wait=lambda timeout=None: 1)
    hooks.children[101] = child
    monkeypatch.setattr(reload, "is_same_process", lambda identity: True)
    assert not hooks.alive(ProcessIdentity(101, "start", 101, "a" * 32))


def test_cli_has_fixed_services_scope_and_reports_failure_as_json(capsys, tmp_path):
    args = []
    for name in ("candidate-release", "rollback-release", "config", "candidate-vlm-config",
                 "rollback-vlm-config", "candidate-ca", "rollback-ca", "state"):
        args.extend(["--" + name, str(tmp_path / name)])
    args.extend(["--watchdog-unit", "jiankong-live-watchdog.service"])
    assert reload.main(args) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    with pytest.raises(SystemExit):
        reload.main(args + ["--component", "deepstream"])


def test_start_preserves_original_environment_python_cwd_and_owns_new_child(tmp_path, monkeypatch):
    hooks = reload.ReloadHooks()
    captured = []
    child = SimpleNamespace(pid=101)
    monkeypatch.setattr(reload.subprocess, "Popen", lambda args, **kwargs: captured.append((args, kwargs)) or child)
    monkeypatch.setattr(reload, "capture_identity", lambda process, owner_token: ProcessIdentity(process.pid, "new", 101, owner_token))
    monkeypatch.setattr(hooks, "alive", lambda identity: True)
    original = {"PATH": "/old/bin", "HOME": "/home/boshi", "LANG": "C.UTF-8", "LD_LIBRARY_PATH": "/old/lib",
                "PYTHONPATH": "/old/release", "JIAN_KONG_VLM_REVIEW_CONFIG": "/private/vlm.json",
                "JIAN_KONG_OWNER_TOKEN": "b" * 32, "VLM_EXTRA": "secret-test-value"}
    context = reload.LaunchContext(original, "/old/python3.13", "/original/cwd", 1000, 1000, (1000, 44))
    identity = hooks.start_services(tmp_path / "candidate", tmp_path / "run", tmp_path / "live.json", context)
    command, options = captured[0]
    assert command == ["/old/python3.13", "-m", "live_operator.cli", "_services", "--run-dir", str(tmp_path / "run"), "--config", str(tmp_path / "live.json")]
    assert options["cwd"] == "/original/cwd" and options["start_new_session"] is True
    assert (options["user"], options["group"], options["extra_groups"]) == (1000, 1000, (1000, 44))
    assert options["env"] == dict(original, PYTHONPATH=str(tmp_path / "candidate"), JIAN_KONG_OWNER_TOKEN=identity.owner_token)
    assert identity.owner_token != "b" * 32
    assert hooks.children[101] is child


def test_stop_reaps_own_child_and_only_stops_services(monkeypatch):
    hooks = reload.ReloadHooks()
    calls = []
    hooks.children[101] = SimpleNamespace(poll=lambda: calls.append("poll"), wait=lambda timeout: calls.append(("wait", timeout)))
    monkeypatch.setattr(hooks, "stop_component", lambda name, identity: calls.append(("stop", name, identity.pid)))
    monkeypatch.setattr(hooks, "alive", lambda identity: False)
    monkeypatch.setattr(hooks, "has_owned_processes", lambda name, identity: False)
    hooks.stop_services(ProcessIdentity(101, "start", 101, "a" * 32))
    assert calls == ["poll", ("stop", "services", 101), ("wait", 5)]


@pytest.mark.parametrize("ack", ["valid", "wrong-owner", "stale", "nan", "bool-active", "busy"])
def test_extended_drain_requires_fresh_strict_ack_and_is_bounded(rig, monkeypatch, ack):
    now = [0.0]
    monkeypatch.setattr(reload.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(reload.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(reload.time, "time", lambda: 101.0)
    status = {"service_instance": "3" * 32, "accepting": False, "active": 0, "updated_at": 100.5}
    if ack == "wrong-owner":
        status["service_instance"] = "a" * 32
    elif ack == "stale":
        status["updated_at"] = 99.0
    elif ack == "nan":
        status["updated_at"] = float("nan")
    elif ack == "bool-active":
        status["active"] = False
    elif ack == "busy":
        status["active"] = 1
    (rig.run_dir / "worker_status.json").write_text(json.dumps(status))
    rig.hooks._storage_for_run = lambda run_dir: SimpleNamespace(metadata_dir=rig.run_dir)
    rig.hooks.drain_clips = lambda *args: (_ for _ in ()).throw(LifecycleError("clip drain handshake timed out for service; forcing stop"))
    result = rig.operation.execute()
    assert result["ok"] is (ack == "valid")
    if ack != "valid":
        assert now[0] == 30.0 and result["rolled_back"]


def test_watchdog_restoration_failure_is_nonzero_even_after_ready_candidate(rig):
    rig.hooks.restore_watchdog = lambda *args: (_ for _ in ()).throw(OSError("secret must not leak"))
    result = rig.operation.execute()
    assert not result["ok"] and not result["watchdog_restored"]
    assert "secret must not leak" not in json.dumps(result)


def test_candidate_domain_change_during_readiness_is_detected_and_rolled_back(rig):
    def check():
        if rig.hooks.next_pid == 21:
            rig.live.write_text("tampered")
    rig.hooks.http_self_check = check
    result = rig.operation.execute()
    assert not result["ok"] and result["rolled_back"]
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()


def test_changed_candidate_ca_does_not_prevent_restoring_valid_rollback_domain(rig):
    def check():
        if rig.hooks.next_pid == 21:
            rig.paths.candidate_ca.write_bytes(b"changed externally")
    rig.hooks.http_self_check = check
    result = rig.operation.execute()
    assert not result["ok"] and result["rolled_back"]
    assert rig.current.resolve() == rig.paths.rollback_release
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()
    assert rig.paths.rollback_ca.read_bytes() == b"old ca"


def test_root_state_publication_preserves_original_state_file_owner(rig, monkeypatch):
    observed = []
    original = rig.store.path.stat()
    monkeypatch.setattr(reload.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reload.os, "chown", lambda path, uid, gid: observed.append((Path(path), uid, gid)))
    assert rig.operation.execute()["ok"]
    assert (rig.store.path, original.st_uid, original.st_gid) in observed


def test_launch_context_requires_root_before_reading_process_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(reload.os, "geteuid", lambda: 1000)
    with pytest.raises(LifecycleError, match="requires root"):
        reload.ReloadHooks().launch_context(ProcessIdentity(9999999, "none"), tmp_path / "live", tmp_path)


def test_launch_context_captures_owned_process_credentials_and_original_command(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    owner = "f" * 32
    config = tmp_path / "live.json"
    run_dir = tmp_path / "run"
    (proc / "environ").write_bytes(f"JIAN_KONG_OWNER_TOKEN={owner}\0PATH=/old/bin\0".encode())
    (proc / "status").write_text("Uid:\t501\t501\t501\t501\nGid:\t20\t20\t20\t20\nGroups:\t20 44\n")
    (proc / "cmdline").write_bytes("\0".join(["/old/python", "-m", "live_operator.cli", "_services", "--run-dir", str(run_dir), "--config", str(config)]).encode() + b"\0")
    (proc / "exe").symlink_to("/original/python3.13")
    (proc / "cwd").symlink_to("/original/cwd")
    real_path = Path
    original_stat = Path.stat
    def proc_stat(path, **kwargs):
        details = original_stat(path, **kwargs)
        if path == proc:
            values = list(details)
            values[4:6] = [501, 20]
            return os.stat_result(values)
        return details
    monkeypatch.setattr(Path, "stat", proc_stat)
    monkeypatch.setattr(reload, "Path", lambda path: proc if str(path) == "/proc/13" else real_path(path))
    monkeypatch.setattr(reload.os, "geteuid", lambda: 0)
    monkeypatch.setattr(reload.LiveConfig, "load", lambda path: object())
    hooks = reload.ReloadHooks()
    monkeypatch.setattr(hooks, "alive", lambda identity: True)
    result = hooks.launch_context(ProcessIdentity(13, "start", 13, owner), config, run_dir)
    assert (result.python, result.cwd, result.groups) == ("/original/python3.13", "/original/cwd", (20, 44))
    assert result.uid == proc.stat().st_uid and result.gid == proc.stat().st_gid
    assert result.environment == {"JIAN_KONG_OWNER_TOKEN": owner, "PATH": "/old/bin"}


def test_watchdog_suspend_failure_still_restores_previous_active_state(rig):
    def suspend(unit):
        rig.hooks.active = False
        raise OSError("systemctl partial failure")
    rig.hooks.suspend_watchdog = suspend
    result = rig.operation.execute()
    assert not result["ok"] and result["watchdog_restored"]
    assert rig.hooks.active
    assert not any(isinstance(call, tuple) and call[0] == "stop" for call in rig.hooks.calls)


def test_rollback_start_failure_is_reported_and_watchdog_restored(rig):
    rig.hooks.start_services = lambda *args: (_ for _ in ()).throw(OSError("failed"))
    result = rig.operation.execute()
    assert not result["ok"] and not result["rolled_back"]
    assert result["rollback_error"]["type"] == "OSError"
    assert rig.hooks.active
    assert rig.current.resolve() == rig.paths.rollback_release
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux /proc process identity")
def test_real_owned_child_is_reaped_after_services_stop():
    from live_operator.processes import capture_identity
    owner = "e" * 32
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env=dict(os.environ, JIAN_KONG_OWNER_TOKEN=owner), start_new_session=True)
    hooks = reload.ReloadHooks()
    hooks.children[child.pid] = child
    try:
        identity = capture_identity(child, owner_token=owner)
        assert hooks.alive(identity)
        hooks.stop_services(identity)
        assert child.returncode is not None
        assert not hooks.alive(identity)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)

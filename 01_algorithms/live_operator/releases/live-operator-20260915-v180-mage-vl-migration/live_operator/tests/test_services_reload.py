"""Services cutover tests: real file/state transaction, fake external processes."""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_operator import services_reload as reload
from live_operator.cli import LifecycleError
from live_operator.processes import ProcessIdentity, StateStore


class FakeHooks:
    def __init__(self, live_config):
        self.live_config = live_config
        self.lock_parent = live_config.parent
        self.live = {11, 12, 13}
        self.calls = []
        self.next_pid = 20
        self.fail = None
        self.active = True
        self.kill_mode = "process"
        self.exec_stop = ""
        self.trusted_uid = os.getuid()
        self.starts = []

    @contextmanager
    def transaction_lock(self):
        with reload.ReloadTransactionLock(self.lock_parent / "reload.lock", owner=os.getuid()):
            yield

    def validate_resolution(self, release, context):
        pass

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

    def launch_context(self, identity, config, run_dir, project_root=None):
        return SimpleNamespace(environment={"JIAN_KONG_VLM_REVIEW_CONFIG": str(self.live_config)},
                               python=sys.executable, cwd=str(run_dir), uid=os.getuid(), gid=os.getgid(), groups=tuple(os.getgroups()))

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
        self.starts.append((release, run_dir, config, dict(context.environment)))
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
        (root / "live_operator" / "__init__.py").write_text("")
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
    context = reload.LaunchContext(original, "/old/python3.13", "/original/cwd", os.getuid(), os.getgid(), (1000, 44))
    (tmp_path / "run" / "logs").mkdir(parents=True)
    (tmp_path / "run" / "logs" / "services.log").touch(mode=0o600)
    identity = hooks.start_services(tmp_path / "candidate", tmp_path / "run", tmp_path / "live.json", context)
    command, options = captured[0]
    assert command == ["/old/python3.13", "-P", "-m", "live_operator.cli", "_services", "--run-dir", str(tmp_path / "run"), "--config", str(tmp_path / "live.json")]
    assert options["cwd"] == "/original/cwd" and options["start_new_session"] is True
    assert (options["user"], options["group"], options["extra_groups"]) == (os.getuid(), os.getgid(), (1000, 44))
    assert options["env"] == dict(original, PYTHONPATH=str(tmp_path / "candidate"), PYTHONSAFEPATH="1", JIAN_KONG_OWNER_TOKEN=identity.owner_token)
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
    original_fchown = os.fchown
    def fchown(fd, uid, gid):
        observed.append((uid, gid))
        original_fchown(fd, uid, gid)
    monkeypatch.setattr(reload.os, "fchown", fchown)
    assert rig.operation.execute()["ok"]
    assert (original.st_uid, original.st_gid) in observed
    assert (rig.store.path.stat().st_uid, rig.store.path.stat().st_gid) == (original.st_uid, original.st_gid)


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
    config.write_text("{}")
    run_dir.mkdir()
    project_alias = tmp_path.with_name(tmp_path.name + "-alias")
    project_alias.symlink_to(tmp_path, target_is_directory=True)
    (proc / "environ").write_bytes(f"JIAN_KONG_OWNER_TOKEN={owner}\0PATH=/old/bin\0".encode())
    (proc / "status").write_text("Uid:\t501\t501\t501\t501\nGid:\t20\t20\t20\t20\nGroups:\t20 44\n")
    (proc / "cmdline").write_bytes("\0".join(["/old/python", "-m", "live_operator.cli", "_services", "--run-dir", str(project_alias / "run"), "--config", str(project_alias / "live.json")]).encode() + b"\0")
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
    monkeypatch.setattr(hooks, "open_services_log", lambda *args: nullcontext())
    result = hooks.launch_context(
        ProcessIdentity(13, "start", 13, owner), config, run_dir, tmp_path
    )
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


@pytest.mark.parametrize("alias", ["state", "lock", "parent", "hardlink"])
def test_state_authority_rejects_aliases_before_any_watchdog_change(rig, alias):
    original_bytes = rig.store.path.read_bytes()
    if alias == "state":
        linked = rig.tmp_path / "alias.json"
        linked.symlink_to(rig.store.path)
        rig.operation.store = StateStore(linked)
    elif alias == "lock":
        (rig.tmp_path / "state.json.lock").symlink_to(rig.live)
    elif alias == "parent":
        linked = rig.tmp_path / "alias-parent"
        linked.symlink_to(rig.tmp_path, target_is_directory=True)
        rig.operation.store = StateStore(linked / "state.json")
    else:
        os.link(rig.store.path, rig.tmp_path / "hardlink.json")
    result = rig.operation.execute()
    assert not result["ok"]
    assert not rig.hooks.calls
    assert rig.store.path.read_bytes() == original_bytes


def test_copied_component_identity_is_rejected_before_services_stop(rig):
    state = rig.store.load()
    state["processes"]["services"] = state["processes"]["deepstream"].copy()
    rig.store.save(state)
    result = rig.operation.execute()
    assert not result["ok"]
    assert not any(isinstance(call, tuple) and call[0] == "stop" for call in rig.hooks.calls)


@pytest.mark.parametrize("changed", ["group", "owner", "zombie"])
def test_alive_checks_live_process_group_owner_and_non_zombie(tmp_path, monkeypatch, changed):
    proc = tmp_path / "process"
    proc.mkdir()
    (proc / "stat").write_text("13 (services) " + ("Z" if changed == "zombie" else "S") + " rest")
    token = "b" * 32 if changed == "owner" else "a" * 32
    (proc / "environ").write_bytes(f"JIAN_KONG_OWNER_TOKEN={token}\0".encode())
    original_path = Path
    monkeypatch.setattr(reload, "Path", lambda path: proc / str(path).removeprefix("/proc/13").lstrip("/") if str(path).startswith("/proc/13") else original_path(path))
    monkeypatch.setattr(reload, "is_same_process", lambda identity: True)
    monkeypatch.setattr(reload.os, "getpgid", lambda pid: 14 if changed == "group" else 13)
    assert not reload.ReloadHooks().alive(ProcessIdentity(13, "start", 13, "a" * 32))


@pytest.mark.parametrize("attack", ["parent-link", "writable-release", "swap-release", "swap-parent"])
def test_release_authority_is_pinned_and_revalidated(rig, attack):
    if attack == "parent-link":
        original = rig.paths.candidate_release.parent
        original.rename(original.with_name("real-releases"))
        original.symlink_to(original.with_name("real-releases"), target_is_directory=True)
    elif attack == "writable-release":
        rig.paths.candidate_release.chmod(0o777)
    else:
        original_drain = rig.hooks.drain_clips
        def drain(*args):
            original_drain(*args)
            original = rig.paths.candidate_release if attack == "swap-release" else rig.paths.candidate_release.parent
            original.rename(original.with_name("replaced"))
            original.mkdir()
            if attack == "swap-release":
                (original / "live_operator").mkdir()
                (original / "live_operator" / "cli.py").write_text("# substituted")
        rig.hooks.drain_clips = drain
    result = rig.operation.execute()
    assert not result["ok"]
    assert ("start", "candidate") not in rig.hooks.calls


@pytest.mark.parametrize("failed_stage", ["stop", "preserved", "config", "current"])
def test_rollback_independently_attempts_both_publications(rig, monkeypatch, failed_stage):
    original_check = rig.hooks.http_self_check
    def check():
        if rig.hooks.next_pid == 21:
            if failed_stage == "stop":
                rig.hooks.stop_services = lambda *args: (_ for _ in ()).throw(OSError("stop failed"))
            elif failed_stage == "preserved":
                rig.hooks.live.remove(12)
            elif failed_stage == "config":
                rig.operation.live_file.replace = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("config failed"))
            else:
                rig.operation.selector.replace_link = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("current failed"))
            raise LifecycleError("readiness failed")
        original_check()
    rig.hooks.http_self_check = check
    result = rig.operation.execute()
    assert not result["ok"]
    if failed_stage != "config":
        assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()
    if failed_stage != "current":
        assert rig.current.resolve() == rig.paths.rollback_release
    assert result["rollback_errors"]
    if failed_stage in {"stop", "preserved"}:
        assert ("start", "rollback") not in rig.hooks.calls


@pytest.mark.parametrize("alias", ["same-ca", "ca-hardlink", "source-live-hardlink", "source-parent-link"])
def test_immutable_ca_and_config_sources_cannot_alias(rig, alias):
    if alias == "same-ca":
        rig.operation.paths = replace(rig.paths, candidate_ca=rig.paths.rollback_ca)
        config = json.loads(rig.paths.candidate_vlm_config.read_text())
        config["tls_ca_file"] = str(rig.paths.rollback_ca)
        rig.paths.candidate_vlm_config.write_text(json.dumps(config))
    elif alias == "ca-hardlink":
        rig.paths.candidate_ca.unlink()
        os.link(rig.paths.rollback_ca, rig.paths.candidate_ca)
    elif alias == "source-live-hardlink":
        rig.paths.rollback_vlm_config.unlink()
        os.link(rig.live, rig.paths.rollback_vlm_config)
    else:
        linked = rig.tmp_path / "source-alias"
        linked.symlink_to(rig.tmp_path, target_is_directory=True)
        rig.operation.paths = replace(rig.paths, candidate_vlm_config=linked / "candidate-vlm")
    result = rig.operation.execute()
    assert not result["ok"]
    assert ("stop", 13) not in rig.hooks.calls


def test_old_cwd_cannot_override_candidate_python_module(tmp_path, monkeypatch):
    old = tmp_path / "old"
    candidate = tmp_path / "candidate"
    for release in (old, candidate):
        (release / "live_operator").mkdir(parents=True)
        (release / "live_operator" / "__init__.py").write_text("")
        (release / "live_operator" / "cli.py").write_text("# test")
    context = reload.LaunchContext(dict(os.environ), sys.executable, str(old), os.getuid(), os.getgid(), tuple(os.getgroups()))
    actual_run = subprocess.run
    def run_as_current_user(command, **kwargs):
        # Exercise the real interpreter/import behavior; macOS test user cannot
        # set supplementary groups. Credential forwarding is asserted separately.
        for key in ("user", "group", "extra_groups"):
            kwargs.pop(key)
        return actual_run(command, **kwargs)
    monkeypatch.setattr(reload.subprocess, "run", run_as_current_user)
    result = reload.ReloadHooks().validate_resolution(candidate, context)
    assert Path(result).resolve() == (candidate / "live_operator" / "cli.py").resolve()


def test_concurrent_reload_waits_through_watchdog_restoration(rig):
    restore_entered = threading.Event()
    release_restore = threading.Event()
    second_seen = threading.Event()
    order = []
    rig.hooks.fail = "http"  # Roll back so the second operation also reaches watchdog capture.
    original_restore = rig.hooks.restore_watchdog
    def restore(*args):
        restore_entered.set()
        assert release_restore.wait(3)
        original_restore(*args)
        order.append("first_restored")
    rig.hooks.restore_watchdog = restore
    second_hooks = FakeHooks(rig.live)
    second_hooks.live.add(22)  # First operation's successfully readied rollback child.
    original_state = second_hooks.watchdog_state
    def second_state(unit):
        order.append("second_capture")
        second_seen.set()
        return original_state(unit)
    second_hooks.watchdog_state = second_state
    second = reload.ServicesReload(rig.store, rig.paths, second_hooks)
    first_thread = threading.Thread(target=rig.operation.execute)
    second_thread = threading.Thread(target=second.execute)
    first_thread.start()
    assert restore_entered.wait(3)
    second_thread.start()
    try:
        assert not second_seen.wait(0.2)
        assert second_thread.is_alive()
    finally:
        release_restore.set()
        first_thread.join(3)
        second_thread.join(3)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert second_seen.is_set()
    assert order == ["first_restored", "second_capture"]


@pytest.mark.parametrize("attack", ["file-link", "directory-link", "hardlink", "wrong-owner", "writable"])
def test_service_log_open_rejects_unsafe_runtime_paths_without_mutating_target(tmp_path, attack):
    logs = tmp_path / "run" / "logs"
    logs.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_bytes(b"must remain unchanged")
    victim.chmod(0o600)
    log = logs / "services.log"
    context = reload.LaunchContext({}, sys.executable, str(tmp_path), os.getuid(), os.getgid(), tuple(os.getgroups()))
    if attack == "file-link":
        log.symlink_to(victim)
    elif attack == "directory-link":
        logs.rename(logs.with_name("real-logs"))
        logs.symlink_to(logs.with_name("real-logs"), target_is_directory=True)
        log.write_bytes(b"must remain unchanged")
    elif attack == "hardlink":
        os.link(victim, log)
    else:
        log.write_bytes(b"must remain unchanged")
        if attack == "wrong-owner":
            context = replace(context, uid=os.getuid() + 1)
        else:
            log.chmod(0o666)
    before = victim.stat()
    with pytest.raises((ValueError, OSError, LifecycleError)):
        with reload.ReloadHooks().open_services_log(tmp_path / "run", context) as stream:
            stream.write(b"forbidden append")
    assert victim.read_bytes() == b"must remain unchanged"
    assert (victim.stat().st_uid, victim.stat().st_gid, victim.stat().st_mode) == (before.st_uid, before.st_gid, before.st_mode)


def test_stop_marker_does_not_follow_predictable_temporary_symlink(tmp_path, monkeypatch):
    victim = tmp_path / "victim"
    victim.write_bytes(b"leave alone")
    (tmp_path / ".stop_events.tmp").symlink_to(victim)
    hooks = reload.ReloadHooks()
    hooks.run_dir = tmp_path
    hooks.context = reload.LaunchContext({}, sys.executable, str(tmp_path), os.getuid(), os.getgid(), tuple(os.getgroups()))
    monkeypatch.setattr(hooks, "_storage_for_run", lambda path: SimpleNamespace(metadata_dir=tmp_path))
    requested = hooks.block_new_events("a" * 32)
    assert victim.read_bytes() == b"leave alone"
    assert json.loads((tmp_path / ".stop_events").read_text()) == {"service_instance": "a" * 32, "requested_at": requested}
    assert (tmp_path / ".stop_events").stat().st_uid == os.getuid()


def test_config_validation_uses_same_captured_bytes_as_publication(rig, monkeypatch):
    original_bytes = rig.paths.candidate_vlm_config.read_bytes()
    bad = json.loads(original_bytes)
    bad["tls_ca_file"] = str(rig.paths.rollback_ca)
    rig.paths.candidate_vlm_config.write_text(json.dumps(bad))
    actual_file = reload.PinnedFile
    def capture_then_replace_source(path, **kwargs):
        file = actual_file(path, **kwargs)
        if path == rig.paths.candidate_vlm_config:
            rig.paths.candidate_vlm_config.write_bytes(original_bytes)
        return file
    monkeypatch.setattr(reload, "PinnedFile", capture_then_replace_source)
    result = rig.operation.execute()
    assert not result["ok"]
    assert ("stop", 13) not in rig.hooks.calls
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()


def test_pinned_config_restore_never_writes_through_swapped_live_symlink(rig):
    victim = rig.tmp_path / "victim"
    victim.write_bytes(b"unrelated file")
    def readiness():
        if rig.hooks.next_pid == 21:
            rig.live.unlink()
            rig.live.symlink_to(victim)
            raise LifecycleError("candidate failed")
    rig.hooks.http_self_check = readiness
    result = rig.operation.execute()
    assert not result["ok"] and result["rolled_back"]
    assert victim.read_bytes() == b"unrelated file"
    assert not rig.live.is_symlink()
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()


def test_root_transaction_lock_rejects_symlink_and_hardlink(tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"untouched")
    lock = tmp_path / "reload.lock"
    lock.symlink_to(victim)
    with pytest.raises(OSError):
        with reload.ReloadTransactionLock(lock, owner=os.getuid()):
            pytest.fail("alias lock acquired")
    lock.unlink()
    os.link(victim, lock)
    with pytest.raises(ValueError):
        with reload.ReloadTransactionLock(lock, owner=os.getuid()):
            pytest.fail("hardlinked lock acquired")
    assert victim.read_bytes() == b"untouched"


def test_state_lock_replacement_during_transaction_prevents_publication(rig):
    real_drain = rig.hooks.drain_clips
    def drain(*args):
        real_drain(*args)
        lock = rig.store.path.with_suffix(".json.lock")
        lock.rename(lock.with_suffix(".replaced"))
        lock.touch(mode=0o600)
    rig.hooks.drain_clips = drain
    result = rig.operation.execute()
    assert not result["ok"]
    assert rig.store.load()["processes"] == rig.original["processes"]
    assert rig.hooks.active


def test_runtime_marker_writes_drop_to_service_credentials_and_restore_root(tmp_path, monkeypatch):
    effective = {"uid": 0, "gid": 0, "groups": [0]}
    monkeypatch.setattr(reload.os, "geteuid", lambda: effective["uid"])
    monkeypatch.setattr(reload.os, "getegid", lambda: effective["gid"])
    monkeypatch.setattr(reload.os, "getgroups", lambda: effective["groups"])
    monkeypatch.setattr(reload.os, "seteuid", lambda uid: effective.__setitem__("uid", uid))
    monkeypatch.setattr(reload.os, "setegid", lambda gid: effective.__setitem__("gid", gid))
    monkeypatch.setattr(reload.os, "setgroups", lambda groups: effective.__setitem__("groups", groups))
    writes = []
    actual_open = os.open
    def checked_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR):
            writes.append((effective["uid"], effective["gid"], tuple(effective["groups"])))
        return actual_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(reload.os, "open", checked_open)
    hooks = reload.ReloadHooks()
    hooks.run_dir = tmp_path
    hooks.context = reload.LaunchContext({}, sys.executable, str(tmp_path), 1001, 1002, (1002, 44))
    monkeypatch.setattr(hooks, "_storage_for_run", lambda path: SimpleNamespace(metadata_dir=tmp_path))
    hooks.block_new_events("a" * 32)
    assert writes == [(1001, 1002, (1002, 44))]
    assert effective == {"uid": 0, "gid": 0, "groups": [0]}


def test_privileged_systemctl_does_not_search_caller_path(monkeypatch):
    commands = []
    monkeypatch.setenv("PATH", "/untrusted/runtime/bin")
    monkeypatch.setattr(reload.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or subprocess.CompletedProcess(command, 0, stdout="active\n"))
    assert reload.ReloadHooks._systemctl("start", "jiankong-live-watchdog.service") == "active\n"
    assert commands == [["/usr/bin/systemctl", "start", "jiankong-live-watchdog.service"]]


def test_watchdog_state_queries_empty_execstop_value_explicitly(monkeypatch):
    unit = "jiankong-live-watchdog.service"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command == [
            "/usr/bin/systemctl", "show", unit,
            "--property=KillMode,ActiveState,LoadState",
        ]:
            output = "KillMode=process\nLoadState=loaded\nActiveState=active\n"
        elif command == [
            "/usr/bin/systemctl", "show", unit, "--property=ExecStop", "--value",
        ]:
            output = ""  # systemd 249: rc=0 and zero bytes means unset/empty.
        else:
            raise AssertionError(f"unexpected systemctl command: {command!r}")
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(reload.subprocess, "run", run)

    assert reload.ReloadHooks().watchdog_state(unit) == {
        "KillMode": "process", "LoadState": "loaded",
        "ActiveState": "active", "ExecStop": "",
    }
    assert len(commands) == 2


@pytest.mark.parametrize("failure", ["after-rename", "directory-fsync", "interrupt", "handled-signal"])
def test_state_publication_failure_reconciles_known_inode_before_rollback(rig, monkeypatch, failure):
    actual_replace = os.replace
    actual_fsync = os.fsync
    renamed = False
    injected = False
    signal_observed_consistent_pin = []
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    def handled_exit(signum, frame):
        pin = rig.operation.store.file
        actual = rig.store.path.stat()
        signal_observed_consistent_pin.append(pin.expected == (actual.st_dev, actual.st_ino))
        raise KeyboardInterrupt()
    previous_handler = signal.signal(signal.SIGTERM, handled_exit)
    def replace_file(source, destination, **kwargs):
        nonlocal renamed, injected
        result = actual_replace(source, destination, **kwargs)
        if destination == rig.store.path.name and not renamed:
            renamed = True
            if failure != "directory-fsync":
                injected = True
                if failure == "interrupt":
                    raise KeyboardInterrupt()
                if failure == "handled-signal":
                    signal.raise_signal(signal.SIGTERM)
                    return result
                raise OSError("injected immediately after successful state rename")
        return result
    def fsync_file(fd):
        nonlocal injected
        if failure == "directory-fsync" and renamed and not injected and stat.S_ISDIR(os.fstat(fd).st_mode):
            injected = True
            raise OSError("injected state directory fsync failure")
        return actual_fsync(fd)
    monkeypatch.setattr(reload.os, "replace", replace_file)
    monkeypatch.setattr(reload.os, "fsync", fsync_file)
    try:
        result = rig.operation.execute()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask
    if failure == "handled-signal":
        assert signal_observed_consistent_pin == [True]
    assert renamed and injected
    assert not result["ok"] and result["rolled_back"]
    assert ("stop", 21) in rig.hooks.calls
    assert ("start", "rollback") in rig.hooks.calls
    assert rig.current.resolve() == rig.paths.rollback_release
    assert rig.live.read_bytes() == rig.paths.rollback_vlm_config.read_bytes()
    expected = dict(rig.original, processes=dict(rig.original["processes"],
                    services=ProcessIdentity(22, "new", 22, "a" * 32).to_dict()))
    assert rig.store.load() == expected
    assert rig.hooks.active and result["watchdog_restored"]


@pytest.mark.parametrize("replacement", ["different-inode", "changed-content", "symlink"])
def test_publication_reconciliation_rejects_unknown_replacements(tmp_path, monkeypatch, replacement):
    path = tmp_path / "pinned.json"
    path.write_bytes(b"original")
    path.chmod(0o600)
    file = reload.PinnedFile(path)
    old_identity = file.expected
    actual_replace = os.replace
    def replace_then_tamper(source, destination, **kwargs):
        result = actual_replace(source, destination, **kwargs)
        if replacement == "changed-content":
            path.write_bytes(b"externally changed")
        else:
            # Keep the known published inode linked elsewhere to prevent inode
            # reuse; the new authority must still be rejected even with same bytes.
            path.rename(path.with_name("known-inode"))
            if replacement == "symlink":
                path.symlink_to(path.with_name("known-inode"))
            else:
                path.write_bytes(b"candidate")
                path.chmod(0o600)
        raise OSError("injected publication failure with unknown replacement")
    monkeypatch.setattr(reload.os, "replace", replace_then_tamper)
    try:
        with pytest.raises(OSError, match="injected publication failure"):
            file.replace(b"candidate")
        assert file.expected == old_identity and file.content == b"original"
        with pytest.raises((ValueError, OSError)):
            file.verify()
    finally:
        file.close()


def test_writable_legacy_alias_cannot_redirect_post_normalization_operations(rig):
    """A retargeted production-style alias is never an ongoing authority."""

    alias_root = rig.tmp_path.with_name(rig.tmp_path.name + "-project-alias")
    alias_root.symlink_to(rig.tmp_path, target_is_directory=True)
    attacker_root = rig.tmp_path.with_name(rig.tmp_path.name + "-attacker")
    attacker_root.mkdir(mode=0o777)
    attacker_root.chmod(0o777)
    (attacker_root / "run").mkdir()
    attacker_live = attacker_root / rig.live.name
    attacker_live.write_text("attacker sentinel")
    (attacker_root / rig.paths.rollback_ca.name).write_bytes(b"attacker ca")
    state = rig.store.load()
    state["run_dir"] = str(alias_root / "run")
    rig.store.save(state)
    rig.hooks.live_config = alias_root / rig.live.name
    rollback = json.loads(rig.paths.rollback_vlm_config.read_text())
    rollback["tls_ca_file"] = str(alias_root / rig.paths.rollback_ca.name)
    rig.paths.rollback_vlm_config.write_text(json.dumps(rollback))
    rig.live.write_bytes(rig.paths.rollback_vlm_config.read_bytes())
    rig.operation.paths = replace(rig.paths, live_vlm_config=rig.live)
    original_suspend = rig.hooks.suspend_watchdog

    def retarget_after_normalization(unit):
        original_suspend(unit)
        alias_root.unlink()
        alias_root.symlink_to(attacker_root, target_is_directory=True)

    rig.hooks.suspend_watchdog = retarget_after_normalization

    result = rig.operation.execute()

    assert result["ok"] is True, result
    _, started_run, started_config, environment = rig.hooks.starts[0]
    assert started_run == rig.run_dir
    assert started_config == rig.config
    assert environment["JIAN_KONG_VLM_REVIEW_CONFIG"] == str(rig.live)
    assert json.loads(rig.live.read_text())["tls_ca_file"] == str(rig.paths.candidate_ca)
    assert attacker_live.read_text() == "attacker sentinel"
    assert (attacker_root / rig.paths.rollback_ca.name).read_bytes() == b"attacker ca"


def test_alias_binding_rejects_writable_target_outside_trusted_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir(mode=0o700)
    expected = project / "inside.json"
    expected.write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o777)
    outside.chmod(0o777)
    target = outside / "target.json"
    target.write_text("outside")
    alias = project / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(LifecycleError, match="trusted project"):
        reload._normalize_legacy_path(alias, expected, project, "test authority")


def test_normalized_path_discards_writable_alias_authority(tmp_path):
    project = tmp_path / "project"
    project.mkdir(mode=0o700)
    expected = project / "expected.json"
    expected.write_text("expected")
    replacement = project / "replacement.json"
    replacement.write_text("replacement")
    alias = project / "alias.json"
    alias.symlink_to(expected)
    canonical = reload._normalize_legacy_path(alias, expected, project, "test authority")
    alias.unlink()
    alias.symlink_to(replacement)

    assert canonical == expected
    assert canonical.read_text() == "expected"


def test_candidate_configuration_rejects_alias_ca_before_mutation(rig):
    candidate_alias = rig.tmp_path / "candidate-ca-alias"
    candidate_alias.symlink_to(rig.paths.candidate_ca)
    payload = json.loads(rig.paths.candidate_vlm_config.read_text())
    payload["tls_ca_file"] = str(candidate_alias)
    rig.paths.candidate_vlm_config.write_text(json.dumps(payload))
    result = rig.operation.execute()

    assert result["ok"] is False
    assert result["error"]["stage"] == "preflight"
    assert not any(
        isinstance(call, tuple) and call[0] in ("stop", "start", "block")
        for call in rig.hooks.calls
    )


def test_launch_context_rejects_relative_process_run_or_config_paths(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    owner = "f" * 32
    config = tmp_path / "live.json"
    run_dir = tmp_path / "run"
    config.write_text("{}")
    run_dir.mkdir()
    (proc / "environ").write_bytes(
        f"JIAN_KONG_OWNER_TOKEN={owner}\0PATH=/old/bin\0".encode()
    )
    (proc / "status").write_text(
        "Uid:\t501\t501\t501\t501\nGid:\t20\t20\t20\t20\nGroups:\t20 44\n"
    )
    (proc / "cmdline").write_bytes(
        "\0".join(
            [
                "/old/python",
                "-m",
                "live_operator.cli",
                "_services",
                "--run-dir",
                "relative-run",
                "--config",
                "relative-config.json",
            ]
        ).encode()
        + b"\0"
    )
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
    monkeypatch.setattr(
        reload,
        "Path",
        lambda path: proc if str(path) == "/proc/13" else real_path(path),
    )
    monkeypatch.setattr(reload.os, "geteuid", lambda: 0)
    hooks = reload.ReloadHooks()
    monkeypatch.setattr(hooks, "alive", lambda identity: True)

    with pytest.raises(LifecycleError, match="absolute"):
        hooks.launch_context(ProcessIdentity(13, "start", 13, owner), config, run_dir)


def test_generic_pinned_directory_still_rejects_project_alias(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)

    with pytest.raises(OSError):
        reload.PinnedDirectory(alias)

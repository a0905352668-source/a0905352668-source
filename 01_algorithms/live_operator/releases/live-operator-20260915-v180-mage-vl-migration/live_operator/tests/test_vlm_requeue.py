from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from live_operator.vlm_requeue import (
    VLMRequeueError,
    apply_http500_requeue,
    plan_http500_requeue,
)
from live_operator.vlm_review import VLMReviewConfig
from live_operator.vlm_state import VLMReviewStateStore


HTTP_500 = "VLM review service returned HTTP 500"


def _config() -> VLMReviewConfig:
    return VLMReviewConfig(
        endpoint="http://127.0.0.1/v1/review",
        shared_secret_file=Path("/tmp/vlm-requeue-test-secret"),
        expected_model_version="mage-vl-v1",
        expected_prompt_revision="prompt-v1",
        expected_evidence_revision="evidence-v1",
    )


def _state(
    result: str,
    *,
    error: str | None = None,
    retryable: bool | None = None,
    model: str = "mage-vl-v1",
    prompt: str = "prompt-v1",
    evidence: str = "evidence-v1",
) -> dict[str, object]:
    value: dict[str, object] = {
        "vlm_filter_result": result,
        "vlm_filter_expected_model_version": model,
        "vlm_filter_expected_prompt_revision": prompt,
        "vlm_filter_evidence_revision": evidence,
    }
    if error is not None:
        value["vlm_filter_error"] = error
    if retryable is not None:
        value["vlm_filter_retryable"] = retryable
    return value


def _write_sidecar(run_dir: Path) -> Path:
    dashboard = run_dir / "dashboard"
    dashboard.mkdir(parents=True)
    path = dashboard / "vlm_filter_states.json"
    states = {
        "a-pass": _state("pass"),
        "b-filter": _state("filter"),
        "c-uncertain": _state("uncertain"),
        "d-pending": _state("pending"),
        "e-http500-z": _state("error", error=HTTP_500, retryable=True),
        "f-not-retryable": _state("error", error=HTTP_500, retryable=False),
        "g-other-error": _state("error", error="VLM review service returned HTTP 502", retryable=True),
        "h-wrong-model": _state("error", error=HTTP_500, retryable=True, model="mage-vl-v0"),
        "i-wrong-prompt": _state("error", error=HTTP_500, retryable=True, prompt="prompt-v0"),
        "j-wrong-evidence": _state("error", error=HTTP_500, retryable=True, evidence="evidence-v0"),
        "k-http500-a": _state("error", error=HTTP_500, retryable=True),
    }
    path.write_text(json.dumps(states, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o640)
    return path


def test_plan_selects_only_retryable_http500_errors_at_exact_revisions(tmp_path: Path) -> None:
    """Removing any selection predicate would incorrectly include a fixture entry."""

    run_dir = tmp_path / "live_20260915_120000"
    sidecar = _write_sidecar(run_dir)
    source = sidecar.read_bytes()
    before = sidecar.stat()

    plan = plan_http500_requeue(run_dir, _config())

    assert plan.candidate_event_ids == ("e-http500-z", "k-http500-a")
    assert plan.source_sha256 == hashlib.sha256(source).hexdigest()
    assert plan.candidate_ids_sha256 == hashlib.sha256(
        b"e-http500-z\nk-http500-a"
    ).hexdigest()
    after = sidecar.stat()
    assert sidecar.read_bytes() == source
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o640
    assert after.st_mtime_ns == before.st_mtime_ns
    assert not sidecar.with_suffix(".json.lock").exists()


def test_apply_removes_only_planned_overlays_with_a_private_exact_backup(tmp_path: Path) -> None:
    """A wrong deletion set, non-atomic write, or weak backup must fail this test."""

    run_dir = tmp_path / "live_20260915_120001"
    sidecar = _write_sidecar(run_dir)
    source = sidecar.read_bytes()
    plan = plan_http500_requeue(run_dir, _config())
    backup = tmp_path / "operator-backup.json"

    result = apply_http500_requeue(plan, backup)

    assert result.removed_count == 2
    assert result.removed_event_ids == ("e-http500-z", "k-http500-a")
    assert backup.read_bytes() == source
    assert result.backup_sha256 == hashlib.sha256(source).hexdigest()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    remaining = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(remaining) == {
        "a-pass",
        "b-filter",
        "c-uncertain",
        "d-pending",
        "f-not-retryable",
        "g-other-error",
        "h-wrong-model",
        "i-wrong-prompt",
        "j-wrong-evidence",
    }
    assert not any(path.suffix == ".tmp" for path in sidecar.parent.iterdir())
    assert plan_http500_requeue(run_dir, _config()).candidate_event_ids == ()


def test_apply_waits_for_the_state_store_lock_then_rejects_changed_source(tmp_path: Path) -> None:
    """Bypassing the state-store lock or skipping the post-lock hash check is unsafe."""

    run_dir = tmp_path / "live_20260915_120002"
    sidecar = _write_sidecar(run_dir)
    plan = plan_http500_requeue(run_dir, _config())
    backup = tmp_path / "operator-backup.json"
    finished = threading.Event()
    failures: list[BaseException] = []

    def apply_in_thread() -> None:
        try:
            apply_http500_requeue(plan, backup)
        except BaseException as error:  # surfaced below with its original type
            failures.append(error)
        finally:
            finished.set()

    store = VLMReviewStateStore(sidecar, run_dir=run_dir)
    with store._exclusive():
        worker = threading.Thread(target=apply_in_thread)
        worker.start()
        assert not finished.wait(0.1)
        sidecar.write_text(sidecar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    worker.join(timeout=2)

    assert finished.is_set()
    assert len(failures) == 1
    assert isinstance(failures[0], VLMRequeueError)
    assert "source changed" in str(failures[0])
    assert not backup.exists()


def test_apply_aborts_when_any_dry_run_fence_is_tampered(tmp_path: Path) -> None:
    """Tampered authority, revisions, error predicate, or selection must never mutate."""

    run_dir = tmp_path / "live_20260915_120003"
    _write_sidecar(run_dir)
    plan = plan_http500_requeue(run_dir, _config())
    changed_config = replace(plan.config, expected_model_version="mage-vl-v2")
    changed_selection = ("f-not-retryable",)
    tampered_plans = (
        replace(plan, metadata_dir=tmp_path / "other-dashboard"),
        replace(plan, config=changed_config),
        replace(plan, error_text="VLM review service returned HTTP 502"),
        replace(
            plan,
            candidate_event_ids=changed_selection,
            candidate_ids_sha256=hashlib.sha256(b"f-not-retryable").hexdigest(),
        ),
    )

    for index, tampered in enumerate(tampered_plans):
        with pytest.raises(VLMRequeueError):
            apply_http500_requeue(tampered, tmp_path / f"backup-{index}.json")

    assert plan_http500_requeue(run_dir, _config()).candidate_event_ids == (
        "e-http500-z",
        "k-http500-a",
    )


def test_wrapper_emits_a_secret_free_dry_run_json_object(tmp_path: Path) -> None:
    """The operator wrapper must be runnable and expose only the dry-run audit data."""

    run_dir = tmp_path / "live_20260915_120004"
    sidecar = _write_sidecar(run_dir)
    config_path = tmp_path / "vlm-review.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "endpoint": "http://127.0.0.1/v1/review",
                "shared_secret_file": "/tmp/not-read-by-requeue",
                "expected_model_version": "mage-vl-v1",
                "expected_prompt_revision": "prompt-v1",
                "expected_evidence_revision": "evidence-v1",
            }
        ),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)
    script = Path(__file__).parents[2] / "scripts" / "jiankong-vlm-requeue"
    completed = subprocess.run(
        [script, "--run-dir", run_dir, "--config", config_path],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "JIAN_KONG_PYTHON_BIN": sys.executable},
    )

    payload = json.loads(completed.stdout)
    assert payload == {
        "candidate_count": 2,
        "candidate_ids_sha256": hashlib.sha256(
            b"e-http500-z\nk-http500-a"
        ).hexdigest(),
        "mode": "dry-run",
        "source_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    assert "not-read-by-requeue" not in completed.stdout

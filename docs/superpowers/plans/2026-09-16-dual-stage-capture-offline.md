# Dual-Stage Screen-Capture Offline Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-first, shared-model second Mage-VL review that decides whether a genuine phone could be photographing a protected screen, then complete a frozen historical offline comparison without changing live alarm visibility.

**Architecture:** Keep the current phone-authenticity review byte-for-byte compatible as stage one. Add a second prompt and evidence path to the same loaded Mage-VL process, protected by a production-first gate and a quiet-period/rate-limited offline dispatcher. Combine camera-specific 2.5D visibility with chronological full-scene/person/phone evidence, and write candidate results only to an isolated experiment report.

**Tech Stack:** Python 3.12, Pillow, OpenCV, PyTorch/Transformers, Mage-VL AWQ INT4, signed HTTPS, JSON/JSONL, pytest, systemd 245.

**Spec:** `docs/superpowers/specs/2026-09-16-screen-capture-possibility-offline-design.md`

## Global Constraints

- Keep `http://192.168.50.2:8767/`, cameras, recording, MediaMTX, DeepStream, and the dashboard running; this plan never restarts them.
- Stage one keeps its exact model version, prompt text/revision, evidence revision, endpoint, labels, and cache semantics.
- `.54` keeps one loaded Mage-VL process. Never repeat the rejected two-process experiment.
- Production is always higher priority. Offline work is single-concurrency, starts after 30 production-quiet seconds, and pauses on production arrival or any health/resource breach.
- Offline work never writes live events, overlays, reviews, dashboard state, or calibrations.
- Historical inputs are read-only; output goes to a separate ignored directory with an immutable manifest hash.
- `UNCERTAIN`, disagreement, or missing calibration fails open to `KEEP_SUSPECTED_CAPTURE`.
- Complete the focused test matrix and frozen historical comparison before proposing shadow or production filtering.
- Use test-first development and a focused commit for every task.

## File Structure

- Create `live_operator/inference_priority.py` for admission, cancellation, quiet time, and metrics.
- Create `live_operator/capture_policy.py` for labels and conservative aggregation.
- Create `live_operator/capture_visibility.py` for sidecar validation and spatial evaluation.
- Create `live_operator/capture_evidence.py` for full-scene/person/phone panels.
- Create `live_operator/capture_dataset.py` for immutable manifests and label validation.
- Create `live_operator/capture_offline.py` for dispatch, resume, and reports.
- Modify `live_operator/mage_vl_service.py` to preserve `/v1/review` and add `/v1/capture-review` using the same model.
- Create `scripts/jiankong-capture-offline` as the operator entry point.
- Add eight camera sidecars under `02_configs/surveillance/capture_visibility_v1/`.
- Add focused tests under `live_operator/tests/` and a final redacted receipt at `01_algorithms/CAPTURE_OFFLINE_VALIDATION_20260916.md`.

---

### Task 1: Production-first inference gate

**Files:**
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/inference_priority.py`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_inference_priority.py`

**Interfaces:**
- `ProductionFirstGate(clock: Callable[[], float], quiet_seconds: float = 30.0)`.
- `production_arrived() -> None`, `try_acquire(kind: Literal["production", "offline"]) -> GateLease | None`, `snapshot() -> dict[str, object]`.
- `GateLease.cancelled: threading.Event`, `release(latency_seconds: float, error: bool = False) -> None`.

- [ ] **Step 1: Write failing priority and cancellation tests**

```python
def test_production_arrival_cancels_active_offline(fake_clock):
    gate = ProductionFirstGate(fake_clock, quiet_seconds=30.0)
    fake_clock.advance(30.0)
    offline = gate.try_acquire("offline")
    assert offline is not None
    gate.production_arrived()
    assert offline.cancelled.is_set()
    assert gate.try_acquire("production") is None
    offline.release(latency_seconds=0.5)
    assert gate.try_acquire("production") is not None
```

- [ ] **Step 2: Run and observe the missing-module failure**

Run: `python -m pytest live_operator/tests/test_inference_priority.py -q`

- [ ] **Step 3: Implement the synchronized gate**

Use one lock, one active lease, waiting-production count, last-production timestamp, admission/rejection counters, latency totals, and error totals. Offline admission fails while production waits, another lease is active, or quiet time is incomplete. Invalid kinds raise `ValueError`.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest live_operator/tests/test_inference_priority.py -q`

Commit: `git commit -m "feat: add production-first inference gate"`

### Task 2: Visibility and conservative policy

**Files:**
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/capture_visibility.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/capture_policy.py`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_visibility.py`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_policy.py`

**Interfaces:**
- `VisibilityRelation(screen_id: str, state: Literal["possible", "blocked", "unknown"], zone_id: str | None, occluder_id: str | None)`.
- `load_visibility_sidecar(path: Path) -> CaptureVisibilityConfig`.
- `evaluate_visibility(config, *, screen_id: str, anchor: tuple[float, float]) -> VisibilityRelation`.
- `StageTwoLabel`, `FinalOutcome`, and `aggregate_capture_decision(first_stage_result, visibility, stage_two) -> FinalDecision`.

- [ ] **Step 1: Write failing schema, polygon, and truth-table tests**

```python
@pytest.mark.parametrize(
    ("first", "visibility", "stage_two", "expected"),
    [
        ("filter", "possible", "CAPTURE_POSSIBLE", "FILTER_NOT_PHONE"),
        ("pass", "blocked", "CAPTURE_POSSIBLE", "FILTER_CAPTURE_IMPOSSIBLE"),
        ("pass", "possible", "CAPTURE_POSSIBLE", "KEEP_CAPTURE_POSSIBLE"),
        ("pass", "possible", "UNCERTAIN", "KEEP_SUSPECTED_CAPTURE"),
        ("uncertain", "possible", "IMPOSSIBLE_FLAT_OR_DOWN", "KEEP_SUSPECTED_CAPTURE"),
        ("pass", "unknown", "IMPOSSIBLE_BLOCKED", "KEEP_SUSPECTED_CAPTURE"),
    ],
)
def test_conservative_table(first, visibility, stage_two, expected):
    relation = VisibilityRelation("screen_01", visibility, None, None)
    decision = aggregate_capture_decision(first, relation, StageTwoLabel(stage_two))
    assert decision.outcome.value == expected
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest live_operator/tests/test_capture_visibility.py live_operator/tests/test_capture_policy.py -q`

- [ ] **Step 3: Implement schema version 1 and policy**

Sidecars contain camera/frame size, screen IDs, capture zones, blocked zones with verified occluder IDs, unknown zones, and occluder polygons. Reject duplicates, non-finite/out-of-frame coordinates, polygons with fewer than three distinct points, and unknown references. Blocked zones win over possible zones; unlabelled space is unknown.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest live_operator/tests/test_capture_visibility.py live_operator/tests/test_capture_policy.py -q`

Commit: `git commit -m "feat: add screen visibility capture policy"`

### Task 3: Immutable historical manifest

**Files:**
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/capture_dataset.py`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_dataset.py`

**Interfaces:**
- `build_manifest(review_pool: Path, runs_root: Path, output: Path, enabled_cameras: frozenset[str], per_camera_limit: int) -> FrozenManifest`.
- `validate_labels(manifest, labels_path) -> tuple[LabeledEvent, ...]`.
- `split_by_source(events, seed: str) -> DatasetSplit` with no source/event leakage.

- [ ] **Step 1: Write deterministic hashing and leakage tests**

```python
def test_manifest_is_deterministic(tmp_path):
    pool = reviewed_pool_fixture(tmp_path)
    runs = source_runs_fixture(tmp_path)
    a = build_manifest(pool, runs, tmp_path / "a.json", frozenset({"camera08"}), 10)
    b = build_manifest(pool, runs, tmp_path / "b.json", frozenset({"camera08"}), 10)
    assert a.digest == b.digest
    assert a.entries == b.entries
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest live_operator/tests/test_capture_dataset.py -q`

- [ ] **Step 3: Implement read-only scanning and label validation**

Accept only non-symlink reviewed-event directories containing `review.json`, `event.mp4`, and `overlay.json`. Resolve the persisted first-stage result and revisions from the matching source run's `dashboard/events.json`, using `source_run_id` and `event_id` from `review.json`; missing matches remain explicit `uncertain` records. Record path, size, SHA-256, camera, and source ID. Write mode-0600 atomically. Labels are `CAPTURE_POSSIBLE`, `SUSPECTED_CAPTURE`, `VIEWING_PHONE`, `FLAT_OR_DOWN`, `AWAY_FROM_SCREEN`, `BLOCKED`, and `NOT_PHONE`, each with a reason.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest live_operator/tests/test_capture_dataset.py -q`

Commit: `git commit -m "feat: freeze capture review datasets"`

### Task 4: Full-scene temporal evidence

**Files:**
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/capture_evidence.py`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_evidence.py`

**Interfaces:**
- `CAPTURE_EVIDENCE_REVISION = "scene-person-phone-span5s-16f-jpeg92-v1"`.
- `decode_capture_frames(video_path, overlay, visibility) -> tuple[CaptureSequence, ...]`.
- `build_capture_panel(full_frame, *, person_box, phone_box, screen_polygons, occluder_polygons, image_size=448) -> Image.Image`.

- [ ] **Step 1: Write failing synthetic-video tests**

```python
def test_panel_keeps_scene_and_native_detail(synthetic_frame):
    panel = build_capture_panel(
        synthetic_frame,
        person_box=(100, 80, 180, 220),
        phone_box=(150, 120, 165, 145),
        screen_polygons=[[(10, 10), (60, 10), (60, 50), (10, 50)]],
        occluder_polygons=[[(70, 0), (90, 0), (90, 200), (70, 200)]],
    )
    assert panel.size == (896, 448)
    colors = set(panel.getdata())
    assert (255, 0, 0) in colors
    assert (0, 255, 0) in colors
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest live_operator/tests/test_capture_evidence.py -q`

- [ ] **Step 3: Implement 16-frame chronological panels**

Reuse metadata/frame-selection concepts, not the stage-one `_evidence_panel`. Show a letterboxed full scene, native-detail person crop, and supplemental phone/hand crop. Draw screen/occluder outlines only on the scene half. Preserve native pixels and JPEG round-trip the final panel at quality 92.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest live_operator/tests/test_capture_evidence.py -q`

Commit: `git commit -m "feat: build temporal capture evidence"`

### Task 5: Shared-model second endpoint

**Files:**
- Modify: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/mage_vl_service.py`
- Modify: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_mage_vl_service.py`
- Modify: `01_algorithms/mage_vl_migration/run_mage_vl_service_54.sh`

**Interfaces:**
- Preserve `POST /v1/review` exactly.
- Add `POST /v1/capture-review` with ZIP members `request.json`, `clip.mp4`, `overlay.json`, and `visibility.json`.
- Add health fields `capture_prompt_revision`, `capture_evidence_revision`, and `scheduler`.
- Add `MageVLReviewer.review_capture(video_path, overlay_path, visibility_path, cancel_event) -> dict[str, object]`.

- [ ] **Step 1: Freeze stage-one compatibility in tests**

Assert existing prompt text/hash, evidence revision, endpoint, labels, request fence, cache hits, and response signatures remain unchanged.

- [ ] **Step 2: Write failing stage-two and priority tests**

```python
def test_capture_endpoint_uses_same_reviewer(fake_reviewer, signed_capture_archive):
    response = post("/v1/capture-review", signed_capture_archive)
    assert response.status == 200
    assert fake_reviewer.capture_calls == 1
    assert fake_reviewer.phone_calls == 0
    assert response.json["label"] == "CAPTURE_POSSIBLE"
```

- [ ] **Step 3: Run and confirm failure**

Run: `python -m pytest live_operator/tests/test_mage_vl_service.py live_operator/tests/test_inference_priority.py -q`

- [ ] **Step 4: Add a second prompt/template without a second model**

The only valid stage-two labels are `CAPTURE_POSSIBLE`, `IMPOSSIBLE_FLAT_OR_DOWN`, `IMPOSSIBLE_AWAY_FROM_SCREEN`, `IMPOSSIBLE_BLOCKED`, `NOT_PHONE_OR_NO_CAPTURE_ACTION`, and `UNCERTAIN`. Prepare a second chat template in `__init__` while retaining one processor/model/CUDA allocation. Use deterministic generation with 24 output tokens. A production-triggered stopping criterion converts cancelled offline work to nonterminal `UNCERTAIN` and does not cache it.

- [ ] **Step 5: Integrate priority and distinct capture cache**

Production records arrival before admission. Offline returns 503 with `Retry-After: 30` until quiet. Preserve existing stage-one cache lookup; write stage two only under a capture namespace.

- [ ] **Step 6: Add `--offline-quiet-seconds 30` to the existing one-process runner**

Do not add a systemd unit, second GPU lock, second model path, or worker process.

- [ ] **Step 7: Run service regressions and commit**

Run: `python -m pytest live_operator/tests/test_mage_vl_service.py live_operator/tests/test_vlm_review.py live_operator/tests/test_vlm_state.py live_operator/tests/test_inference_priority.py -q`

Commit: `git commit -m "feat: add shared-model capture review"`

### Task 6: Resumable dispatcher and report

**Files:**
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/capture_offline.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_offline.py`
- Create: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/scripts/jiankong-capture-offline`

**Interfaces:**
- Commands: `build-manifest`, `validate-labels`, `dry-run`, `run`, `report`.
- `run` requires manifest, labels, visibility directory, isolated output directory, endpoint, shared-secret file, foreground status URL, and maximum two offline requests per minute.
- Outputs append-only `results.jsonl`, atomic `checkpoint.json`, `metrics.json`, and `report.html`.

- [ ] **Step 1: Write failing fencing, pause, and resume tests**

```python
@pytest.mark.parametrize("breach", ["production_busy", "fps_drop", "camera_loss", "vlm_unhealthy", "memory_pressure"])
def test_breach_pauses_before_claim(rig, breach):
    rig.set_breach(breach)
    assert rig.run_once().state == "paused"
    assert rig.capture_requests == 0
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest live_operator/tests/test_capture_offline.py -q`

- [ ] **Step 3: Implement signed requests and gates**

Before each claim require dashboard 200, eight cameras, FPS at least 95% of frozen baseline, no alerts, Mage ready, no production active/waiting, and 30 quiet seconds. A 503, cancellation, timeout, or breach writes a pause record and never fabricates a label.

- [ ] **Step 4: Implement deterministic comparison reports**

Include baseline result, stage-two label, visibility, final outcome, ground truth, per-camera confusion matrices, false-positive reduction, positive retention, uncertain/cancel counts, request latency, and production health deltas. Keep identifiers only in mode-0600 ignored output.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest live_operator/tests/test_capture_offline.py live_operator/tests/test_capture_dataset.py live_operator/tests/test_capture_policy.py -q`

Commit: `git commit -m "feat: add capture offline evaluator"`

### Task 7: Eight camera sidecars and labeled set

**Files:**
- Create: `02_configs/surveillance/capture_visibility_v1/ffs_d4.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera08.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera09.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera10.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera11.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera13.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera14.json`
- Create: `02_configs/surveillance/capture_visibility_v1/camera15.json`
- Test: `01_algorithms/live_operator/releases/live-operator-20260915-v180-mage-vl-migration/live_operator/tests/test_capture_visibility_configs.py`
- Runtime-only: `<experiment-root>/manifest.json`, `<experiment-root>/labels.json`, previews, and clips.

- [ ] **Step 1: Write a failing enabled-camera coverage test**

```python
def test_sidecars_cover_enabled_screen_cameras():
    assert set(load_all_sidecars()) == {"ffs_d4", "camera08", "camera09", "camera10", "camera11", "camera13", "camera14", "camera15"}
```

- [ ] **Step 2: Draft sidecars from completed historical clips**

Use no active recording files. Mark only clearly supported capture/blocked zones; leave all other space unknown. Never infer a hard block from image overlap alone.

- [ ] **Step 3: Render and inspect one non-Git preview per camera**

Check source resolution and all polygons. Require a second manual pass before marking a blocked zone verified.

- [ ] **Step 4: Freeze and label at least ten reviewed events per enabled camera when available**

Include each present category; record absent categories as limitations. Do not invent positive captures.

- [ ] **Step 5: Run schema/coverage/leakage tests and commit sidecars only**

Run: `python -m pytest live_operator/tests/test_capture_visibility.py live_operator/tests/test_capture_visibility_configs.py live_operator/tests/test_capture_dataset.py -q`

Commit: `git commit -m "config: map screen capture visibility"`

### Task 8: Complete local verification

**Files:**
- Modify only narrow implementation/test files if a failure proves necessary.

- [ ] **Step 1: Run all new tests**

Run: `python -m pytest live_operator/tests/test_inference_priority.py live_operator/tests/test_capture_visibility.py live_operator/tests/test_capture_policy.py live_operator/tests/test_capture_dataset.py live_operator/tests/test_capture_evidence.py live_operator/tests/test_capture_offline.py live_operator/tests/test_capture_visibility_configs.py -q`

- [ ] **Step 2: Run existing service regressions**

Run: `python -m pytest live_operator/tests/test_mage_vl_service.py live_operator/tests/test_vlm_review.py live_operator/tests/test_vlm_state.py live_operator/tests/test_services_reload.py -q`

- [ ] **Step 3: Run fake-reviewer end-to-end tests**

Run: `python -m pytest live_operator/tests/test_capture_offline.py -q -k 'end_to_end or pause or resume or report'`

- [ ] **Step 4: Check hygiene**

Run: `git diff --check && git status --short`

Verify no video, preview, JSONL, secret, or runtime manifest is staged. Record candidate hashes in the ignored execution log.

### Task 9: Guarded shared-service canary and completed historical test

**Files:**
- Create: `01_algorithms/CAPTURE_OFFLINE_VALIDATION_20260916.md`
- Runtime-only: versioned Mage release/rollback on `.54` and experiment directory on 50-2.

- [ ] **Step 1: Record baselines and hard stops**

Capture camera/FPS/dashboard/alerts, `.54` PID/start token/restarts/listener/GPU processes/memory/swap, signed stage-one health, production arrival/latency, and outstanding reconciliation. Stop if reconciliation is nonzero or quiet time is under 30 seconds.

- [ ] **Step 2: Stage a hashed versioned `.54` release without switching `current`**

Validate ownership, modes, imports, CLI, and stage-one revision constants. Do not load a second model.

- [ ] **Step 3: Guardedly restart only `jiankong-mage-vl-54.service`**

Atomically switch the Mage release, restart only that unit, and require one listener/one GPU PID, unchanged stage-one revisions, new capture revisions, and signed health. Restore the previous release on any failure.

- [ ] **Step 4: Run stage-one compatibility canaries**

Compare approved fixture labels, revisions, signatures, and cache behavior with baseline. Keep offline disabled unless all match.

- [ ] **Step 5: Run five stage-two events at at most two per minute**

Require terminal experiment outcomes while cameras/FPS/alerts and production latency remain within baseline. Any breach pauses without auto-resume.

- [ ] **Step 6: Complete the frozen manifest**

Process one event at a time until every entry is terminal (`completed` or explicit `input_error`). Pending, paused, or absent entries mean incomplete.

- [ ] **Step 7: Generate and inspect the comparison**

Require retention of all available capture-positive cases, no uncertain auto-filters, at least 50% aggregate false-positive reduction without hiding any per-camera regression, evidence for every filter, representation of every enabled camera available in the archive, and acceptable production deltas.

- [ ] **Step 8: Write the redacted receipt and rerun focused verification**

Record revisions, manifest digest, category/camera counts, confusion matrices, reduction/retention, limitations, production deltas, cancellations, pauses, and rollback state without IDs, images, credentials, or private paths.

Run: `git diff --check && python -m pytest live_operator/tests/test_inference_priority.py live_operator/tests/test_capture_policy.py live_operator/tests/test_capture_offline.py live_operator/tests/test_mage_vl_service.py -q`

- [ ] **Step 9: Commit the receipt**

Commit: `git commit -m "docs: record offline capture validation"`

Do not enable shadow mode or production filtering in this plan.

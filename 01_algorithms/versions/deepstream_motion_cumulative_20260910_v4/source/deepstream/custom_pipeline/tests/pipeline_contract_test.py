import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(
    os.environ.get("JIAN_KONG_REPO_ROOT")
    or os.environ.get("JIANKONG_REPO_ROOT")
    or ROOT.parents[1]
).resolve()
EXTRACT_STATIC_FIXTURE = REPO_ROOT / "tools" / "extract_static_phone_regression_fixture.py"
VALIDATE_STATIC_REGRESSION = REPO_ROOT / "tools" / "validate_static_phone_regression.py"
CAMERA04_STATIC_FIXTURE = ROOT / "tests" / "fixtures" / "camera04_person299_static.json"
HEADER = (ROOT / "include" / "deepstream_batch_reader.hpp").read_text(encoding="utf-8")
RUNTIME_HEADER = (ROOT / "include" / "rtsp_live_runtime.hpp").read_text(encoding="utf-8")
READER = (ROOT / "src" / "deepstream_batch_reader.cpp").read_text(encoding="utf-8")
MAIN = (ROOT / "src" / "jiankong_custom_pipeline.cu").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / "config" / "rtsp_7x8.env.example").read_text(encoding="utf-8")
RUNBOOK = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1:index]
    raise AssertionError(f"unterminated function: {signature}")


def _run_python(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-B", str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


STATIC_DIAGNOSTIC_FIELDS = {
    "static_cluster_samples": 7,
    "static_detection_ratio": 1.0,
    "static_center_spread_px": 1.25,
    "static_bbox_iou_median": 0.94,
    "static_pending": False,
    "static_pending_duration": 0.0,
    "static_hotspot_score": 0.0,
    "static_context_reason": "observed",
    "static_shadow_hits": 0,
    "static_suppressed": False,
    "static_exit_reason": "none",
}


def _candidate_static_row(frame_index: int, *, pending: bool, suppressed: bool) -> dict:
    diagnostics = dict(STATIC_DIAGNOSTIC_FIELDS)
    diagnostics.update(
        static_pending=pending,
        static_pending_duration=0.75 if pending else 0.0,
        static_suppressed=suppressed,
        static_context_reason="pending" if pending else "motion_decoupled",
        static_shadow_hits=6 if pending else 0,
    )
    person = {
        "track_id": 299,
        "state": "S2_RISK" if pending else "S0_CLEAR",
        "alarm": False,
        **diagnostics,
    }
    phone = {
        "track_id": 299,
        "accepted": True,
        "risk_score": 0.70,
        **diagnostics,
    }
    return {
        "stream_index": 3,
        "frame_index": frame_index,
        "persons": [person],
        "phones": [phone],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_static_phone_fixture_extractor_redacts_and_normalizes_target_samples() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        source = temp / "frame_events.jsonl"
        output = temp / "fixture.json"
        rows = []
        for frame_index, time_sec, phone_x, risk in (
            (48425, 6053.125, 1100.25, 0.6952715516),
            (48426, 6053.250, 1100.75, 0.7025),
        ):
            rows.append({
                "stream_index": 3,
                "frame_index": frame_index,
                "time_sec": time_sec,
                "persons": [{
                    "track_id": 299,
                    "box": [1049.0, 238.0, 1184.0, 393.0],
                    "alarm": False,
                }],
                "phones": [{
                    "track_id": 299,
                    "phone_id": 987654,
                    "box": [phone_x, 379.0, phone_x + 22.0, 397.0],
                    "risk_score": risk,
                    "accepted": True,
                }],
                "input_video": "must-not-leak.mp4",
            })
        rows.extend([
            {"stream_index": 2, "frame_index": 48425, "persons": [], "phones": []},
            {"stream_index": 3, "frame_index": 48427, "persons": [], "phones": []},
        ])
        _write_jsonl(source, rows)

        result = _run_python(
            EXTRACT_STATIC_FIXTURE,
            "--jsonl", str(source),
            "--stream-index", "3",
            "--track-id", "299",
            "--first-frame", "48425",
            "--last-frame", "48426",
            "--output", str(output),
        )
        assert result.returncode == 0, result.stderr
        fixture = json.loads(output.read_text(encoding="utf-8"))
        assert set(fixture) == {"camera", "track_id", "infer_fps", "expected", "samples"}
        assert fixture["camera"] == "camera04"
        assert fixture["track_id"] == 299
        assert fixture["infer_fps"] == 8.0
        assert fixture["expected"] == {"alarm": False, "static_suppressed": True}
        assert len(fixture["samples"]) == 2
        assert fixture["samples"][0] == {
            "time_sec": 0.0,
            "phone_bbox": [1100.25, 379.0, 1122.25, 397.0],
            "person_bbox": [1049.0, 238.0, 1184.0, 393.0],
            "risk_score": 0.6952715516,
            "accepted": True,
        }
        assert fixture["samples"][1]["time_sec"] == 0.125
        assert "phone_id" not in output.read_text(encoding="utf-8")
        assert "input_video" not in output.read_text(encoding="utf-8")


def test_camera04_static_fixture_is_exactly_the_reviewed_redacted_interval() -> None:
    fixture_text = CAMERA04_STATIC_FIXTURE.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)
    assert set(fixture) == {"camera", "track_id", "infer_fps", "expected", "samples"}
    assert fixture["camera"] == "camera04"
    assert fixture["track_id"] == 299
    assert fixture["infer_fps"] == 8.0
    assert fixture["expected"] == {"alarm": False, "static_suppressed": True}
    assert len(fixture["samples"]) == 73
    assert fixture["samples"][0]["time_sec"] == 0.0
    assert fixture["samples"][-1]["time_sec"] == 9.0
    sample_keys = {"time_sec", "phone_bbox", "person_bbox", "risk_score", "accepted"}
    assert all(set(sample) == sample_keys for sample in fixture["samples"])
    centers = [
        ((sample["phone_bbox"][0] + sample["phone_bbox"][2]) / 2.0,
         (sample["phone_bbox"][1] + sample["phone_bbox"][3]) / 2.0)
        for sample in fixture["samples"]
    ]
    displacement = max(
        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        for x1, y1 in centers
        for x2, y2 in centers
    )
    assert displacement <= 2.0
    for forbidden in ("phone_id", "password", "credential", "input_video", "output_video", "pixels"):
        assert forbidden not in fixture_text


def test_static_phone_candidate_validator_enforces_event_level_contract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        jsonl = Path(temp_dir) / "candidate.jsonl"
        pending = _candidate_static_row(48431, pending=True, suppressed=False)
        suppressed = _candidate_static_row(48449, pending=False, suppressed=True)
        _write_jsonl(jsonl, [pending, suppressed])
        valid = _run_python(
            VALIDATE_STATIC_REGRESSION,
            "--jsonl", str(jsonl),
            "--stream-index", "3",
            "--track-id", "299",
        )
        assert valid.returncode == 0, valid.stderr
        assert "static phone regression passed" in valid.stdout.lower()

        _write_jsonl(jsonl, [suppressed, pending])
        wrong_order = _run_python(
            VALIDATE_STATIC_REGRESSION,
            "--jsonl", str(jsonl),
            "--stream-index", "3",
            "--track-id", "299",
        )
        assert wrong_order.returncode != 0
        assert "before suppression" in wrong_order.stderr.lower()

        alarm = _candidate_static_row(48449, pending=False, suppressed=True)
        alarm["persons"][0]["state"] = "S4_ALARM"
        alarm["persons"][0]["alarm"] = True
        _write_jsonl(jsonl, [pending, alarm])
        s4 = _run_python(
            VALIDATE_STATIC_REGRESSION,
            "--jsonl", str(jsonl),
            "--stream-index", "3",
            "--track-id", "299",
        )
        assert s4.returncode != 0
        assert "s4_alarm" in s4.stderr.lower()

        missing = _candidate_static_row(48449, pending=False, suppressed=True)
        del missing["phones"][0]["static_bbox_iou_median"]
        _write_jsonl(jsonl, [pending, missing])
        incomplete = _run_python(
            VALIDATE_STATIC_REGRESSION,
            "--jsonl", str(jsonl),
            "--stream-index", "3",
            "--track-id", "299",
        )
        assert incomplete.returncode != 0
        assert "static_bbox_iou_median" in incomplete.stderr
RUN_SCRIPT = (ROOT / "scripts" / "run_7x8.sh").read_text(encoding="utf-8")
HOST_BUILD_SCRIPT = (ROOT / "scripts" / "build_container_50p2.sh").read_text(encoding="utf-8")
HOST_RUN_SCRIPT = (ROOT / "scripts" / "run_container_50p2.sh").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "docker" / "Dockerfile.samples-dev").read_text(encoding="utf-8")
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def test_deepstream_is_transport_only() -> None:
    assert 'make_element("nvurisrcbin"' in READER
    assert 'make_element("nvstreammux"' in READER
    assert 'make_element("nvvideoconvert"' in READER
    assert 'make_element("appsink"' in READER
    assert 'make_element("nvinfer"' not in READER


def test_calibration_coordinate_space_is_preserved() -> None:
    assert "config.width != 2560" in READER
    assert "config.height != 1440" in READER
    assert '"nvbuf-memory-type", 2' in READER
    assert "NVBUF_COLOR_FORMAT_RGBA" in READER


def test_gated_screen_intent_uses_calibrated_capture_risk_corridor() -> None:
    body = _function_body(
        MAIN,
        "static jiankong::custom_pipeline::EvidenceState gated_screen_intent(",
    )
    assert "screen.params.angle_thresh" in body
    assert "screen.params.angle_relaxed_thresh" in body
    assert "70.0f" in body
    assert "90.0f" in body
    assert "screen.params.corridor_width_ratio" in body
    assert "screen.params.corridor_width_min" in body
    assert "forearm_direction" in body
    assert "phone_direction" in body
    assert "nearest_wrist + side_slack" in body
    assert "candidate_exact_hit ? 2 : (candidate_corridor_hit ? 1 : 0)" in body
    assert "corridor_ray_hit" in body
    assert "ray_hit_out = exact_ray_hit_out || corridor_ray_hit_out" in body
    assert "angle_out <= 50.0f" not in body
    assert "angle_out <= 55.0f" not in body
    assert 'p["exact_screen_ray_hit"]' in MAIN
    assert 'p["corridor_screen_ray_hit"]' in MAIN
    assert "NVBUF_LAYOUT_PITCH" in READER


def test_live_alarm_uses_approved_raw_phone_confidence_floor() -> None:
    assert "alarm_raw_phone_confidence = 0.75f" in MAIN
    assert "alarm_raw_phone_confidence = 0.80f" not in MAIN


def test_live_runner_disables_both_static_phone_suppression_paths() -> None:
    assert 'ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION="${ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION:-0}"' in RUN_SCRIPT
    assert "--disable-static-phone-suppression" in RUN_SCRIPT


def test_original_event_semantics_are_present() -> None:
    for symbol in (
        "decode_pose",
        "decode_phone_batch",
        "assign_person_track_ids",
        "evaluate_phone",
        "update_person_states",
        "static_phone_suppressed_candidates",
        "write_event_metadata",
    ):
        assert symbol in MAIN


def test_static_phone_suppression_preserves_the_existing_risk_formula() -> None:
    assert "0.35f * phone_score + 0.25f * screen_score + 0.20f * hand_score" in MAIN
    assert "+ 0.10f * pose_score + 0.10f * temporal_score" in MAIN
    assert "static_config_risk_multiplier" in MAIN
    assert "apply_legacy_static_phone_suppression(" in MAIN
    assert "observe_spatial_static_phone(" in MAIN


def test_reviewed_fixed_phone_templates_are_mounted_and_safely_gated() -> None:
    host = (ROOT / "scripts" / "run_container_50p2.sh").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts" / "run_7x8.sh").read_text(encoding="utf-8")
    assert ':/fixed_templates:ro"' in host
    assert "FIXED_TEMPLATE_DIR=/fixed_templates" in host
    assert "--fixed-template-dir" in runner
    assert "--fixed-template-dir" in MAIN
    assert "FixedTemplateEvidence::matched" in MAIN
    assert "FixedTemplateEvidence::mismatch" in MAIN
    assert "fixed_template_mismatch" in MAIN
    assert "cudaMemcpy2D(" in MAIN


def test_spatial_static_policy_precedes_alarm_accumulation() -> None:
    assert '#include "static_phone_spatial_policy.hpp"' in MAIN
    body = _function_body(MAIN, "static void update_person_states(")
    policy_call = body.index("observe_spatial_static_phone(")
    enabled_alarm_update = body.rindex("update_person_alarm_state(")
    assert policy_call < enabled_alarm_update
    assert "state.static_shadow_history" in MAIN
    assert "replay_static_shadow" in MAIN
    assert "discard_static_shadow" in MAIN
    assert "spatial_decision.primary_candidate_index" in body
    assert "dominant_spatial_candidate" not in MAIN
    assert "spatial_primary_box" not in MAIN
    assert "const float static_hotspot_score = 0.0f;" in MAIN
    assert "phone_track_id" not in MAIN


def test_legacy_temporal_alarm_self_test_is_isolated_from_static_suppression() -> None:
    body = _function_body(MAIN, "static int run_rule_self_test()")
    assert "states[people[0].track_id].spatial_static_enabled = false;" in body


def test_spatial_static_cli_defaults_parse_validation_and_config_wiring() -> None:
    args = _function_body(MAIN, "struct Args")
    parse = _function_body(MAIN, "static Args parse_args(")
    validate = _function_body(
        MAIN, "static void validate_spatial_static_args(const Args& args) {")
    config = _function_body(MAIN, "static jiankong::custom_pipeline::SpatialStaticConfig spatial_static_config_from_args(")
    assign = _function_body(MAIN, "static int assign_person_track_ids(")

    expected = {
        "spatial_static_phone_suppression_enabled = true": "--enable-spatial-static-phone-suppression",
        "static_observation_seconds = 3.0": "--static-observation-seconds",
        "static_pending_seconds = 0.75": "--static-pending-seconds",
        "static_long_confirm_seconds = 6.0": "--static-long-confirm-seconds",
        "static_min_detection_ratio = 0.60": "--static-min-detection-ratio",
        "static_max_gap_seconds = 0.75": "--static-max-gap-seconds",
        "static_position_radius_ratio = 0.03": "--static-position-radius-ratio",
        "static_hotspot_enabled = true": "--static-hotspot-enabled",
    }
    for default, option in expected.items():
        assert default in args
        assert option in parse

    assert "--disable-spatial-static-phone-suppression" in parse
    assert "conflicting spatial static phone suppression switches" in parse
    assert "std::isfinite" in validate
    assert "must be finite and positive" in validate
    assert "must be within [0,1]" in validate
    assert "--static-long-confirm-seconds must be >= --static-observation-seconds" in validate
    assert (
        'require_ratio(args.static_position_radius_ratio, '
        '"--static-position-radius-ratio", true);'
    ) in validate
    assert "validate_spatial_static_args(a);" in MAIN
    for member in (
        "short_seconds", "pending_seconds", "long_seconds", "min_detection_ratio",
        "max_gap_seconds", "radius_ratio",
    ):
        assert f"config.{member}" in config
    assert "states.try_emplace(best_id, spatial_static_config)" in assign
    assert "spatial_static_config_from_args(args)" in MAIN


def test_disabled_spatial_static_path_keeps_gated_alarm_behavior() -> None:
    state = _function_body(MAIN, "struct PersonTrackState")
    update = _function_body(MAIN, "static void update_person_states(")
    legacy = _function_body(MAIN, "static void update_person_states_legacy(")
    suppression = _function_body(MAIN, "static void apply_legacy_static_phone_suppression(")
    bypass_start = update.index("if (!spatial_static_enabled)")
    bypass_end = update.index("return;", bypass_start)
    bypass = update[bypass_start:bypass_end]
    policy_call = update.index("observe_spatial_static_phone(")
    assert bypass_start < policy_call
    assert "update_person_states_legacy(" in bypass
    assert "static_phone_history" in state
    assert "apply_legacy_static_phone_suppression(" in legacy
    assert "for (auto& ev : evals)" in legacy
    assert "ev.risk_score > evals[it->second].risk_score" in legacy
    assert "state.risk_history.clear();" in legacy
    assert "state.candidate_history.clear();" in legacy
    assert "state.handheld_history.clear();" in legacy
    assert "push_limited(state.risk_history, risk, window_size);" in legacy
    assert "push_limited(state.candidate_history, candidate ? 1 : 0, window_size);" in legacy
    assert "push_limited(state.handheld_history, handheld_candidate ? 1 : 0, window_size);" in legacy
    assert "state.legacy_alarm_triggered = state.window_hits >= min_hits;" in legacy
    assert "state.gated_event_policy.update(" in legacy
    assert "state.alarm_triggered = state.gated_event_decision.alarm;" in legacy
    assert "gated_event_state_name(" in legacy
    for token in (
        "static_phone_history", "phone_in_lower_person_area", "rect_size_change_ratio",
        "motion_decoupled", "prolonged_lower_hint", "static_phone_suppressed",
    ):
        assert token in suppression
    assert "static_shadow_history" not in bypass
    assert "static_hotspots" not in bypass
    assert "last_static_decision" not in bypass
    assert "live_mode && args.spatial_static_phone_suppression_enabled" in MAIN
    assert "args.static_hotspot_enabled" in MAIN


def test_spatial_static_diagnostics_are_complete_and_bounded_in_both_json_objects() -> None:
    metadata = _function_body(MAIN, "static void write_event_metadata(")
    persons_start = metadata.index('j["persons"] = json::array()')
    phones_start = metadata.index('j["phones"] = json::array()')
    person_json = (
        metadata[persons_start:phones_start]
        + _function_body(MAIN, "static void write_static_diagnostics(")
        + _function_body(MAIN, "static void write_neutral_static_diagnostics(")
    )
    phone_json = metadata[phones_start:]
    fields = (
        "static_cluster_samples",
        "static_detection_ratio",
        "static_center_spread_px",
        "static_bbox_iou_median",
        "static_pending",
        "static_pending_duration",
        "static_hotspot_score",
        "static_context_reason",
        "static_shadow_hits",
        "static_suppressed",
        "static_exit_reason",
    )
    for field in fields:
        token = f'p["{field}"]'
        assert token in person_json
        assert token in phone_json
    assert "spatial_static_context_reason(" in metadata
    context_reason = _function_body(MAIN, "static const char* spatial_static_context_reason(")
    context_allowlist_start = context_reason.index("allowed[]")
    context_allowlist_end = context_reason.index("};", context_allowlist_start)
    context_values = set(re.findall(
        r'"([a-z_]+)"',
        context_reason[context_allowlist_start:context_allowlist_end],
    ))
    assert context_values == {
        "disabled", "not_primary", "observed", "pending",
        "motion_decoupled", "hotspot",
    }
    assert 'std::strcmp(reason, "pending_static_context") == 0' in context_reason
    assert 'std::strcmp(reason, "pending_dropout") == 0' in context_reason
    assert 'return "pending";' in context_reason
    assert 'return "observed";' in context_reason
    assert "return reason;" not in context_reason

    exit_reason = _function_body(MAIN, "static const char* spatial_static_exit_reason(")
    exit_allowlist_start = exit_reason.index("allowed[]")
    exit_allowlist_end = exit_reason.index("};", exit_allowlist_start)
    exit_values = set(re.findall(
        r'"([a-z_]+)"',
        exit_reason[exit_allowlist_start:exit_allowlist_end],
    ))
    assert exit_values == {
        "phone_moving", "follows_wrist", "static_context_timeout",
        "preconfirm_disappearance", "suppressed_disappearance", "invalid_observation",
        "fixed_template_mismatch",
    }
    assert context_values.isdisjoint(exit_values | {"none"})
    assert 'return "none";' in exit_reason

    context_samples = {
        "disabled": "disabled",
        "pending_static_context": "pending",
        "pending_dropout": "pending",
        "motion_decoupled": "motion_decoupled",
        "phone_moving": "observed",
        "follows_wrist": "observed",
        "static_context_timeout": "observed",
        "preconfirm_disappearance": "observed",
        "suppressed_disappearance": "observed",
        "invalid_observation": "observed",
    }
    for reason, expected_context in context_samples.items():
        if reason in {"pending_static_context", "pending_dropout"}:
            actual_context = "pending"
        elif reason in context_values:
            actual_context = reason
        else:
            actual_context = "observed"
        assert actual_context == expected_context

    diagnostics = _function_body(MAIN, "static void update_spatial_static_diagnostics(")
    assert "spatial_static_context_reason(decision.reason)" in diagnostics
    assert "spatial_static_exit_reason(decision.reason)" in diagnostics
    assert "ev->static_context_reason = bounded_context_reason;" in diagnostics
    assert "ev->static_exit_reason = bounded_exit_reason;" in diagnostics


def test_spatial_static_diagnostics_bind_only_to_current_primary_candidate() -> None:
    diagnostics = _function_body(MAIN, "static void update_spatial_static_diagnostics(")
    assert "decision.primary_candidate_index" in diagnostics
    assert "candidate_index == *decision.primary_candidate_index" in diagnostics
    secondary_start = diagnostics.index("if (!is_current_primary)")
    secondary_end = diagnostics.index("continue;", secondary_start)
    secondary = diagnostics[secondary_start:secondary_end]
    for neutral_assignment in (
        "ev->static_cluster_samples = 0;",
        "ev->static_detection_ratio = 0.0f;",
        "ev->static_center_spread_px = 0.0f;",
        "ev->static_bbox_iou_median = 0.0f;",
        "ev->static_pending = false;",
        "ev->static_pending_duration = 0.0;",
        "ev->static_hotspot_score = 0.0f;",
        'ev->static_context_reason = "not_primary";',
        "ev->static_shadow_hits = 0;",
        "ev->static_suppressed = false;",
        'ev->static_exit_reason = "none";',
    ):
        assert neutral_assignment in secondary
    assert "candidate_static_hotspot_score" not in diagnostics

    metadata = _function_body(MAIN, "static void write_event_metadata(")
    persons_start = metadata.index('j["persons"] = json::array()')
    phones_start = metadata.index('j["phones"] = json::array()')
    visible_persons = metadata[persons_start:metadata.index("for (const auto& tid : suspect_tracks)")]
    stale_persons = metadata[metadata.index("for (const auto& tid : suspect_tracks)"):phones_start]
    assert "write_static_diagnostics(p, ev);" in visible_persons
    assert "last_static_decision" not in visible_persons
    assert "write_neutral_static_diagnostics(p, \"not_primary\");" in stale_persons


def test_spatial_static_runtime_script_env_and_runbook_are_reproducible() -> None:
    env_to_cli = {
        "ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION": "--enable-spatial-static-phone-suppression",
        "STATIC_OBSERVATION_SECONDS": "--static-observation-seconds",
        "STATIC_PENDING_SECONDS": "--static-pending-seconds",
        "STATIC_LONG_CONFIRM_SECONDS": "--static-long-confirm-seconds",
        "STATIC_MIN_DETECTION_RATIO": "--static-min-detection-ratio",
        "STATIC_MAX_GAP_SECONDS": "--static-max-gap-seconds",
        "STATIC_POSITION_RADIUS_RATIO": "--static-position-radius-ratio",
        "STATIC_HOTSPOT_ENABLED": "--static-hotspot-enabled",
    }
    for env_name, option in env_to_cli.items():
        assert env_name in RUN_SCRIPT
        assert option in RUN_SCRIPT
        assert env_name in ENV_EXAMPLE
        assert f"-e {env_name}" in HOST_RUN_SCRIPT
    assert "ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION=1" in ENV_EXAMPLE
    assert "ENABLE_SPATIAL_STATIC_PHONE_SUPPRESSION=0" in RUNBOOK
    assert "--disable-spatial-static-phone-suppression" in RUNBOOK
    rollback = RUNBOOK[RUNBOOK.index("One-switch rollback"):]
    assert "bash deepstream/custom_pipeline/scripts/run_container_50p2.sh" in rollback
    assert "bash deepstream/custom_pipeline/scripts/run_7x8.sh" not in rollback


def test_static_discard_uses_primary_cluster_euclidean_membership() -> None:
    state = _function_body(MAIN, "struct PersonTrackState")
    discard = _function_body(MAIN, "static void discard_static_shadow(")
    replay = _function_body(MAIN, "static void replay_static_shadow(")
    update = _function_body(MAIN, "static void update_person_states(")

    assert "formal_spatial_history" in state
    assert "primary_cluster_centers" in discard
    assert "primary_cluster_tolerance_px" in discard
    assert "spatial_point_distance" in discard
    assert "decision" in discard
    assert "sample.valid" in discard
    assert "discard_static_shadow(state, spatial_decision)" in update
    assert "history_offset" in replay
    assert "replay_offset" in replay
    assert "state.risk_history[history_index] = std::max" in replay
    assert "state.candidate_history[history_index]" in replay
    assert "append_formal_history(" not in replay
    assert "formal_is_spatial_primary" not in MAIN
    assert "spatial_primary_history" not in MAIN
    assert "primary_cluster_bounds" not in discard
    assert "sample.center_x <" not in discard
    assert "sample.center_y <" not in discard
    assert "spatial secondary formal history preservation failed" in MAIN
    assert "spatial diagonal secondary formal history preservation failed" in MAIN
    assert "return 22;" in MAIN


def test_live_path_consumes_device_surface_without_upload() -> None:
    assert "DeepStreamBatchReader" in MAIN
    assert "preprocess_rgba_device_frame" in MAIN
    assert "device_frames" in MAIN
    assert "completed_fps" in MAIN


def test_live_events_use_appsink_wall_clock_capture_time() -> None:
    assert "received_at_unix_seconds" in HEADER
    assert "std::chrono::system_clock" in READER
    assert "received_at_unix_seconds" in READER
    assert "LiveWallClockAnchor" in MAIN
    assert "anchor_pts_ns" in MAIN
    assert "anchor_wall_unix_seconds" in MAIN
    assert "last_pts_ns" in MAIN
    assert "frame.pts_ns < anchor.last_pts_ns" in MAIN
    assert "frame.pts_ns - anchor.anchor_pts_ns" in MAIN
    assert "kLiveWallClockMaxDriftSeconds" in MAIN
    assert "std::abs(projected_unix_seconds - frame.received_at_unix_seconds)" in MAIN
    assert "live_wall_clock_anchors[view.source_id]" in MAIN
    assert "live wall-clock PTS anchor self-test failed" in MAIN
    assert "live wall-clock drift re-anchor self-test failed" in MAIN
    assert "live wall-clock PTS reset self-test failed" in MAIN
    assert "format_utc_timestamp" in MAIN
    assert 'j["captured_at"] = format_utc_timestamp(captured_at_unix_seconds);' in MAIN
    assert "frame.received_at_unix_seconds" in MAIN
    live_call = MAIN.index("write_event_metadata(json_records", MAIN.index("live_processed[si] += 1"))
    live_call_end = MAIN.index(");", live_call)
    assert "frame_captured_at_unix_seconds[b]" in MAIN[live_call:live_call_end]


def test_live_wall_clock_reanchors_after_invalid_pts() -> None:
    assert "bool pts_valid" in HEADER
    assert "GST_CLOCK_TIME_IS_VALID(frame->buf_pts)" in READER
    assert "if (!frame.pts_valid)" in MAIN
    assert "anchor.initialized = false;" in MAIN
    assert "invalid PTS uses receive wall-clock self-test failed" in MAIN
    assert "post-invalid PTS re-anchor self-test failed" in MAIN


def test_inner_run_has_no_lock_and_supports_managed_camera_inventory() -> None:
    assert "/tmp/jiankong_gpu0.lock" not in RUN_SCRIPT
    assert "flock" not in RUN_SCRIPT
    assert 'CAMERA_MANIFEST_FILE="${CAMERA_MANIFEST_FILE:-}"' in RUN_SCRIPT
    assert '--camera-manifest "${CAMERA_MANIFEST_FILE}" --relay-base "${RTSP_BASE}"' in RUN_SCRIPT
    # The legacy seven-route list remains only as a rollback-compatible path.
    assert 'camera_args=(' in RUN_SCRIPT
    assert '--rtsp "dianqi1=${RTSP_BASE}/camera01"' in RUN_SCRIPT


def test_inner_run_validates_models_output_and_calibrations() -> None:
    for token in (
        'require_nonempty_file "${POSE_PLAN}" "POSE_ENGINE"',
        'require_nonempty_file "${PHONE_PLAN}" "PHONE_ENGINE"',
        "OUTPUT_NOT_WRITABLE",
        "camera_01_screen_calibration_v21.json",
        "camera_02_screen_calibration_v21.json",
        "camera_mechanical_01_screen_calibration_v21.json",
        "camera_mechanical_02_screen_calibration_v21.json",
        "camera_software_01_screen_calibration_v21.json",
        "camera_software_02_screen_calibration_v21.json",
        "camera_corridor_screen_calibration_v21.json",
        "CAMERA01_CALIBRATION_MUST_HAVE_EXPLICIT_EMPTY_SCREENS",
    ):
        assert token in RUN_SCRIPT


def test_samples_derived_image_contains_build_dependencies() -> None:
    assert "FROM nvcr.io/nvidia/deepstream:7.0-samples-multiarch" in DOCKERFILE
    for package in (
        "build-essential",
        "cmake",
        "pkg-config",
        "libopencv-dev",
        "nlohmann-json3-dev",
        "libgstreamer1.0-dev",
        "libgstreamer-plugins-base1.0-dev",
    ):
        assert package in DOCKERFILE
    assert "${OpenCV_INCLUDE_DIRS}" in CMAKE


def test_samples_derived_image_selects_cuda_toolkit_with_nvcc() -> None:
    assert "ENV CUDA_HOME=/usr/local/cuda-12.2" in DOCKERFILE
    assert 'ENV PATH="${CUDA_HOME}/bin:${PATH}"' in DOCKERFILE


def test_samples_derived_image_restores_removed_opencv_runtime_codecs() -> None:
    assert "--reinstall" in DOCKERFILE
    for package in (
        "libavcodec58",
        "libavutil56",
        "libmpg123-0",
        "libde265-0",
        "libx265-199",
        "libvpx7",
        "libx264-163",
    ):
        assert package in DOCKERFILE


def test_50p2_wrappers_mount_canonical_inputs_and_keep_lock_on_host() -> None:
    for script in (HOST_BUILD_SCRIPT, HOST_RUN_SCRIPT):
        assert "/media/boshi/Data/JianKong" in script
        assert "/media/boshi/Data/00_active_projects/JianKong" not in script
        assert '--runtime="${NVIDIA_RUNTIME:-nvidia}"' in script
        assert "--network host" in script
        assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in script
        assert "env -u USE_NEW_NVSTREAMMUX" in script
        for mount in (
            "/workspace/source",
            "/workspace/build",
            "/models/pose960_static_b7.plan",
            "/models/phone_b16_img512_fp16_trt86.engine",
            "/configs/calibration",
            "/output",
        ):
            assert mount in script
    assert "/tmp/jiankong_gpu0.lock" not in HOST_BUILD_SCRIPT
    assert 'REPO_ROOT="${REPO_ROOT:-}"' in HOST_BUILD_SCRIPT
    assert 'REPO_ROOT="$(cd "${SOURCE_DIR}/../.." && pwd)"' in HOST_BUILD_SCRIPT
    for tool in (
        "tools/extract_static_phone_regression_fixture.py",
        "tools/validate_static_phone_regression.py",
    ):
        assert f'require_path "${{REPO_ROOT}}/{tool}" file' in HOST_BUILD_SCRIPT
    assert '-v "${REPO_ROOT}:/workspace/repo:ro"' in HOST_BUILD_SCRIPT
    assert "-e JIANKONG_REPO_ROOT=/workspace/repo" in HOST_BUILD_SCRIPT
    assert "/tmp/jiankong_gpu0.lock" in HOST_RUN_SCRIPT
    assert "GPU_LOCK_BUSY=" in HOST_RUN_SCRIPT
    assert "flock -n" in HOST_RUN_SCRIPT


def test_live_measurement_has_warmup_and_fixed_window() -> None:
    assert "const bool live_complete_batch" in MAIN
    assert "if (live_complete_batch)" in MAIN
    assert "deepstream_reader->begin_measurement" in MAIN
    assert "live_measurement_stop_at = live_measurement_started_at + args.duration_sec" in MAIN
    assert "now_sec() >= live_measurement_stop_at" in MAIN
    assert 'summary["effective_measurement_seconds"]' in MAIN
    assert 'summary["startup_seconds"]' in MAIN
    assert 'summary["shutdown_seconds"]' in MAIN
    assert "completion_is_within_measurement" in MAIN
    assert 'summary["planned_measurement_seconds"]' in MAIN
    assert 'summary["actual_effective_measurement_seconds"]' in MAIN
    assert "inferred_frames_including_rejected_cutoff_batch" in MAIN
    assert "source->last_downstream_pts = GST_CLOCK_TIME_NONE" in READER
    cutoff = MAIN.index("if (!formal_completion)")
    formal_count = MAIN.index("live_processed[si] += 1", cutoff)
    formal_event = MAIN.index("write_event_metadata(json_records", formal_count)
    assert cutoff < formal_count < formal_event


def test_exact_live_fps_and_inactive_roi_policy_are_wired() -> None:
    assert "live_infer_fps_is_supported(args.infer_fps)" in MAIN
    assert "live mode requires --infer-fps 8 exactly" in MAIN
    roi_guard = MAIN.index("should_build_phone_roi(streams[si].screen_active)")
    roi_loop = MAIN.index("for (const auto& p : people)", roi_guard)
    phone_loop = MAIN.index("for (int start = 0; start < static_cast<int>(roi_jobs.size())", roi_loop)
    assert roi_guard < roi_loop < phone_loop
    inactive_guard = MAIN.index("if (!streams[si].screen_active)")
    inactive_cleanup = MAIN.index("reset_inactive_stream_state(", inactive_guard)
    inactive_continue = MAIN.index("continue;", inactive_cleanup)
    assert inactive_guard < inactive_cleanup < inactive_continue


def test_live_calibration_is_hard_validated() -> None:
    assert "validate_live_calibration" in MAIN
    assert "validate_explicit_zero_screen_calibration" in MAIN
    assert "live calibration missing" in MAIN
    assert "live calibration has no screens" in MAIN
    assert "live calibration has invalid screen polygon" in MAIN
    assert "preserve_invalid_screens" in MAIN
    assert '"camera_01_screen_calibration.json"' not in MAIN
    assert "screen-inactive calibration must explicitly contain zero screens" in MAIN
    assert "if (!streams[si].screen_active)" in MAIN
    assert 'item["screen_active"]' in MAIN
    assert 'item["screen_count"]' in MAIN
    assert 'summary["screen_active_streams"]' in MAIN
    assert 'summary["screen_inactive_streams"]' in MAIN
    assert '"[CALIB_SELECTED] view="' in MAIN
    assert "/media/boshi/Data/JianKong/02_configs/surveillance" in MAIN
    assert "/media/boshi/Data/00_active_projects/JianKong" not in MAIN


def test_source_scoped_errors_do_not_fail_the_global_pipeline() -> None:
    assert "source_id_for_message" in READER
    assert "source-scoped GStreamer error" in READER
    assert "retained for nvurisrcbin reconnect" in READER
    assert "continue;" in READER
    assert 'throw std::runtime_error("DeepStream pipeline error: " + text)' in READER


def test_downstream_loss_is_observable_per_source() -> None:
    for field in ("downstream_frames", "admitted_minus_pulled_upper_bound", "pts_gap_lost", "source_errors"):
        assert field in HEADER
        assert field in READER
        assert field in MAIN
    assert "downstream_lost" not in HEADER
    assert '"backlog_loss_upper_bound"' in MAIN
    assert '"pts_gap_lost_is_estimate"' in MAIN
    assert '"downstream_lost"' not in MAIN
    assert "record_downstream" in READER
    assert "estimate_missing_pts_frames" in HEADER
    assert "runtime_contract_test" in CMAKE
    assert "pipeline_contract_test" in CMAKE


def test_live_latency_statistics_use_a_bounded_window() -> None:
    assert "class BoundedSampleWindow" in RUNTIME_HEADER
    assert "kDefaultCapacity = 4096" in RUNTIME_HEADER
    assert "std::vector<jiankong::BoundedSampleWindow> live_latencies" in MAIN
    assert "std::vector<std::vector<double>> live_latencies" not in MAIN


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} contract tests passed")

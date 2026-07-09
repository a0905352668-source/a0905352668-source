from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCES = [
    ROOT / "tools" / "cpp_full_pipeline_bench.cu",
    ROOT / "tools" / "cpp_full_pipeline_bench.cu.novideo_candidate",
]


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_side_screen_handheld_phone_can_hold_suspect_track():
    for cpp_source in CPP_SOURCES:
        source = read_source(cpp_source)
        assert "handheld_suspect_min_hits" in source, str(cpp_source)
        assert "is_handheld_phone_suspect_candidate" in source, str(cpp_source)
        assert "handheld_phone_hits" in source, str(cpp_source)
        assert "handheld_suspect" in source, str(cpp_source)
        assert "S2_HAND_PHONE" in source, str(cpp_source)


def test_frame_events_include_rule_debug_scores_for_phone_boxes():
    for cpp_source in CPP_SOURCES:
        source = read_source(cpp_source)
        for field in [
            'p["reject_reason"]',
            'p["candidate_reason"]',
            'p["zone_reason"]',
            'p["static_zone_score"]',
            'p["person_match_score"]',
            'p["phone_score"]',
            'p["phone_hand_score"]',
            'p["screen_relation_score"]',
            'p["pose_score"]',
            'p["temporal_score"]',
        ]:
            assert field in source, f"{field} missing in {cpp_source}"

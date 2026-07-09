from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCES = [
    ROOT / "tools" / "cpp_full_pipeline_bench.cu",
    ROOT / "tools" / "cpp_full_pipeline_bench.cu.novideo_candidate",
]


def check_source(cpp_source: Path):
    source = cpp_source.read_text(encoding="utf-8", errors="replace")

    assert "rect_diag(screen_box) * 0.80f" not in source, str(cpp_source)
    assert "screen_distance_max_px" in source, str(cpp_source)
    assert "screen_distance_min_px" in source, str(cpp_source)
    assert "clampf(screen_distance_base" in source, str(cpp_source)
    assert "return {0.25f, \"screen_distance\"};" in source, str(cpp_source)


def test_screen_distance_no_longer_scales_with_calibrated_screen_size():
    for cpp_source in CPP_SOURCES:
        check_source(cpp_source)


if __name__ == "__main__":
    test_screen_distance_no_longer_scales_with_calibrated_screen_size()
    print("ok")

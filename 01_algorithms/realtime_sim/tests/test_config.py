import json
import tempfile
import unittest
from pathlib import Path

from realtime_sim.config import VIEW_ORDER, ViewSpec, default_runtime_config, select_videos_by_rank_from_end, view_key_from_path
from realtime_sim.segment_producer import batch_count_for_views, write_source_manifest


class ConfigTests(unittest.TestCase):
    def test_view_order_has_seven_streams(self):
        self.assertEqual(
            VIEW_ORDER,
            ["dianqi1", "dianqi2", "jixie1", "jixie2", "ruanjian1", "ruanjian2", "zoulang"],
        )

    def test_view_key_from_path_matches_current_folder_names(self):
        self.assertEqual(view_key_from_path(Path("/x/dianqi1_192.168.210.148/a.mkv")), "dianqi1")
        self.assertEqual(view_key_from_path(Path("/x/ruanjian2_192.168.210.136/a.mkv")), "ruanjian2")
        self.assertEqual(view_key_from_path(Path("/x/zoulang_192.168.210.145/a.mkv")), "zoulang")

    def test_default_paths_point_inside_project(self):
        cfg = default_runtime_config()
        self.assertEqual(cfg.project_root, Path("/media/boshi/Data/JianKong"))
        self.assertEqual(cfg.cpp_binary.name, "cpp_full_pipeline_bench_gpu")
        self.assertEqual(cfg.infer_fps, 8.0)
        self.assertEqual(cfg.segment_seconds, 10.0)
        self.assertNotIn("latest_completed", str(cfg.phone_plan))
        self.assertIn("phone_models/versions/", str(cfg.phone_plan))
        self.assertEqual(cfg.phone_plan.name, "best_trt86_50_2_b16_fp16.engine")
        self.assertEqual(cfg.phone_conf, 0.5)

    def test_write_source_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = default_runtime_config()
            view = ViewSpec(
                stream_index=0,
                view_key="dianqi1",
                view_label="电气1",
                source_folder=root / "dianqi1_192.168.210.148",
                source_video=root / "dianqi1_192.168.210.148" / "input.mkv",
                calibration_file=cfg.calib_dir / "camera_01_screen_calibration_v21.json",
            )
            path = write_source_manifest(root, [view], 10.0)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["segment_seconds"], 10.0)
            self.assertEqual(data["views"][0]["view_key"], "dianqi1")

    def test_select_videos_by_rank_from_end_picks_same_group_across_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for view in VIEW_ORDER:
                folder = root / f"{view}_cam"
                folder.mkdir()
                for group in (1, 2, 3):
                    path = folder / f"20260708_group{group}_{view}.mkv"
                    path.write_text("x", encoding="utf-8")
            cfg = default_runtime_config()

            latest = select_videos_by_rank_from_end(root, cfg, rank_from_end=1)
            oldest_of_three = select_videos_by_rank_from_end(root, cfg, rank_from_end=3)

            self.assertTrue(all("group3" in str(v.source_video) for v in latest))
            self.assertTrue(all("group1" in str(v.source_video) for v in oldest_of_three))

    def test_batch_count_for_views_uses_shortest_video_duration(self):
        cfg = default_runtime_config()
        views = [
            ViewSpec(0, "dianqi1", "dianqi1", Path("/x/a"), Path("/x/a/1.mkv"), cfg.calib_dir / "a.json"),
            ViewSpec(1, "dianqi2", "dianqi2", Path("/x/b"), Path("/x/b/1.mkv"), cfg.calib_dir / "b.json"),
        ]
        import realtime_sim.segment_producer as producer
        original = producer.estimate_video_duration_sec
        try:
            producer.estimate_video_duration_sec = lambda p: 21.2 if "a" in str(p) else 35.0
            self.assertEqual(batch_count_for_views(views, 10.0), 3)
        finally:
            producer.estimate_video_duration_sec = original


if __name__ == "__main__":
    unittest.main()

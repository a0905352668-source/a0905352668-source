import unittest
from pathlib import Path, PurePosixPath

from realtime_sim.run_realtime_sim import build_clip_builder_command, build_clip_worker_command, parse_args


class RunRealtimeSimTests(unittest.TestCase):
    def test_build_clip_builder_command_uses_selected_python(self):
        cmd = build_clip_builder_command(
            PurePosixPath("/runs/realtime_sim_x"),
            PurePosixPath("/opt/python/bin/python"),
            Path("/media/boshi/Data/JianKong/01_algorithms"),
        )
        self.assertEqual(cmd[0], "/opt/python/bin/python")
        self.assertIn("-m", cmd)
        self.assertIn("realtime_sim.clip_builder", cmd)
        self.assertIn("/runs/realtime_sim_x", cmd)

    def test_build_clip_worker_command_uses_selected_python(self):
        cmd = build_clip_worker_command(
            PurePosixPath("/runs/realtime_sim_x"),
            PurePosixPath("/opt/python/bin/python"),
            Path("/media/boshi/Data/JianKong/01_algorithms"),
        )
        self.assertEqual(cmd[0], "/opt/python/bin/python")
        self.assertIn("realtime_sim.clip_worker", cmd)
        self.assertIn("--until-complete", cmd)

    def test_parse_args_defaults_to_metadata_first_realtime_path(self):
        args = parse_args([])
        self.assertEqual(args.infer_fps, None)
        self.assertTrue(args.no_cpp_video)
        self.assertEqual(args.clip_build_mode, "background")
        self.assertEqual(args.clip_python, "/home/boshi/miniconda3/bin/python")

    def test_dashboard_service_passes_enterprise_realtime_defaults(self):
        script = Path("realtime_sim/run_dashboard_service.sh").read_text(encoding="utf-8")
        self.assertIn('INFER_FPS="${INFER_FPS:-8}"', script)
        self.assertIn('CLIP_BUILD_MODE="${CLIP_BUILD_MODE:-background}"', script)
        self.assertIn('NO_CPP_VIDEO="${NO_CPP_VIDEO:-1}"', script)
        self.assertIn('--infer-fps "$INFER_FPS"', script)
        self.assertIn('--clip-build-mode "$CLIP_BUILD_MODE"', script)
        self.assertIn('/home/boshi/miniconda3/bin/python', script)


if __name__ == "__main__":
    unittest.main()

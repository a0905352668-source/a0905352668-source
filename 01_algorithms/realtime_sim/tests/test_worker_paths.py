import unittest

from pathlib import Path

from realtime_sim.config import default_runtime_config
from realtime_sim.cpp_worker import build_cpp_command, expected_stream_file_count, parse_cpp_summary, view_folder_name


class WorkerPathTests(unittest.TestCase):
    def test_view_folder_name_contains_view_key(self):
        self.assertEqual(view_folder_name("dianqi1"), "dianqi1_live")
        self.assertEqual(view_folder_name("zoulang"), "zoulang_live")

    def test_expected_stream_file_count(self):
        batch = {"files": [{"view_key": "dianqi1"}, {"view_key": "dianqi2"}]}
        self.assertEqual(expected_stream_file_count(batch), 2)

    def test_build_cpp_command_can_disable_video_writer(self):
        cfg = default_runtime_config()
        batch = {"duration_sec": 10.0}
        cmd = build_cpp_command(cfg, Path("/tmp/root"), Path("/tmp/out"), batch, no_video=True)
        self.assertIn("--no-video", cmd)
        self.assertIn("--output-dir", cmd)
        self.assertEqual(cmd[cmd.index("--phone-plan") + 1], str(cfg.phone_plan))
        self.assertNotIn("latest_completed", cmd[cmd.index("--phone-plan") + 1])
        self.assertEqual(cmd[cmd.index("--phone-conf") + 1], "0.5")

    def test_parse_cpp_summary_extracts_fps_and_counts(self):
        text = "[SUMMARY] frames=707 wall_sec=12.5 aggregate_fps=56.4 persons=10 accepted=3 alarm_frames=2\n"
        parsed = parse_cpp_summary(text)
        self.assertEqual(parsed["frames"], 707)
        self.assertEqual(parsed["aggregate_fps"], 56.4)
        self.assertEqual(parsed["accepted"], 3)


if __name__ == "__main__":
    unittest.main()

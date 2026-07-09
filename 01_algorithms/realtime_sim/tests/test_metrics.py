import json
import tempfile
import unittest
from pathlib import Path

from realtime_sim.metrics import append_batch_metrics, build_batch_metrics


class MetricsTests(unittest.TestCase):
    def test_build_batch_metrics_records_metadata_first_runtime(self):
        metrics = build_batch_metrics(
            batch={"batch_index": 4, "duration_sec": 10.0},
            elapsed_sec=1.25,
            worker_summary={"cpp_summary": {"aggregate_fps": 56.4, "frames": 70, "persons": 8, "accepted": 2}},
            state={"event_count": 3, "segment_count": 1, "alarm_event_count": 1},
            infer_fps=8.0,
            phone_conf=0.5,
            pose_plan=Path("/models/pose.plan"),
            phone_plan=Path("/models/phone.engine"),
            no_video=True,
            clip_build_mode="background",
        )

        self.assertEqual(metrics["schema_version"], "jk-runtime-metrics-v1")
        self.assertEqual(metrics["batch_index"], 4)
        self.assertEqual(metrics["aggregate_fps"], 56.4)
        self.assertEqual(metrics["realtime_path"], "metadata_only")
        self.assertEqual(metrics["clip_build_mode"], "background")
        self.assertEqual(metrics["phone_model_id"], "phone.engine")

    def test_append_batch_metrics_writes_jsonl_and_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metric = append_batch_metrics(
                run_dir=run_dir,
                batch={"batch_index": 1, "duration_sec": 10.0},
                elapsed_sec=2.0,
                worker_summary={"cpp_summary": {"aggregate_fps": 42.0}},
                state={},
                infer_fps=8.0,
                phone_conf=0.5,
                pose_plan=Path("pose.plan"),
                phone_plan=Path("phone.engine"),
                no_video=True,
                clip_build_mode="background",
            )

            lines = (run_dir / "events" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            latest = json.loads((run_dir / "events" / "metrics_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), metric)
            self.assertEqual(latest["aggregate_fps"], 42.0)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from realtime_sim.aggregator import merge_segments, normalize_json_event, parse_event_rows


class AggregatorTests(unittest.TestCase):
    def test_merge_segments_groups_close_events_per_stream(self):
        events = [
            {"stream_index": 0, "timestamp": 10.0, "level": "risk", "risk": 0.6, "output_video": "a.mp4"},
            {"stream_index": 0, "timestamp": 12.0, "level": "alarm", "risk": 0.9, "output_video": "a.mp4"},
            {"stream_index": 1, "timestamp": 12.0, "level": "risk", "risk": 0.7, "output_video": "b.mp4"},
        ]
        segments = merge_segments(events, gap_seconds=4.0, pad_seconds=2.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["level"], "alarm")
        self.assertEqual(segments[0]["event_count"], 2)

    def test_normalize_json_event_records_batch_identity(self):
        row = normalize_json_event(
            {
                "stream_index": 2,
                "frame_index": 16,
                "time_sec": 2.0,
                "max_risk": 0.6,
                "persons": [{"track_id": 5, "risk": True, "risk_score": 0.6}],
            },
            {"batch_index": 3, "batch_name": "batch_000003", "global_start_sec": 30.0},
        )

        self.assertEqual(row["batch_index"], 3)
        self.assertEqual(row["source_batch"], "batch_000003")
        self.assertEqual(row["timestamp"], 32.0)

    def test_merge_segments_keeps_source_batches_for_metadata_only_playback(self):
        events = [
            {"stream_index": 0, "timestamp": 10.0, "level": "risk", "risk": 0.6, "output_video": "", "source_batch": "batch_000001"},
            {"stream_index": 0, "timestamp": 12.0, "level": "risk", "risk": 0.7, "output_video": "", "source_batch": "batch_000002"},
        ]

        segments = merge_segments(events, gap_seconds=4.0, pad_seconds=2.0)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["source_batches"], ["batch_000001", "batch_000002"])

    def test_parse_event_rows_reads_jsonl_suspect_track_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp)
            video_dir = processed / "videos"
            video_dir.mkdir()
            (video_dir / "frame_events.jsonl").write_text(
                json.dumps({
                    "stream_index": 0,
                    "frame_index": 42,
                    "time_sec": 4.2,
                    "max_risk": 0.71,
                    "output_video": "boxed.mp4",
                    "persons": [{
                        "track_id": 7,
                        "risk": True,
                        "suspect": True,
                        "risk_score": 0.71,
                        "box": [10, 20, 50, 80],
                    }],
                }) + "\n",
                encoding="utf-8",
            )

            rows = parse_event_rows(processed, {"global_start_sec": 100.0})

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["track_id"], 7)
            self.assertEqual(rows[0]["timestamp"], 104.2)
            self.assertEqual(rows[0]["risk"], 0.71)


if __name__ == "__main__":
    unittest.main()

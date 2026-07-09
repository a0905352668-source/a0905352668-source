import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from realtime_sim.clip_builder import build_h264_transcode_command, build_missing_clips, draw_event_overlay, merge_clip_updates


class ClipBuilderTests(unittest.TestCase):
    def test_h264_transcode_command_uses_browser_compatible_codec(self):
        cmd = build_h264_transcode_command(PurePosixPath("/tmp/in.mp4"), PurePosixPath("/tmp/out.mp4"))
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("+faststart", cmd)
        self.assertEqual(str(cmd[-1]), "/tmp/out.mp4")

    def test_existing_ready_clip_clears_stale_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "events").mkdir()
            (run_dir / "dashboard" / "clips").mkdir(parents=True)
            (run_dir / "source_manifest.json").write_text(json.dumps({"views": []}), encoding="utf-8")
            (run_dir / "dashboard" / "clips" / "event_000001.mp4").write_bytes(b"fake")
            segments = {
                "segments": [{
                    "id": "event_000001",
                    "clip": "clips/event_000001.mp4",
                    "clip_status": "failed",
                    "clip_error": "old error",
                }]
            }
            (run_dir / "events" / "segments.json").write_text(json.dumps(segments), encoding="utf-8")

            build_missing_clips(run_dir)

            saved = json.loads((run_dir / "events" / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["segments"][0]["clip_status"], "ready")
            self.assertNotIn("clip_error", saved["segments"][0])

    def test_merge_clip_updates_keeps_newer_segments(self):
        stale_payload = {
            "segments": [{
                "id": "event_000001",
                "clip": "clips/event_000001.mp4",
                "clip_status": "ready",
            }]
        }
        latest_payload = {
            "segments": [
                {"id": "event_000001", "clip_status": "pending"},
                {"id": "event_000002", "clip_status": "pending"},
            ]
        }
        merged = merge_clip_updates(stale_payload, latest_payload)
        self.assertEqual(len(merged["segments"]), 2)
        self.assertEqual(merged["segments"][0]["clip_status"], "ready")
        self.assertEqual(merged["segments"][1]["clip_status"], "pending")

    def test_overlay_draws_suspect_person_without_phone_hit(self):
        class FakeCV2:
            FONT_HERSHEY_SIMPLEX = 0
            LINE_AA = 16

            def __init__(self):
                self.rectangles = []
                self.labels = []

            def rectangle(self, frame, p1, p2, color, thickness):
                self.rectangles.append((p1, p2, color, thickness))

            def putText(self, frame, text, org, font, scale, color, thickness, line_type):
                self.labels.append(text)

            def getTextSize(self, text, font, scale, thickness):
                return ((len(text) * 8, 12), 0)

            def polylines(self, *args, **kwargs):
                pass

        fake = FakeCV2()
        draw_event_overlay(fake, object(), {
            "persons": [{
                "track_id": 3,
                "suspect": True,
                "risk": False,
                "alarm": False,
                "risk_score": 0.58,
                "roi": [10, 20, 60, 100],
            }],
            "phones": [],
        })

        self.assertTrue(fake.rectangles)
        self.assertTrue(any("Person3" in text for text in fake.labels))


if __name__ == "__main__":
    unittest.main()

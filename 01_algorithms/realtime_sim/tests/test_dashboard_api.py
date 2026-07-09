import json
import tempfile
import unittest
from pathlib import Path

from realtime_sim.dashboard_live import write_dashboard_api


class DashboardApiTests(unittest.TestCase):
    def test_write_dashboard_api_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001"}]}),
                encoding="utf-8",
            )
            write_dashboard_api(tmp_path)
            self.assertTrue((tmp_path / "dashboard" / "api" / "run_state.json").exists())
            self.assertTrue((tmp_path / "dashboard" / "api" / "segments.json").exists())
            self.assertTrue((tmp_path / "dashboard" / "index.html").exists())

    def test_dashboard_uses_chinese_labels_for_status_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(
                json.dumps({"status": "complete", "event_count": 1}),
                encoding="utf-8",
            )
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "level": "alarm", "clip_status": "ready"}]}),
                encoding="utf-8",
            )
            write_dashboard_api(tmp_path)
            html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
            self.assertIn("状态文本", html)
            self.assertIn("等级文本", html)
            self.assertIn("报警", html)
            self.assertIn("可播放", html)
            self.assertIn("事件总数", html)
            self.assertIn("事件筛选", html)
            self.assertIn("队列状态", html)
            self.assertIn("检测吞吐", html)
            self.assertIn("全部", html)
            self.assertIn("报警", html)
            self.assertIn("疑似", html)

    def test_write_dashboard_api_marks_existing_clips_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "dashboard" / "clips").mkdir(parents=True)
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "clip_status": "pending"}]}),
                encoding="utf-8",
            )
            (tmp_path / "dashboard" / "clips" / "event_000001.mp4").write_bytes(b"not-empty")

            write_dashboard_api(tmp_path)

            payload = json.loads((tmp_path / "dashboard" / "api" / "segments.json").read_text(encoding="utf-8"))
            segment = payload["segments"][0]
            self.assertEqual(segment["clip_status"], "ready")
            self.assertEqual(segment["clip"], "clips/event_000001.mp4")
            self.assertEqual(segment["boxed_video"], "clips/event_000001.mp4")

    def test_write_dashboard_api_maps_segments_to_batch_media_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            batch_dir = tmp_path / "incoming" / "batch_000001"
            processed_dir = tmp_path / "processed" / "batch_000001" / "videos"
            batch_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "source_manifest.json").write_text(json.dumps({"segment_seconds": 10.0}), encoding="utf-8")
            (batch_dir / "stream_2_jixie1.mp4").write_bytes(b"not-empty")
            (batch_dir / "batch.json").write_text(
                json.dumps({"batch_index": 1, "global_start_sec": 10.0, "duration_sec": 10.0}),
                encoding="utf-8",
            )
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "stream_index": 2, "start": 12.0, "end": 16.0}]}),
                encoding="utf-8",
            )
            (processed_dir / "frame_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "stream_index": 2,
                                "frame_index": 20,
                                "time_sec": 2.0,
                                "width": 2560,
                                "height": 1440,
                                "persons": [
                                    {
                                        "track_id": 7,
                                        "suspect": True,
                                        "risk": True,
                                        "alarm": False,
                                        "risk_score": 0.74,
                                        "box": [10, 20, 110, 220],
                                    },
                                    {
                                        "track_id": 8,
                                        "suspect": False,
                                        "risk": False,
                                        "alarm": False,
                                        "risk_score": 0.0,
                                        "box": [300, 300, 400, 400],
                                    },
                                ],
                                "phones": [
                                    {
                                        "track_id": 7,
                                        "accepted": True,
                                        "alarm": False,
                                        "confidence": 0.66,
                                        "box": [40, 50, 70, 90],
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "stream_index": 2,
                                "frame_index": 30,
                                "time_sec": 3.0,
                                "width": 2560,
                                "height": 1440,
                                "persons": [
                                    {
                                        "track_id": 7,
                                        "suspect": False,
                                        "risk": False,
                                        "alarm": False,
                                        "risk_score": 0.0,
                                        "box": [12, 22, 112, 222],
                                    }
                                ],
                                "phones": [],
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            write_dashboard_api(tmp_path)

            payload = json.loads((tmp_path / "dashboard" / "api" / "segments.json").read_text(encoding="utf-8"))
            segment = payload["segments"][0]
            self.assertEqual(segment["clip_status"], "ready")
            self.assertEqual(segment["media_status"], "batch_ready")
            self.assertEqual(segment["playback_video"], "media/batch_000001/stream_2_jixie1.mp4")
            self.assertEqual(segment["playback_start"], 2.0)
            self.assertEqual(segment["overlay"], "api/overlays/batch_000001_stream_2.json")
            self.assertEqual(segment["target_track_id"], 7)
            self.assertTrue((tmp_path / "dashboard" / "media" / "batch_000001" / "stream_2_jixie1.mp4").exists())

            overlay = json.loads(
                (tmp_path / "dashboard" / "api" / "overlays" / "batch_000001_stream_2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(overlay["schema_version"], "jk-overlay-v1")
            self.assertEqual(overlay["coordinate_space"], "pixel_with_normalized_copy")
            self.assertEqual(overlay["width"], 2560)
            self.assertEqual(overlay["height"], 1440)
            self.assertEqual(len(overlay["frames"]), 2)
            self.assertEqual(len(overlay["frames"][0]["persons"]), 1)
            self.assertEqual(overlay["frames"][0]["persons"][0]["track_id"], 7)
            self.assertEqual(overlay["frames"][0]["persons"][0]["box_norm"], [0.003906, 0.013889, 0.042969, 0.152778])
            self.assertEqual(len(overlay["frames"][0]["phones"]), 1)
            self.assertEqual(overlay["frames"][1]["persons"][0]["track_id"], 7)
            self.assertEqual(overlay["frames"][1]["persons"][0]["status"], "track")

    def test_write_dashboard_api_publishes_runtime_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
            (tmp_path / "events" / "metrics.jsonl").write_text(
                json.dumps({"batch_index": 1, "aggregate_fps": 56.2, "realtime_path": "metadata_only"}) + "\n",
                encoding="utf-8",
            )

            write_dashboard_api(tmp_path)

            metrics = json.loads((tmp_path / "dashboard" / "api" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["schema_version"], "jk-runtime-metrics-v1")
            self.assertEqual(metrics["latest"]["aggregate_fps"], 56.2)
            self.assertEqual(metrics["latest"]["realtime_path"], "metadata_only")

    def test_dashboard_prefers_event_clip_so_new_event_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "dashboard" / "clips").mkdir(parents=True)
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "stream_index": 0, "start": 12.0, "end": 16.0}]}),
                encoding="utf-8",
            )
            (tmp_path / "dashboard" / "clips" / "event_000001.mp4").write_bytes(b"not-empty")

            write_dashboard_api(tmp_path)

            html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
            clip_branch = "if(seg.clip){parts.push({video:seg.clip,start:0"
            batch_branch = "seg.playback_parts||[]"
            self.assertIn(clip_branch, html)
            self.assertIn(batch_branch, html)
            self.assertLess(html.index(clip_branch), html.index(batch_branch))

    def test_dashboard_html_contains_canvas_overlay_playback_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 0}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(json.dumps({"segments": []}), encoding="utf-8")

            write_dashboard_api(tmp_path)

            html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
            self.assertIn("overlayCanvas", html)
            self.assertIn("loadOverlay", html)
            self.assertIn("drawOverlay", html)
            self.assertIn("drawBox(ctx,ph,", html)
            self.assertNotIn("drawBox(ctx,ph.box", html)
            self.assertIn("playback_video", html)
            self.assertIn("坐标叠加", html)
            self.assertIn("OVERLAY_HOLD_SECONDS=2.0", html)

    def test_dashboard_html_preserves_selected_video_across_poll_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "stream_index": 0, "start": 0, "end": 10}]}),
                encoding="utf-8",
            )

            write_dashboard_api(tmp_path)

            html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
            self.assertIn("selectedSegmentId", html)
            self.assertIn("video.dataset.source", html)
            self.assertIn("sameSelection", html)
            self.assertIn("userInitiated", html)
            self.assertIn("enforcePlaybackWindow", html)
            self.assertNotIn("video.src=part.video+'?t='", html)
            self.assertNotIn("video.src=seg.clip+'?t='", html)

    def test_dashboard_html_restarts_selected_event_without_blue_tracking_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "events").mkdir()
            (tmp_path / "events" / "state.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
            (tmp_path / "events" / "segments.json").write_text(
                json.dumps({"segments": [{"id": "event_000001", "stream_index": 0, "start": 0, "end": 10}]}),
                encoding="utf-8",
            )

            write_dashboard_api(tmp_path)

            html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
            self.assertIn("el.onclick=()=>select(rows[i],true)", html)
            self.assertIn("if(!sameSelection||userInitiated)", html)
            self.assertIn("主角", html)
            self.assertNotIn("p.status==='track'?'#38bdf8'", html)
            self.assertNotIn("'跟踪'", html)


if __name__ == "__main__":
    unittest.main()

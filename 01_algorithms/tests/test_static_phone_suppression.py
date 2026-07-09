import unittest

from surveillance_static_phone_suppression import (
    StaticPhoneConfig,
    StaticPhoneObservation,
    StaticPhoneSuppressionState,
    load_desk_static_zones_from_data,
)


class StaticPhoneSuppressionTests(unittest.TestCase):
    def test_stationary_phone_in_desk_zone_is_suppressed(self):
        state = StaticPhoneSuppressionState(StaticPhoneConfig(window_seconds=1.5, max_disp_ratio=0.03))
        zones = [[(90, 140), (230, 140), (230, 260), (90, 260)]]
        result = None
        for i, ts in enumerate([0.0, 0.5, 1.0, 1.5, 1.75]):
            result = state.update(
                StaticPhoneObservation(
                    camera_id="cam1",
                    person_id=7,
                    frame_id=i,
                    timestamp=ts,
                    phone_center=(150 + (i % 2), 180),
                    phone_bbox=(135 + (i % 2), 165, 165 + (i % 2), 195),
                    person_bbox=(80, 40, 260, 300),
                    nearest_wrist=(148, 178),
                    risk_score=0.82,
                    candidate=True,
                ),
                desk_static_zones=zones,
            )
        self.assertIsNotNone(result)
        self.assertTrue(result.phone_static)
        self.assertTrue(result.static_suppressed)
        self.assertTrue(result.phone_in_desk_zone)
        self.assertGreaterEqual(result.phone_static_duration, 1.5)
        self.assertLess(result.phone_motion_px, 6.0)

    def test_phone_following_wrist_is_not_suppressed(self):
        state = StaticPhoneSuppressionState(StaticPhoneConfig(window_seconds=1.5, max_disp_ratio=0.03))
        zones = [[(0, 0), (400, 0), (400, 400), (0, 400)]]
        result = None
        for i, ts in enumerate([0.0, 0.5, 1.0, 1.5]):
            dx = i * 18.0
            result = state.update(
                StaticPhoneObservation(
                    camera_id="cam1",
                    person_id=7,
                    frame_id=i,
                    timestamp=ts,
                    phone_center=(120 + dx, 190),
                    phone_bbox=(108 + dx, 176, 132 + dx, 204),
                    person_bbox=(80, 40, 280, 320),
                    nearest_wrist=(118 + dx, 188),
                    risk_score=0.85,
                    candidate=True,
                ),
                desk_static_zones=zones,
            )
        self.assertIsNotNone(result)
        self.assertFalse(result.phone_static)
        self.assertFalse(result.static_suppressed)
        self.assertTrue(result.phone_follow_wrist)

    def test_missing_desk_zone_falls_back_to_lower_person_bbox(self):
        state = StaticPhoneSuppressionState(StaticPhoneConfig(window_seconds=1.5, max_disp_ratio=0.03))
        result = None
        for i, ts in enumerate([0.0, 0.5, 1.0, 1.5]):
            result = state.update(
                StaticPhoneObservation(
                    camera_id="cam1",
                    person_id=9,
                    frame_id=i,
                    timestamp=ts,
                    phone_center=(170, 245),
                    phone_bbox=(156, 232, 184, 258),
                    person_bbox=(100, 80, 240, 300),
                    nearest_wrist=None,
                    risk_score=0.75,
                    candidate=True,
                ),
                desk_static_zones=[],
            )
        self.assertIsNotNone(result)
        self.assertTrue(result.phone_static)
        self.assertTrue(result.static_suppressed)
        self.assertFalse(result.phone_in_desk_zone)

    def test_loads_top_level_and_screen_level_desk_zones(self):
        data = {
            "frame_size": [1000, 500],
            "desk_static_zone": [[10, 10], [100, 10], [100, 100], [10, 100]],
            "screens": [
                {"screen_id": "s1", "desk_static_zones": [{"polygon": [[200, 100], [300, 100], [300, 200], [200, 200]]}]}
            ],
        }
        zones = load_desk_static_zones_from_data(data, frame_width=2000, frame_height=1000, field="desk_static_zones")
        self.assertEqual(len(zones), 2)
        self.assertEqual(zones[0][0], (20.0, 20.0))
        self.assertEqual(zones[1][0], (400.0, 200.0))


if __name__ == "__main__":
    unittest.main()

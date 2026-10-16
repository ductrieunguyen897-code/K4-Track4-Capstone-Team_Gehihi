"""Synthetic tracker contract tests; no detector, weights, or video required."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.tracker import LaneTracker


def lane(x=100, ys=(100, 200, 300), confidence=None, lane_index=0):
    """Build a detector lane with constant x and configurable y samples.

    Args:
        x: Horizontal pixel coordinate.
        ys: Vertical pixel coordinates.
        confidence: Detector confidence or None.
        lane_index: Per-frame detector index.

    Returns:
        A complete detector lane dictionary.
    """
    return {"lane_index": lane_index, "points": [[x, y] for y in ys],
            "confidence": confidence}


def frame(*lanes, width=640, height=360):
    """Build a detector result.

    Args:
        *lanes: Detector lane dictionaries.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A complete detector output dictionary.
    """
    return {"lanes": list(lanes), "frame_width": width,
            "frame_height": height, "inference_ms": 20.0}


class LaneTrackerTests(unittest.TestCase):
    """Exercise public behavior using independent synthetic observations."""

    def test_initial_contract_and_empty_frame(self):
        """Return complete, JSON-compatible output for zero and two lanes."""
        tracker = LaneTracker()
        self.assertEqual(tracker.update(frame())["lanes"], [])
        result = tracker.update(frame(lane(), lane(400)))
        self.assertEqual(set(result), {"lanes", "frame_width", "frame_height", "tracking_ms"})
        self.assertEqual((result["frame_width"], result["frame_height"]), (640, 360))
        self.assertGreaterEqual(result["tracking_ms"], 0)
        self.assertEqual([t["track_id"] for t in result["lanes"]], [1, 2])
        for track in result["lanes"]:
            self.assertEqual(set(track), {"track_id", "points", "confidence", "predicted", "missed"})
            self.assertIs(track["predicted"], False)
            self.assertEqual(track["missed"], 0)
            self.assertIsNone(track["confidence"])
        json.dumps(result, allow_nan=False)

    def test_ema_and_detection_order_do_not_change_identity(self):
        """Match by geometry and apply the specified EMA weight."""
        tracker = LaneTracker(alpha=0.35)
        tracker.update(frame(lane(), lane(400, lane_index=1)))
        tracks = tracker.update(frame(lane(403), lane(103, lane_index=1)))["lanes"]
        self.assertEqual([t["track_id"] for t in tracks], [1, 2])
        np.testing.assert_allclose(np.array(tracks[0]["points"])[:, 0], 101.05)
        np.testing.assert_allclose(np.array(tracks[1]["points"])[:, 0], 401.05)

    def test_missing_lane_hold_recovery_and_confidence(self):
        """Hold missing points and confidence, then clear stale flags on recovery."""
        tracker = LaneTracker()
        first = tracker.update(frame(lane(confidence=0.8), lane(400)))["lanes"][0]
        held = tracker.update(frame(lane(400)))["lanes"][0]
        self.assertEqual(held["points"], first["points"])
        self.assertEqual((held["predicted"], held["missed"], held["confidence"]), (True, 1, 0.8))
        recovered = tracker.update(frame(lane(102, confidence=0.2), lane(400)))["lanes"][0]
        self.assertEqual((recovered["track_id"], recovered["predicted"], recovered["missed"]), (1, False, 0))
        self.assertEqual(recovered["confidence"], 0.2)
        self.assertIsNone(tracker.update(frame(lane(), lane(400)))["lanes"][0]["confidence"])

    def test_expiration_boundary_and_ids_are_not_reused(self):
        """Keep exactly max_missed frames and allocate a fresh ID after expiry."""
        tracker = LaneTracker(max_missed=3)
        tracker.update(frame(lane()))
        for missed in (1, 2, 3):
            track = tracker.update(frame())["lanes"][0]
            self.assertEqual((track["predicted"], track["missed"]), (True, missed))
        self.assertEqual(tracker.update(frame())["lanes"], [])
        self.assertEqual(tracker.update(frame(lane()))["lanes"][0]["track_id"], 2)

    def test_far_lane_and_one_to_one_matching(self):
        """A track cannot absorb two observations or a distant observation."""
        tracker = LaneTracker()
        tracker.update(frame(lane()))
        tracks = tracker.update(frame(lane(101), lane(102), lane(500)))["lanes"]
        self.assertEqual([t["track_id"] for t in tracks], [1, 2, 3])
        self.assertTrue(all(not t["predicted"] for t in tracks))

    def test_global_association_avoids_greedy_identity_loss(self):
        """Keep both identities when the nearest pair blocks a feasible match."""
        tracker = LaneTracker(alpha=1)
        tracker.update(frame(lane(100), lane(140)))
        tracks = tracker.update(frame(lane(125), lane(180)))["lanes"]
        self.assertEqual([t['track_id'] for t in tracks], [1, 2])
        self.assertEqual([t['points'][0][0] for t in tracks], [125, 180])
        self.assertTrue(all(not t['predicted'] for t in tracks))

    def test_matching_threshold_and_width_normalization(self):
        """Accept distance at the gate and reject movement just beyond it."""
        for width, shift, ids in ((1000, 80, [1]), (1000, 81, [1, 2]),
                                  (500, 40, [1]), (500, 41, [1, 2])):
            with self.subTest(width=width, shift=shift):
                tracker = LaneTracker()
                tracker.update(frame(lane(), width=width))
                tracks = tracker.update(frame(lane(100 + shift), width=width))["lanes"]
                self.assertEqual([t["track_id"] for t in tracks], ids)

    def test_camera_center_crossing_does_not_assign_left_right_ids(self):
        """Preserve identity when a lane moves across the camera axis."""
        tracker = LaneTracker(alpha=1)
        tracker.update(frame(lane(300)))
        track = tracker.update(frame(lane(340)))["lanes"][0]
        self.assertEqual(track["track_id"], 1)
        self.assertEqual(track["points"][0][0], 340)

    def test_variable_keypoints_smooth_only_observed_overlap(self):
        """Interpolate unequal samples without extending either observed domain."""
        tracker = LaneTracker(alpha=0.5)
        tracker.update(frame({"lane_index": 0, "points": [[100, 100], [120, 300]], "confidence": None}))
        track = tracker.update(frame({"lane_index": 0, "points": [[115, 150], [125, 250], [129, 290]], "confidence": None}))["lanes"][0]
        self.assertEqual(track["track_id"], 1)
        np.testing.assert_allclose(track["points"], [[110, 150], [120, 250], [124, 290]])

    def test_nonoverlap_or_single_common_y_cannot_match(self):
        """Do not match by extrapolating across disjoint or touching domains."""
        for ys in ((200, 300), (150, 250)):
            with self.subTest(ys=ys):
                tracker = LaneTracker()
                tracker.update(frame(lane(100, (50, 150))))
                tracks = tracker.update(frame(lane(100, ys)))["lanes"]
                self.assertEqual([t["track_id"] for t in tracks], [1, 2])
                self.assertTrue(tracks[0]["predicted"])

    def test_recovered_observation_restores_bottom_and_geometry(self):
        """A brief occlusion must not permanently remove the measurement row."""
        from src.geometry import analyze_lanes

        tracker = LaneTracker()
        tracker.update(frame(lane(200, (180, 350)), lane(440, (180, 350))))
        shortened = tracker.update(frame(lane(200, (180, 260)), lane(440, (180, 350))))
        self.assertFalse(analyze_lanes(shortened)['valid_lane'])
        recovered = tracker.update(frame(lane(202, (160, 350)), lane(442, (160, 350))))
        self.assertEqual([t['track_id'] for t in recovered['lanes']], [1, 2])
        self.assertTrue(analyze_lanes(recovered)['valid_lane'])
        self.assertEqual(recovered['lanes'][0]['points'][0], [202.0, 160.0])
        self.assertEqual(recovered['lanes'][0]['points'][-1], [202.0, 350.0])

    def test_reset_clears_state_and_allows_new_dimensions(self):
        """Start each video with ID 1 and no retained tracks."""
        tracker = LaneTracker()
        tracker.update(frame(lane(), lane(400)))
        tracker.reset()
        result = tracker.update(frame(lane(10, (0, 99)), width=100, height=100))
        self.assertEqual([t["track_id"] for t in result["lanes"]], [1])
        self.assertEqual((result["frame_width"], result["frame_height"]), (100, 100))

    def test_size_change_raises_without_corrupting_state(self):
        """Reject coordinates from a new size until the caller resets."""
        tracker = LaneTracker()
        tracker.update(frame(lane()))
        with self.assertRaisesRegex(ValueError, "reset"):
            tracker.update(frame(width=800))
        self.assertEqual(tracker.update(frame(lane()))["lanes"][0]["track_id"], 1)

    def test_alpha_endpoints_and_zero_missed(self):
        """Support pure hold/pure observation EMA and immediate track expiry."""
        for alpha, expected_x in ((0, 100), (1, 110)):
            with self.subTest(alpha=alpha):
                tracker = LaneTracker(alpha=alpha, max_missed=0)
                tracker.update(frame(lane()))
                track = tracker.update(frame(lane(110)))["lanes"][0]
                self.assertEqual(track["points"][0][0], expected_x)
                self.assertEqual(tracker.update(frame())["lanes"], [])

    def test_invalid_parameters(self):
        """Reject invalid smoothing weights and retention counts."""
        for kwargs in ({"alpha": -0.1}, {"alpha": 1.1}, {"alpha": float("nan")},
                       {"alpha": True}, {"alpha": "0.5"}, {"max_missed": -1},
                       {"max_missed": 1.5}, {"max_missed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LaneTracker(**kwargs)

    def test_invalid_input_is_atomic(self):
        """Reject malformed observations before modifying existing tracks."""
        bad_points = ([], [[1, 2]], [[1, 2, 3], [2, 3, 4]],
                      [[1, 1], [2, float("nan")]], [[float("inf"), 1], [2, 2]],
                      [[-1, 0], [2, 2]], [[640, 0], [2, 2]],
                      [[0, 360], [2, 2]], [[1, 2], [2, 2]])
        invalid = [frame(lane(), {"lane_index": 1, "points": p, "confidence": None}) for p in bad_points]
        invalid += [None, {}, frame(width=0), frame(height=1.5), frame(width=True),
                    {**frame(), "lanes": None}, frame(lane(confidence=1.1)),
                    frame(lane(confidence=float("nan"))), frame(lane(confidence=True))]
        for detection in invalid:
            with self.subTest(detection=detection):
                tracker = LaneTracker()
                tracker.update(frame(lane()))
                with self.assertRaises(ValueError):
                    tracker.update(detection)
                track = tracker.update(frame())["lanes"][0]
                self.assertEqual((track["track_id"], track["missed"]), (1, 1))

    def test_sorting_bounds_and_numpy_input(self):
        """Accept numpy data and sort points while preserving pixel boundaries."""
        detection = frame({"lane_index": 0, "points": np.array([[639, 359], [0, 0]]), "confidence": np.float32(0.5)})
        track = LaneTracker().update(detection)["lanes"][0]
        self.assertEqual(track["points"], [[0, 0], [639, 359]])
        self.assertEqual(track["confidence"], 0.5)

    def test_input_and_returned_points_do_not_alias_state(self):
        """External edits must not change the retained tracking state."""
        tracker = LaneTracker()
        detection = frame(lane())
        original = copy.deepcopy(detection)
        result = tracker.update(detection)
        self.assertEqual(detection, original)
        detection["lanes"][0]["points"][0][0] = 500
        result["lanes"][0]["points"][0][0] = 600
        self.assertEqual(tracker.update(frame())["lanes"][0]["points"][0], [100, 100])

    def test_tracking_time_uses_perf_counter(self):
        """Measure elapsed update time in milliseconds."""
        with patch("src.tracker.perf_counter", side_effect=[10.0, 10.002]):
            result = LaneTracker().update(frame())
        self.assertAlmostEqual(result["tracking_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()

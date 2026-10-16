"""Independent mock-tracking tests: python -m unittest discover -s tests -p test_geometry.py -v."""

from copy import deepcopy
import math
import unittest

from src.geometry import analyze_lanes


def lane(track_id, x, *, predicted=False, missed=0, points=None):
    return dict(track_id=track_id, points=points if points is not None else
                [[x, y] for y in (250, 350, 430, 470)],
                confidence=None, predicted=predicted, missed=missed)


def tracking(*lanes, width=640, height=480):
    return dict(lanes=list(lanes), frame_width=width, frame_height=height, tracking_ms=0.8)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.data = tracking(lane(1, 219.5), lane(2, 419.5))

    def assertUnknown(self, result):
        self.assertEqual(set(result), {
            'valid_lane', 'reference_y', 'vehicle_center_x', 'lane_center_x',
            'left_track_id', 'right_track_id', 'offset_normalized', 'warning',
            'curve', 'lane_type_left', 'lane_type_right'})
        self.assertIs(result['valid_lane'], False)
        self.assertEqual(result['warning'], 'UNKNOWN')
        for key in ('lane_center_x', 'left_track_id', 'right_track_id', 'offset_normalized'):
            self.assertIsNone(result[key])
        for key in ('curve', 'lane_type_left', 'lane_type_right'):
            self.assertEqual(result[key], 'UNKNOWN')

    def test_centered_schema_and_no_mutation(self):
        original = deepcopy(self.data)
        result = analyze_lanes(self.data)
        self.assertEqual(result, dict(valid_lane=True, reference_y=407,
            vehicle_center_x=319.5, lane_center_x=319.5, left_track_id=1,
            right_track_id=2, offset_normalized=0.0, warning='SAFE',
            curve='UNKNOWN', lane_type_left='UNKNOWN', lane_type_right='UNKNOWN'))
        self.assertEqual(self.data, original)

    def test_thresholds_and_sign_both_directions(self):
        for offset, expected in ((0, 'SAFE'), (0.149999, 'SAFE'), (0.15, 'CAUTION'),
                                 (0.150001, 'CAUTION'), (0.249999, 'CAUTION'),
                                 (0.25, 'WARNING'), (0.250001, 'WARNING'), (0.49, 'WARNING')):
            for sign in (-1, 1):
                with self.subTest(offset=offset, sign=sign):
                    center = 319.5 - sign * offset * 200
                    result = analyze_lanes(tracking(lane(1, center-100), lane(2, center+100)))
                    self.assertAlmostEqual(result['offset_normalized'], sign * offset)
                    self.assertEqual(result['warning'], expected)

    def test_nearest_boundaries_not_ids_or_order(self):
        result = analyze_lanes(tracking(lane(70, 600), lane(99, 200), lane(3, 30), lane(5, 400)))
        self.assertEqual((result['left_track_id'], result['right_track_id']), (99, 5))

    def test_missing_or_not_bracketing(self):
        for lanes in ([], [lane(1, 200)], [lane(1, 400)],
                      [lane(1, 100), lane(2, 200)], [lane(1, 400), lane(2, 500)]):
            with self.subTest(lanes=lanes):
                self.assertUnknown(analyze_lanes(tracking(*lanes)))

    def test_exact_camera_boundary_is_ambiguous(self):
        self.data['lanes'].append(lane(3, 319.5))
        self.assertUnknown(analyze_lanes(self.data))

    def test_prediction_rules(self):
        for ml, mr, valid in ((0, 0, True), (1, 0, True), (0, 1, True),
                              (1, 1, False), (2, 0, False), (0, 2, False), (3, 3, False)):
            with self.subTest(missed=(ml, mr)):
                data = tracking(lane(1, 200, predicted=ml > 0, missed=ml),
                                lane(2, 400, predicted=mr > 0, missed=mr))
                if valid:
                    self.assertTrue(analyze_lanes(data)['valid_lane'])
                else:
                    self.assertUnknown(analyze_lanes(data))

    def test_stale_nearest_does_not_fall_back_to_outer(self):
        self.data['lanes'][0].update(predicted=True, missed=2)
        self.data['lanes'].append(lane(3, 50))
        self.assertUnknown(analyze_lanes(self.data))

    def test_missing_required_metadata_raises(self):
        for key in ('predicted', 'missed'):
            data = deepcopy(self.data)
            del data['lanes'][0][key]
            with self.assertRaisesRegex(ValueError, key):
                analyze_lanes(data)

    def test_invalid_metadata(self):
        for updates in ({'predicted': False, 'missed': 1}, {'predicted': True, 'missed': 0},
                        {'predicted': 'False'}, {'missed': -1}, {'missed': True},
                        {'missed': 0.5}, {'track_id': None}, {'track_id': True},
                        {'confidence': float('nan')}, {'confidence': 1.1}):
            with self.subTest(updates=updates):
                data = deepcopy(self.data)
                data['lanes'][0].update(updates)
                self.assertUnknown(analyze_lanes(data))

    def test_duplicate_ids_or_positions(self):
        for extra in (lane(1, 50), lane(3, 219.5)):
            self.assertUnknown(analyze_lanes(tracking(*self.data['lanes'], extra)))

    def test_resolution_and_camera_position(self):
        data = tracking(lane(1, 400, points=[[400, 0], [400, 719]]),
                        lane(2, 800, points=[[800, 0], [800, 719]]), width=1280, height=720)
        result = analyze_lanes(data, 700/1279)
        self.assertEqual(result['reference_y'], 611)
        self.assertEqual(result['vehicle_center_x'], 700)
        self.assertAlmostEqual(result['offset_normalized'], 0.25)
        self.assertEqual(result['warning'], 'WARNING')

    def test_invalid_ratio(self):
        for value in (-0.1, 1.2, math.nan, math.inf, -math.inf, None, '0.5', True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'camera_center_ratio'):
                analyze_lanes(self.data, value)
        for value in (0, 1):
            self.assertUnknown(analyze_lanes(self.data, value))

    def test_invalid_frame_or_structure(self):
        for value in (None, [], {}, {'lanes': None}, tracking(None)):
            self.assertUnknown(analyze_lanes(value))
        for key in ('frame_width', 'frame_height'):
            for value in (0, 1, -1, 640.5, '640', None, True, math.nan, math.inf):
                data = deepcopy(self.data)
                data[key] = value
                self.assertUnknown(analyze_lanes(data))

    def test_bad_points_invalidate_even_unselected_lane(self):
        for points in (None, [], [[10, 300]], [[10, 300, 0], [10, 470]],
                       [[math.nan, 300], [10, 470]], [[10, math.inf], [10, 470]],
                       [[-1, 300], [10, 470]], [[640, 300], [10, 470]],
                       [[10, 480], [10, 470]], [['10', 300], [10, 470]],
                       [[10, 300], [11, 300], [10, 470]]):
            with self.subTest(points=points):
                extra = lane(3, 10)
                extra['points'] = points
                self.assertUnknown(analyze_lanes(tracking(*self.data['lanes'], extra)))

    def test_sort_duplicates_and_linear_interpolation(self):
        left = lane(1, 0, points=[[220, 470], [180, 250], [180, 250], [200, 350]])
        right = lane(2, 0, points=[[400, 350], [420, 470], [380, 250]])
        data = tracking(left, right)
        original = deepcopy(data)
        result = analyze_lanes(data)
        self.assertAlmostEqual(result['lane_center_x'], 309.5)
        self.assertAlmostEqual(result['offset_normalized'], 0.05)
        self.assertEqual(data, original)

    def test_no_extrapolation_and_inclusive_endpoints(self):
        for ys, valid in (([250, 406], False), ([408, 470], False),
                          ([250, 407], True), ([407, 470], True)):
            data = tracking(lane(1, 0, points=[[200, y] for y in ys]), lane(2, 400))
            result = analyze_lanes(data)
            if valid:
                self.assertTrue(result['valid_lane'])
            else:
                self.assertUnknown(result)

    def test_tiny_width_and_zero_width(self):
        for left, right in ((319.5, 319.5), (319, 320), (300, 338)):
            self.assertUnknown(analyze_lanes(tracking(lane(1, left), lane(2, right))))
        self.assertTrue(analyze_lanes(tracking(lane(1, 299.5), lane(2, 339.5)))['valid_lane'])

    def test_crossing_touching_and_interior_knot(self):
        for points in ([[450, 250], [200, 470]], [[400, 250], [200, 470]],
                       [[200, 250], [450, 300], [200, 470]]):
            self.assertUnknown(analyze_lanes(tracking(lane(1, 0, points=points), lane(2, 400))))

    def test_crossing_outside_common_domain_is_not_extrapolated(self):
        data = tracking(lane(1, 0, points=[[450, 0], [200, 350], [200, 470]]),
                        lane(2, 0, points=[[400, 350], [400, 470]]))
        self.assertTrue(analyze_lanes(data)['valid_lane'])

    def test_p1_stays_unknown_and_no_fabricated_confidence(self):
        result = analyze_lanes(self.data)
        for key in ('curve', 'lane_type_left', 'lane_type_right'):
            self.assertEqual(result[key], 'UNKNOWN')
        self.assertNotIn('confidence', result)


if __name__ == '__main__':
    unittest.main()

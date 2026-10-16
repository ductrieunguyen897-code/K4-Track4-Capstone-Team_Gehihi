"""Regression checks for image support, camera crop mapping and stream reset."""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.detector import LaneDetector
from src.lane_evidence import estimate_horizon, marking_response, refine_lanes


def proposal(points):
    return {'lane_index': 0, 'points': points, 'confidence': None}


def test_white_and_yellow_markings_refine_toward_known_pixels():
    frame = np.full((720, 1280, 3), 65, np.uint8)
    for x, color in [(300, (245, 245, 245)), (900, (0, 215, 255))]:
        cv2.line(frame, (x, 390), (x, 700), color, 7)
    proposals = [proposal([[310, 380], [310, 710]]), proposal([[910, 380], [910, 710]])]
    lanes, score = refine_lanes(proposals, marking_response(frame), 1280, 720)
    assert len(lanes) == 2 and score > 0
    for lane, expected in zip(lanes, [300, 900]):
        assert max(abs(x - expected) for x, _ in lane['points']) <= 6
        assert lane['confidence'] is None
    assert proposals[0]['points'][0][0] == 310


@pytest.mark.parametrize('color', [(0, 0, 0), (65, 65, 65), (245, 245, 245), (220, 180, 130)])
def test_uniform_surfaces_do_not_support_lane_proposals(color):
    frame = np.full((360, 640, 3), color, np.uint8)
    lanes, score = refine_lanes([proposal([[200, 100], [200, 355]])], marking_response(frame), 640, 360)
    assert lanes == [] and score == 0


def test_sky_evidence_and_short_vehicle_edge_do_not_form_road_lane():
    frame = np.full((360, 640, 3), 65, np.uint8)
    cv2.line(frame, (250, 60), (250, 150), (255, 255, 255), 5)
    cv2.line(frame, (400, 270), (400, 290), (255, 255, 255), 5)
    lanes, _ = refine_lanes([proposal([[250, 60], [250, 150]]),
                             proposal([[400, 200], [400, 350]])], marking_response(frame), 640, 360)
    assert lanes == []


def test_horizon_distinguishes_camera_with_large_sky_area():
    def road(vanish_y):
        frame = np.full((720, 1280, 3), 65, np.uint8)
        cv2.line(frame, (640, vanish_y), (180, 719), (245, 245, 245), 7)
        cv2.line(frame, (640, vanish_y), (1100, 719), (245, 245, 245), 7)
        return frame
    assert estimate_horizon(marking_response(road(390))) < 0.64
    assert estimate_horizon(marking_response(road(510))) > 0.64


def test_cropped_coordinates_map_back_to_full_frame():
    detector = object.__new__(LaneDetector)
    detector.input_name, detector.output_name = 'input', 'output'
    detector._preprocess = Mock(return_value=np.zeros((1, 3, 288, 800), np.float32))
    detector.session = Mock()
    detector.session.run.return_value = [np.zeros((1, 101, 56, 4))]
    detector._decode_output = Mock(return_value=[proposal([[250, 100], [400, 500]])])
    frame = np.zeros((1080, 1920, 3), np.uint8)
    lanes = detector._detect_crop(frame, 0.45)
    assert detector._preprocess.call_args.args[0].shape == (594, 1920, 3)
    assert detector._decode_output.call_args.kwargs == {'frame_w': 1920, 'frame_h': 594}
    assert lanes[0]['points'] == [[250, 586], [400, 986]]


def test_detector_reset_discards_previous_automatic_camera_crop():
    detector = object.__new__(LaneDetector)
    detector.road_crop_ratio = None
    detector._selected_crop = 0.45
    detector._frame_size = (1920, 1080)
    detector.reset()
    assert detector._selected_crop is None and detector._frame_size is None


@pytest.mark.parametrize('horizon,expected', [(0.56, 0.0), (0.744, 0.45), (0.765, 0.45)])
def test_automatic_crop_does_not_depend_on_first_frame_detection_count(monkeypatch, horizon, expected):
    detector = object.__new__(LaneDetector)
    detector.road_crop_ratio = None
    detector.reset()
    detector._detect_crop = Mock(return_value=[])
    monkeypatch.setattr('src.detector.estimate_horizon', lambda response: horizon)
    frame = np.zeros((360, 640, 3), np.uint8)
    assert detector.detect(frame)['lanes'] == []
    assert detector._selected_crop == expected
    # Keep the camera calibration as later frames lose their markings.
    monkeypatch.setattr('src.detector.estimate_horizon', lambda response: 0.4)
    detector.detect(frame)
    assert detector._selected_crop == expected

"""Regression: slanted boundaries can cross the image axis above the vehicle."""

import pytest

from src.geometry import analyze_lanes


def test_near_vehicle_row_selects_the_containing_lane_on_slanted_road():
    # The middle marking crosses x=500 between y=850 and y=900.
    tracking = dict(frame_width=1000, frame_height=1000, lanes=[
        dict(track_id=1, points=[[200, 700], [100, 980]], confidence=None, predicted=False, missed=0),
        dict(track_id=2, points=[[300, 700], [650, 980]], confidence=None, predicted=False, missed=0),
        dict(track_id=3, points=[[450, 700], [950, 980]], confidence=None, predicted=False, missed=0)])
    distant = analyze_lanes(tracking)
    nearby = analyze_lanes(tracking, reference_y_ratio=.9)
    assert (distant['left_track_id'], distant['right_track_id']) == (2, 3)
    assert (nearby['left_track_id'], nearby['right_track_id']) == (1, 2)
    assert nearby['reference_y'] == 899


def test_reference_row_does_not_extrapolate_missing_observations():
    tracking = dict(frame_width=640, frame_height=480, lanes=[
        dict(track_id=i, points=[[x, 250], [x, 420]], confidence=None, predicted=False, missed=0)
        for i, x in [(1, 150), (2, 450)]])
    assert analyze_lanes(tracking)['valid_lane']
    assert not analyze_lanes(tracking, reference_y_ratio=.9)['valid_lane']


@pytest.mark.parametrize('ratio', [0, 1, -.1, float('nan'), True])
def test_invalid_reference_row_is_rejected(ratio):
    with pytest.raises(ValueError, match='reference_y_ratio'):
        analyze_lanes({}, reference_y_ratio=ratio)

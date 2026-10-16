"""Independent decoder-coordinate and absence tests; no downloaded model needed."""

import numpy as np
import pytest

from src.ufld_v2 import configuration, decode


def logits():
    tensors = {name: np.full(shape, -20., np.float32) for name, shape in {
        'loc_row': (1, 200, 72, 4), 'loc_col': (1, 100, 81, 4),
        'exist_row': (1, 2, 72, 4), 'exist_col': (1, 2, 81, 4)}.items()}
    tensors['exist_row'][:, 0] = 20
    tensors['exist_col'][:, 0] = 20
    return tensors


def test_absent_lanes_produce_no_points():
    assert decode(logits(), 1640, 590) == []


def test_row_coordinates_use_full_roi_instead_of_network_crop():
    tensors = logits()
    for slot, cell in [(1, 50), (2, 150)]:
        tensors['exist_row'][0, 0, :, slot] = -20
        tensors['exist_row'][0, 1, :, slot] = 20
        tensors['loc_row'][0, cell, :, slot] = 20
    lanes = decode(tensors, 1640, 590)
    assert len(lanes) == 2
    assert lanes[0]['points'][0] == [int(50.5 / 199 * 1640), int(.42 * 590)]
    assert lanes[1]['points'][0][0] == int(150.5 / 199 * 1640)
    assert all(0 <= y < 590 for lane in lanes for _, y in lane['points'])
    assert all(0 <= lane['confidence'] <= 1 for lane in lanes)


def test_column_predictions_merge_duplicate_integer_rows():
    tensors = logits()
    tensors['exist_col'][0, 0, :, 3] = -20
    tensors['exist_col'][0, 1, :, 3] = 20
    # Increasing y with occasional identical integer rows after downscaling.
    for index in range(81):
        tensors['loc_col'][0, min(98, index + 10), index, 3] = 20
    lanes = decode(tensors, 640, 80)
    assert len(lanes) == 1
    rows = [y for _, y in lanes[0]['points']]
    assert rows == sorted(set(rows))


def test_folded_column_prediction_is_rejected_instead_of_drawing_zigzag():
    tensors = logits()
    tensors['exist_col'][0, 0, :, 3] = -20
    tensors['exist_col'][0, 1, :, 3] = 20
    for index in range(81):
        cell = 10 + (index if index < 40 else 80 - index)
        tensors['loc_col'][0, cell, index, 3] = 20
    assert decode(tensors, 1280, 720) == []


def test_large_row_head_jump_is_not_a_usable_boundary():
    tensors = logits()
    tensors['exist_row'][0, 0, :, 1] = -20
    tensors['exist_row'][0, 1, :, 1] = 20
    for index in range(72):
        tensors['loc_row'][0, 50 if index < 36 else 160, index, 1] = 20
    assert decode(tensors, 1280, 720) == []


def test_local_expectation_at_grid_edge_does_not_count_clipped_cell_twice():
    tensors = logits()
    tensors['exist_row'][0, 0, :, 1] = -20
    tensors['exist_row'][0, 1, :, 1] = 20
    tensors['loc_row'][0, 0, :, 1] = np.log(2)
    tensors['loc_row'][0, 1, :, 1] = 0
    lane = decode(tensors, 1640, 590)[0]
    expected_x = int(((1 / 3) + .5) / 199 * 1640)
    assert all(x == expected_x for x, _ in lane['points'])


def test_unknown_layout_and_nonfinite_outputs_fail_explicitly():
    with pytest.raises(ValueError, match='requires'):
        decode({}, 640, 360)
    tensors = logits()
    tensors['loc_row'][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        decode(tensors, 640, 360)
    with pytest.raises(ValueError, match='row count'):
        configuration(123)


def test_crop_configuration_matches_training_datasets():
    rows, crop = configuration(72)
    assert crop == .6 and rows[0] == .42 and rows[-1] == 1
    rows, crop = configuration(56)
    assert crop == .8 and rows[0] == 160 / 720 and rows[-1] == 710 / 720

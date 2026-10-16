"""Caching must reuse only matching model/configuration and invalidate new files."""

from unittest.mock import patch

from src.runtime import _load_detector, get_detector


def test_detector_session_cache_keys_include_camera_device_and_weight_version(tmp_path):
    _load_detector.cache_clear()
    path = tmp_path / 'weights.onnx'
    path.write_bytes(b'first')
    try:
        with patch('src.runtime.LaneDetector') as create:
            first = get_detector(str(path), 'cpu', 0.0)
            assert get_detector(str(path), 'cpu', 0.0) is first
            assert create.call_count == 1
            get_detector(str(path), 'cpu', .35)
            assert create.call_count == 2
            get_detector(str(path), 'auto', .35)
            assert create.call_count == 3
            path.write_bytes(b'replaced weights')
            get_detector(str(path), 'auto', .35)
            assert create.call_count == 4
    finally:
        _load_detector.cache_clear()

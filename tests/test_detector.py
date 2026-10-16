"""Unit tests and CLI smoke tests for LaneDetector (src/detector.py).

Owner: Quang (branch `quang`)
Usage:
    pytest tests/test_detector.py
    python tests/test_detector.py --model models/ufld_tusimple.onnx --image data/sample_frame.jpg
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import LaneDetector  # noqa: E402

DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "ufld_tusimple.onnx")


def make_synthetic_road_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a synthetic BGR road image with sky, asphalt, and white/yellow lane lines."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    horizon_y = int(height * 0.42)

    # Sky (top) and asphalt road (bottom)
    frame[:horizon_y, :] = (210, 175, 130)
    frame[horizon_y:, :] = (65, 65, 65)

    vanish_x = int(width * 0.50)
    vanish_y = int(height * 0.44)

    # Left lane line (yellow-ish in BGR)
    left_bottom_x = int(width * 0.18)
    cv2.line(
        frame,
        (vanish_x - int(width * 0.04), vanish_y),
        (left_bottom_x, height - 1),
        (0, 215, 255),
        max(4, width // 160),
        lineType=cv2.LINE_AA,
    )

    # Right lane line (white in BGR)
    right_bottom_x = int(width * 0.82)
    cv2.line(
        frame,
        (vanish_x + int(width * 0.04), vanish_y),
        (right_bottom_x, height - 1),
        (245, 245, 245),
        max(4, width // 160),
        lineType=cv2.LINE_AA,
    )

    # Adjacent dashed lanes
    far_left_bottom_x = int(width * 0.02)
    cv2.line(
        frame,
        (vanish_x - int(width * 0.10), vanish_y + int(height * 0.04)),
        (far_left_bottom_x, int(height * 0.85)),
        (220, 220, 220),
        max(3, width // 200),
        lineType=cv2.LINE_AA,
    )

    far_right_bottom_x = int(width * 0.98)
    cv2.line(
        frame,
        (vanish_x + int(width * 0.10), vanish_y + int(height * 0.04)),
        (far_right_bottom_x, int(height * 0.85)),
        (220, 220, 220),
        max(3, width // 200),
        lineType=cv2.LINE_AA,
    )

    return frame


def assert_valid_detector_contract(result: dict[str, Any], expected_w: int, expected_h: int) -> None:
    """Verify 100% compliance with Section 5.1 & 5.2 of PROJECT_OVERVIEW.md."""
    assert isinstance(result, dict), "Output must be a dict"
    assert set(result.keys()) == {"lanes", "frame_width", "frame_height", "inference_ms"}

    assert isinstance(result["frame_width"], int)
    assert isinstance(result["frame_height"], int)
    assert result["frame_width"] == expected_w
    assert result["frame_height"] == expected_h

    assert isinstance(result["inference_ms"], float)
    assert math.isfinite(result["inference_ms"])
    assert result["inference_ms"] >= 0.0

    lanes = result["lanes"]
    assert isinstance(lanes, list)
    assert 0 <= len(lanes) <= 4

    for expected_idx, lane in enumerate(lanes):
        assert isinstance(lane, dict)
        assert set(lane.keys()) == {"lane_index", "points", "confidence"}
        assert isinstance(lane["lane_index"], int)
        assert lane["lane_index"] == expected_idx

        conf = lane["confidence"]
        assert conf is None or (isinstance(conf, float) and 0.0 <= conf <= 1.0)

        points = lane["points"]
        assert isinstance(points, list)
        assert len(points) >= 2, "Each detected lane must have at least 2 points"

        prev_y = -1
        for pt in points:
            assert isinstance(pt, list) and len(pt) == 2
            x, y = pt[0], pt[1]
            assert isinstance(x, int) and isinstance(y, int)
            assert not (math.isnan(x) or math.isnan(y))
            assert 0 <= x < expected_w, f"x={x} out of bounds [0, {expected_w - 1}]"
            assert 0 <= y < expected_h, f"y={y} out of bounds [0, {expected_h - 1}]"
            assert y > prev_y, f"Points must have strictly increasing y: {y} <= {prev_y}"
            prev_y = y


# ---------------------------------------------------------------------------
# Unit tests (always run, even without weights)
# ---------------------------------------------------------------------------

def test_missing_model_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        LaneDetector("models/non_existent_weights_12345.onnx")

    with pytest.raises(FileNotFoundError):
        LaneDetector("")


def test_validate_frame_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="numpy.ndarray"):
        LaneDetector._validate_frame(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dtype"):
        LaneDetector._validate_frame(np.zeros((480, 640, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        LaneDetector._validate_frame(np.zeros((480, 640), dtype=np.uint8))

    with pytest.raises(ValueError, match="shape"):
        LaneDetector._validate_frame(np.zeros((480, 640, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="dimensions"):
        LaneDetector._validate_frame(np.zeros((1, 640, 3), dtype=np.uint8))


def test_decode_output_synthetic_logits_and_empty() -> None:
    """Verify decoder math, y-sorting, deduplication, left-to-right ordering, and empty output."""
    detector = object.__new__(LaneDetector)
    detector.input_width = 800
    detector.input_height = 288
    detector.griding_num = 100
    detector.cls_num_per_lane = 56
    detector.num_lanes = 4
    detector.row_anchor = LaneDetector._resolve_row_anchors(56)

    # 1. All-background logits -> index 100 (griding_num) has highest logit -> lanes == []
    empty_logits = np.full((1, 101, 56, 4), -10.0, dtype=np.float32)
    empty_logits[0, 100, :, :] = 10.0
    lanes_empty = detector._decode_output(empty_logits, frame_w=640, frame_h=480)
    assert lanes_empty == []

    # 2. Two valid lanes placed in reverse order in tensor (lane slot 3 on left, lane slot 0 on right)
    logits = np.full((1, 101, 56, 4), -10.0, dtype=np.float32)
    logits[0, 100, :, :] = 10.0  # default absence

    # Slot 0: right lane around grid cell 70
    for r in range(10, 45):
        logits[0, 100, r, 0] = -10.0
        logits[0, 70, r, 0] = 15.0

    # Slot 3: left lane around grid cell 25
    for r in range(10, 45):
        logits[0, 100, r, 3] = -10.0
        logits[0, 25, r, 3] = 15.0

    lanes = detector._decode_output(logits, frame_w=1280, frame_h=720)
    assert len(lanes) == 2
    assert lanes[0]["lane_index"] == 0
    assert lanes[1]["lane_index"] == 1
    # Left lane (slot 3, grid ~25) must come before right lane (slot 0, grid ~70)
    assert lanes[0]["points"][-1][0] < lanes[1]["points"][-1][0]

    fake_result = {
        "lanes": lanes,
        "frame_width": 1280,
        "frame_height": 720,
        "inference_ms": 1.0,
    }
    assert_valid_detector_contract(fake_result, 1280, 720)


# ---------------------------------------------------------------------------
# Real ONNX Model Smoke Tests (run when models/ufld_tusimple.onnx is present)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> LaneDetector:
    return LaneDetector(DEFAULT_MODEL_PATH, device="cpu")


@pytest.mark.skipif(
    not os.path.isfile(DEFAULT_MODEL_PATH),
    reason=f"Pretrained ONNX model not found at {DEFAULT_MODEL_PATH}",
)
class TestRealONNXDetector:
    def test_multiple_resolutions_and_immutability(self, detector: LaneDetector) -> None:
        resolutions = [(640, 360), (1280, 720), (640, 480)]
        for w, h in resolutions:
            frame = make_synthetic_road_frame(width=w, height=h)
            frame_copy = frame.copy()
            result = detector.detect(frame)

            # Frame must not be modified in place
            assert np.array_equal(frame, frame_copy), "detect() must not mutate input frame_bgr"
            assert_valid_detector_contract(result, expected_w=w, expected_h=h)

    def test_blank_frame_does_not_crash(self, detector: LaneDetector) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        assert_valid_detector_contract(result, expected_w=640, expected_h=480)

    def test_three_consecutive_frames_consistency(self, detector: LaneDetector) -> None:
        frame = make_synthetic_road_frame(width=1280, height=720)
        results = [detector.detect(frame) for _ in range(3)]
        for res in results:
            assert_valid_detector_contract(res, expected_w=1280, expected_h=720)

        # Deterministic ONNX CPU execution on identical frames must yield identical lanes
        assert results[0]["lanes"] == results[1]["lanes"] == results[2]["lanes"]


# ---------------------------------------------------------------------------
# CLI Smoke Test Runner
# ---------------------------------------------------------------------------

def draw_lanes_overlay(frame_bgr: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
    """Render detected lane points and polylines onto a copy of frame_bgr for visual inspection."""
    vis = frame_bgr.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for lane in detection["lanes"]:
        idx = lane["lane_index"]
        color = colors[idx % len(colors)]
        pts = np.array(lane["points"], dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
        for x, y in lane["points"]:
            cv2.circle(vis, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
    return vis


def run_cli_smoke_test(model_path: str, image_path: str | None, output_vis: str | None) -> int:
    print(f"[SmokeTest] Loading LaneDetector from: {model_path}")
    detector = LaneDetector(model_path=model_path, device="cpu")
    print(
        f"[SmokeTest] Model loaded successfully | "
        f"input={detector.input_name}:{detector.input_shape} -> "
        f"output={detector.output_name}:{detector.output_shape}"
    )

    if image_path and os.path.isfile(image_path):
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to decode image from '{image_path}'.")
        source_desc = f"file '{image_path}'"
    else:
        if image_path:
            print(
                f"[SmokeTest] Note: '{image_path}' not found on disk; "
                "using generated 1280x720 BGR road frame."
            )
        frame = make_synthetic_road_frame(width=1280, height=720)
        source_desc = "synthetic 1280x720 road frame"

    h, w = frame.shape[:2]
    frame_backup = frame.copy()
    result = detector.detect(frame)

    assert np.array_equal(frame, frame_backup), "Input frame was mutated in-place!"
    assert_valid_detector_contract(result, expected_w=w, expected_h=h)

    print(f"[SmokeTest] Source: {source_desc} ({w}x{h})")
    print(f"[SmokeTest] inference_ms: {result['inference_ms']:.2f} ms")
    print(f"[SmokeTest] Detected lanes count: {len(result['lanes'])}")
    for lane in result["lanes"]:
        pts = lane["points"]
        print(
            f"  - lane_index={lane['lane_index']}: {len(pts)} points | "
            f"top={pts[0]} -> bottom={pts[-1]} | confidence={lane['confidence']}"
        )

    # Test 3 resolutions (640x360, 1280x720, 640x480)
    for res_w, res_h in [(640, 360), (1280, 720), (640, 480)]:
        resized = cv2.resize(frame, (res_w, res_h), interpolation=cv2.INTER_LINEAR)
        res_out = detector.detect(resized)
        assert_valid_detector_contract(res_out, expected_w=res_w, expected_h=res_h)
        print(
            f"[SmokeTest] Resolution {res_w}x{res_h}: "
            f"lanes={len(res_out['lanes'])}, inference_ms={res_out['inference_ms']:.2f} ms"
        )

    # Test 3 consecutive frames consistency
    seq_results = [detector.detect(frame) for _ in range(3)]
    assert seq_results[0]["lanes"] == seq_results[1]["lanes"] == seq_results[2]["lanes"]
    print("[SmokeTest] 3-frame consistency check: PASSED")

    if output_vis:
        vis_img = draw_lanes_overlay(frame, result)
        os.makedirs(os.path.dirname(os.path.abspath(output_vis)), exist_ok=True)
        cv2.imwrite(output_vis, vis_img)
        print(f"[SmokeTest] Saved visualization overlay to: {output_vis}")

    print("[SmokeTest] ALL CHECKS PASSED!")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for UFLD ONNX LaneDetector.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to UFLD ONNX model file (default: models/ufld_tusimple.onnx)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to test road image (BGR)",
    )
    parser.add_argument(
        "--output-vis",
        type=str,
        default=None,
        help="Optional path to save overlay visualization image",
    )
    args = parser.parse_args()
    sys.exit(run_cli_smoke_test(args.model, args.image, args.output_vis))


if __name__ == "__main__":
    main()

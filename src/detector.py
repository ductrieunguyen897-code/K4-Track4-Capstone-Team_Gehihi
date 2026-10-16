"""UFLD ONNX Lane Detection module for LaneGuard AI.

Owner: Quang (branch `quang`)
Contract reference: tasks/PROJECT_OVERVIEW.md (Section 5.1 & 5.2)
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.lane_evidence import estimate_horizon, marking_response, refine_lanes

# Standard row anchors defined in the 288-height coordinate system of UFLD
TUSIMPLE_ROW_ANCHOR: list[int] = [
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
    116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
    168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
    220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
    272, 276, 280, 284,
]

CULANE_ROW_ANCHOR: list[int] = [
    121, 131, 141, 150, 160, 170, 180, 189, 199,
    209, 219, 228, 238, 248, 258, 267, 277, 287,
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_NORM_BIAS = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


def _softmax_axis0(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 0."""
    shifted = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def _interpolate_x_at_y(points: list[list[int]], target_y: float) -> float:
    """Estimate x coordinate of a sorted lane polyline at target_y for left-to-right ordering."""
    if target_y <= points[0][1]:
        return float(points[0][0])
    if target_y >= points[-1][1]:
        return float(points[-1][0])

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if y0 <= target_y <= y1:
            if y1 == y0:
                return float(x0)
            ratio = (target_y - y0) / (y1 - y0)
            return float(x0 + ratio * (x1 - x0))
    return float(points[-1][0])


class LaneDetector:
    """Ultra-Fast-Lane-Detection (UFLD) ONNX inference adapter."""

    def __init__(self, model_path: str, device: str = "cpu", road_crop_ratio: float | None = None) -> None:
        """Load ONNX session once. Raise FileNotFoundError if weights are missing."""
        if not model_path or not isinstance(model_path, str):
            raise FileNotFoundError(
                f"Invalid model_path: {model_path!r}. "
                "Please provide a valid path to the UFLD ONNX file (e.g. 'models/ufld_tusimple.onnx')."
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"UFLD ONNX weights not found at '{model_path}'. "
                "Please download the pretrained UFLD ONNX weights and place them at "
                "'models/ufld_tusimple.onnx' (see models/README.md for instructions)."
            )

        self.model_path = model_path
        self.device = (device or "cpu").lower()
        if road_crop_ratio is not None and (
            isinstance(road_crop_ratio, bool) or not isinstance(road_crop_ratio, (int, float))
            or not math.isfinite(road_crop_ratio) or not 0 <= road_crop_ratio <= 0.5
        ):
            raise ValueError("road_crop_ratio must be None (auto) or a number in [0, 0.5]")
        self.road_crop_ratio = road_crop_ratio
        self.reset()

        available_providers = ort.get_available_providers()
        providers: list[str] = []
        if self.device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = available_providers

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize ONNX Runtime session for '{self.model_path}' "
                f"with providers={providers} (onnxruntime=={ort.__version__}): {exc}"
            ) from exc

        # Inspect input metadata
        input_meta = self.session.get_inputs()[0]
        self.input_name: str = input_meta.name
        self.input_shape = input_meta.shape
        # Standard NCHW layout: [batch, 3, H, W]
        in_h = self.input_shape[2] if len(self.input_shape) >= 4 else 288
        in_w = self.input_shape[3] if len(self.input_shape) >= 4 else 800
        self.input_height: int = int(in_h) if isinstance(in_h, int) and in_h > 0 else 288
        self.input_width: int = int(in_w) if isinstance(in_w, int) and in_w > 0 else 800

        # Inspect output metadata
        output_meta = self.session.get_outputs()[0]
        self.output_name: str = output_meta.name
        self.output_shape = output_meta.shape

        # Determine griding_num, cls_num_per_lane, and row_anchor from model output shape
        # Expected output shape: (1, griding_num + 1, cls_num_per_lane, num_lanes)
        if len(self.output_shape) == 4 and isinstance(self.output_shape[1], int) and isinstance(self.output_shape[2], int):
            self.griding_num: int = int(self.output_shape[1]) - 1
            self.cls_num_per_lane: int = int(self.output_shape[2])
            self.num_lanes: int = int(self.output_shape[3]) if isinstance(self.output_shape[3], int) else 4
        else:
            self.griding_num = 100
            self.cls_num_per_lane = 56
            self.num_lanes = 4

        self.row_anchor = self._resolve_row_anchors(self.cls_num_per_lane)

    @staticmethod
    def _resolve_row_anchors(cls_num_per_lane: int) -> list[int]:
        if cls_num_per_lane == len(TUSIMPLE_ROW_ANCHOR):
            return TUSIMPLE_ROW_ANCHOR
        if cls_num_per_lane == len(CULANE_ROW_ANCHOR):
            return CULANE_ROW_ANCHOR
        # Fallback linear anchors across the lower road region in 288p reference space
        return [int(round(y)) for y in np.linspace(64, 284, cls_num_per_lane)]

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> tuple[int, int]:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise ValueError("Invalid frame_bgr: expected a non-null numpy.ndarray.")
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Invalid frame_bgr dtype: expected uint8, got {frame_bgr.dtype}.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Invalid frame_bgr shape: expected (H, W, 3), got {frame_bgr.shape}."
            )
        frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if frame_h < 2 or frame_w < 2:
            raise ValueError(
                f"Invalid frame_bgr dimensions: height={frame_h}, width={frame_w} must be >= 2."
            )
        return frame_h, frame_w

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 frame to normalized NCHW float32 tensor without modifying input."""
        img_resized = cv2.resize(
            frame_bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_norm = img_rgb * _NORM_SCALE + _NORM_BIAS
        chw = img_norm.transpose(2, 0, 1)
        return np.ascontiguousarray(chw[np.newaxis, :, :, :], dtype=np.float32)

    def _decode_output(
        self, raw_output: np.ndarray, frame_w: int, frame_h: int
    ) -> list[dict[str, Any]]:
        """Decode UFLD grid logits into original-frame pixel coordinates."""
        processed = np.asarray(raw_output)
        if processed.ndim == 4 and processed.shape[0] == 1:
            processed = processed[0]
        elif processed.ndim != 3:
            processed = np.squeeze(processed)
            if processed.ndim != 3:
                return []

        griding_num = processed.shape[0] - 1
        cls_num_per_lane = processed.shape[1]
        max_lanes = processed.shape[2]
        row_anchor = (
            self.row_anchor
            if len(self.row_anchor) == cls_num_per_lane
            else self._resolve_row_anchors(cls_num_per_lane)
        )

        # Follow official UFLD / ibaiGorordo decoding:
        # 1. Reverse row dimension
        processed_rev = processed[:, ::-1, :]
        # 2. Softmax over grid cells (excluding the background/absence bin at index griding_num)
        prob = _softmax_axis0(processed_rev[:-1, :, :])
        idx = (np.arange(griding_num, dtype=np.float32) + 1.0).reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        # 3. Zero out locations where argmax predicts the absence class (griding_num)
        max_indices = np.argmax(processed_rev, axis=0)
        loc[max_indices == griding_num] = 0.0

        col_sample = np.linspace(0, self.input_width - 1, griding_num)
        col_sample_w = float(col_sample[1] - col_sample[0]) if griding_num > 1 else 1.0
        # Row anchors always use the training reference height, even for exports
        # whose input resolution differs from the original 288 pixels.
        anchor_ref_h = 288.0

        candidate_lanes: list[list[list[int]]] = []

        for lane_idx in range(max_lanes):
            lane_col = loc[:, lane_idx]
            # Require more than 2 valid grid predictions as in reference UFLD decoder
            if int(np.sum(lane_col != 0.0)) <= 2:
                continue

            raw_points: list[tuple[int, int]] = []
            for point_num in range(cls_num_per_lane):
                grid_pos = float(lane_col[point_num])
                if grid_pos <= 0.0 or not math.isfinite(grid_pos):
                    continue

                anchor_y = float(row_anchor[cls_num_per_lane - 1 - point_num])
                x_orig = int(grid_pos * col_sample_w * frame_w / float(self.input_width)) - 1
                y_orig = int(frame_h * (anchor_y / anchor_ref_h)) - 1

                if 0 <= x_orig < frame_w and 0 <= y_orig < frame_h:
                    raw_points.append((x_orig, y_orig))

            if len(raw_points) < 2:
                continue

            # Sort by y ascending (top to bottom) and deduplicate identical y coordinates
            raw_points.sort(key=lambda pt: pt[1])
            dedup_points: list[list[int]] = []
            seen_y: set[int] = set()
            for x_val, y_val in raw_points:
                if y_val not in seen_y:
                    seen_y.add(y_val)
                    dedup_points.append([int(x_val), int(y_val)])

            if len(dedup_points) >= 2:
                candidate_lanes.append(dedup_points)

        # Sort lanes from left to right at the reference y (0.85 * (H - 1))
        ref_y = 0.85 * float(frame_h - 1)
        candidate_lanes.sort(key=lambda pts: _interpolate_x_at_y(pts, ref_y))

        lanes_out: list[dict[str, Any]] = []
        for idx_sorted, pts in enumerate(candidate_lanes):
            lanes_out.append(
                {
                    "lane_index": int(idx_sorted),
                    "points": pts,
                    "confidence": None,
                }
            )

        return lanes_out

    def reset(self) -> None:
        """Forget automatic camera framing between video streams."""
        self._selected_crop = self.road_crop_ratio
        self._frame_size = None

    def _detect_crop(self, frame_bgr: np.ndarray, crop: float) -> list[dict[str, Any]]:
        height, width = frame_bgr.shape[:2]
        top = round(height * crop)
        tensor = self._preprocess(frame_bgr[top:])
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        lanes = self._decode_output(output, frame_w=width, frame_h=height - top)
        for lane in lanes:
            for point in lane['points']:
                point[1] += top
        return lanes

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Run UFLD ONNX lane detection on a single BGR frame without modifying frame_bgr."""
        frame_h, frame_w = self._validate_frame(frame_bgr)

        t0 = time.perf_counter()
        if self._frame_size != (frame_w, frame_h):
            self.reset()
            self._frame_size = (frame_w, frame_h)
        response = marking_response(frame_bgr)
        if self._selected_crop is None:
            horizon = estimate_horizon(response)
            # Calibrate once so per-frame crop changes do not shake the overlay.
            # Bring the estimated horizon near the training framing. Do not
            # select by the number of detections: vehicles and compression can
            # change that count even when the camera itself has not moved.
            target_crop = max(0.0, min(0.45, (horizon - 0.55) / 0.45))
            self._selected_crop = (min((0.2, 0.35, 0.45), key=lambda crop: abs(crop - target_crop))
                                   if horizon > 0.64 else 0.0)
        proposed = self._detect_crop(frame_bgr, self._selected_crop)
        lanes, _ = refine_lanes(proposed, response, frame_w, frame_h,
                                max(0.52, self._selected_crop + 0.45 * (1 - self._selected_crop)))
        inference_ms = float((time.perf_counter() - t0) * 1000.0)

        return {
            "lanes": lanes,
            "frame_width": int(frame_w),
            "frame_height": int(frame_h),
            "inference_ms": inference_ms,
        }

"""NumPy decoder for the official UFLDv2 CULane and TuSimple exports.

Reference: cfzd/Ultra-Fast-Lane-Detection-v2, demo.py (MIT).
Use local three-cell expectations, row heads for the ego boundaries and
column heads for the two outside boundaries. Anchors refer to the full ROI,
including the top portion removed by the model's training crop.
"""

import numpy as np

OUTPUT_NAMES = ("loc_row", "loc_col", "exist_row", "exist_col")


def configuration(row_count: int) -> tuple[np.ndarray, float]:
    if row_count == 72:
        return np.linspace(0.42, 1.0, 72), 0.6
    if row_count == 56:
        return np.linspace(160 / 720, 710 / 720, 56), 0.8
    raise ValueError(f"Unsupported UFLDv2 row count: {row_count}; expected 72 or 56")


def decode(outputs: dict[str, np.ndarray], width: int, height: int) -> list[dict]:
    """Decode hybrid anchors, remove invalid pixels and merge repeated y rows."""
    if set(outputs) != set(OUTPUT_NAMES):
        raise ValueError("UFLDv2 requires loc_row, loc_col, exist_row and exist_col")
    tensors = {name: np.asarray(value) for name, value in outputs.items()}
    for name, tensor in tensors.items():
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[3] != 4:
            raise ValueError(f"Invalid UFLDv2 {name} shape: {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError(f"Non-finite UFLDv2 logits in {name}")
    row_anchors, _ = configuration(tensors['loc_row'].shape[2])
    lanes = []
    for axis, slots, minimum in (("row", (1, 2), 0.5), ("col", (0, 3), 0.25)):
        grid = tensors['loc_' + axis][0]
        exists = tensors['exist_' + axis][0]
        if grid.shape[0] < 2 or exists.shape != (2, grid.shape[1], 4):
            raise ValueError(f"Inconsistent UFLDv2 {axis} heads")
        anchors = row_anchors if axis == 'row' else np.linspace(0, 1, grid.shape[1])
        for slot in slots:
            present = exists[:, :, slot].argmax(axis=0) == 1
            if present.sum() <= minimum * len(present):
                continue
            indices = np.flatnonzero(present)
            peaks = grid[:, indices, slot].argmax(axis=0)
            cells = peaks[:, None] + np.array([-1, 0, 1])
            available = (cells >= 0) & (cells < grid.shape[0])
            values = grid[np.clip(cells, 0, grid.shape[0] - 1), indices[:, None], slot]
            values = np.where(available, values, -np.inf)
            weights = np.exp(values - values.max(axis=1, keepdims=True))
            locations = ((weights * cells).sum(axis=1) / weights.sum(axis=1) + 0.5) / (grid.shape[0] - 1)
            xs, ys = ((locations * width, anchors[indices] * height) if axis == 'row'
                      else (anchors[indices] * width, locations * height))
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if axis == 'col':
                # Column-head anchors are ordered horizontally. Outside lane
                # markings should progress toward the vehicle monotonically,
                # apart from small noise. Sorting a folded y(x) prediction by
                # y would otherwise turn it into a large zigzag or a false
                # central line after duplicate rows are merged.
                direction = -1 if slot == 0 else 1
                backward = np.maximum(-direction * np.diff(ys[inside]), 0)
                if backward.sum() > .08 * height:
                    continue
            points = {}
            for x, y in zip(xs[inside].astype(int), ys[inside].astype(int)):
                points.setdefault(int(y), []).append(int(x))
            polyline = [[points[y][0] if len(points[y]) == 1 else int(np.median(points[y])), y]
                        for y in sorted(points)]
            if len(polyline) >= 3:
                steps = np.diff(np.asarray(polyline), axis=0)
                if np.any((np.abs(steps[:, 0]) > .06 * width) & (steps[:, 1] < .04 * height)):
                    continue
                # Existence is model output, not a calibrated accuracy score.
                margins = exists[1, present, slot] - exists[0, present, slot]
                confidence = float(np.mean(1 / (1 + np.exp(-np.clip(margins, -80, 80)))))
                lanes.append(dict(lane_index=len(lanes), points=polyline, confidence=confidence))
    lanes.sort(key=lambda lane: np.interp(0.85 * height,
                                          np.asarray(lane['points'])[:, 1],
                                          np.asarray(lane['points'])[:, 0]))
    for index, lane in enumerate(lanes):
        lane['lane_index'] = index
    return lanes

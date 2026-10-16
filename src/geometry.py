"""Image-coordinate ego-lane geometry; instantaneous demo labels, not ADAS.

No calibration, temporal state, confidence synthesis, or metric distances.
Uses only the standard library; tracking points may also be NumPy arrays.
"""

from bisect import bisect_left
from math import isfinite
from numbers import Integral, Real


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _interpolate(points, y):
    """Interpolate sorted, unique rows; caller must ensure domain coverage."""
    index = bisect_left(points, y, key=lambda point: point[1])
    if index < len(points) and points[index][1] == y:
        return points[index][0]
    x0, y0 = points[index - 1]
    x1, y1 = points[index]
    return x0 + (x1 - x0) * ((y - y0) / (y1 - y0))


def _points(raw, width, height):
    """Reject malformed points and ambiguous rows; collapse exact duplicates."""
    if isinstance(raw, (str, bytes, dict)):
        return None
    try:
        rows = {}
        for point in raw:
            if len(point) != 2:
                return None
            x, y = point
            if not (_number(x) and _number(y)):
                return None
            if not (0 <= x < width and 0 <= y < height):
                return None
            if y in rows and rows[y] != x:
                return None
            rows[y] = x
        if len(rows) < 2:
            return None
        return [(float(rows[y]), float(y)) for y in sorted(rows)]
    except (TypeError, ValueError, OverflowError):
        return None


def analyze_lanes(tracking: dict, camera_center_ratio: float = 0.5) -> dict:
    """Analyze the currently observed ego-lane in image coordinates.

    Select the nearest boundaries strictly bracketing the camera at
    round(0.85*(H-1)); never extrapolate. Positive offset means camera right
    of lane center. Widths below 0.06*W are rejected as a demo heuristic.
    Warning thresholds on absolute offset are 0.15 and 0.25 (inclusive).

    Invalid ratio or missing predicted/missed fields raises ValueError.
    Other malformed observations invalidate the entire frame, including
    inconsistent metadata, duplicate IDs and conflicting x at the same y.
    A fresh prediction (missed=1) needs a real observation on the other side.
    Selected boundaries must not touch/cross anywhere in their common domain.

    UNKNOWN always clears both IDs, center and offset. Invalid frame size
    uses reference_y=0 and vehicle_center_x=0.0 as nonmeasurement sentinels.
    Curve and marking types remain UNKNOWN: P1 is intentionally unimplemented.
    Input is never mutated; sorting and crossing checks cost O(N log N).
    """
    if not _number(camera_center_ratio) or not 0 <= camera_center_ratio <= 1:
        raise ValueError("camera_center_ratio must be a finite number in [0, 1]")
    result = {
        "valid_lane": False,
        "reference_y": 0,
        "vehicle_center_x": 0.0,
        "lane_center_x": None,
        "left_track_id": None,
        "right_track_id": None,
        "offset_normalized": None,
        "warning": "UNKNOWN",
        "curve": "UNKNOWN",
        "lane_type_left": "UNKNOWN",
        "lane_type_right": "UNKNOWN",
    }
    if not isinstance(tracking, dict):
        return result
    lanes = tracking.get("lanes")
    if not isinstance(lanes, (list, tuple)):
        return result
    for lane in lanes:
        if isinstance(lane, dict):
            for field in ("predicted", "missed"):
                if field not in lane:
                    raise ValueError(f"tracking lane missing required field: {field}")
    width, height = tracking.get("frame_width"), tracking.get("frame_height")
    if not (_integer(width) and _integer(height) and width >= 2 and height >= 2):
        return result
    ref = round(0.85 * (height - 1))
    vehicle = float(camera_center_ratio * (width - 1))
    result.update(reference_y=int(ref), vehicle_center_x=vehicle)
    candidates = []
    ids = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            return result
        track_id = lane.get("track_id")
        predicted, missed = lane["predicted"], lane["missed"]
        if not _integer(track_id) or track_id in ids:
            return result
        ids.add(track_id)
        if not isinstance(predicted, bool) or not _integer(missed) or missed < 0:
            return result
        if predicted != (missed > 0):
            return result
        confidence = lane.get("confidence")
        if confidence is not None and (not _number(confidence) or not 0 <= confidence <= 1):
            return result
        points = _points(lane.get("points"), width, height)
        if points is None:
            return result
        if points[0][1] <= ref <= points[-1][1]:
            candidates.append((_interpolate(points, ref), lane, points))
    # A boundary exactly on the camera makes ego-lane membership ambiguous.
    if any(x == vehicle for x, _, _ in candidates):
        return result
    left = [item for item in candidates if item[0] < vehicle]
    right = [item for item in candidates if item[0] > vehicle]
    if not left or not right:
        return result
    left = max(left, key=lambda item: item[0])
    right = min(right, key=lambda item: item[0])
    xl, ll, pl = left
    xr, lr, pr = right
    # Equal positions from different tracks have no unique nearest identity.
    if sum(x == xl for x, _, _ in candidates) > 1 or sum(x == xr for x, _, _ in candidates) > 1:
        return result
    if xr - xl < 0.06 * width:
        return result
    if ll["missed"] > 1 or lr["missed"] > 1 or (ll["predicted"] and lr["predicted"]):
        return result
    start, end = max(pl[0][1], pr[0][1]), min(pl[-1][1], pr[-1][1])
    # Difference of piecewise linear boundaries is linear between union knots.
    rows = {start, end} | {y for _, y in pl + pr if start <= y <= end}
    if any(_interpolate(pr, y) <= _interpolate(pl, y) for y in rows):
        return result
    center = (xl + xr) / 2.0
    offset = (vehicle - center) / (xr - xl)
    magnitude = abs(offset)
    warning = "SAFE" if magnitude < 0.15 else "CAUTION" if magnitude < 0.25 else "WARNING"
    result.update(
        valid_lane=True, lane_center_x=center, offset_normalized=offset,
        left_track_id=int(ll["track_id"]), right_track_id=int(lr["track_id"]), warning=warning,
    )
    return result

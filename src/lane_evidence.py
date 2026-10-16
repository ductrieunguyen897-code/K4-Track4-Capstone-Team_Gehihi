"""Check UFLD proposals against bright white/yellow road markings.

The response measures local horizontal contrast, not model confidence.
It deliberately leaves unmarked/occluded boundaries unknown.
"""

import cv2
import numpy as np


def marking_response(frame: np.ndarray) -> np.ndarray:
    """Return a small contrast map; broad sky and vehicle panels score poorly."""
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (min(width, 960), max(2, round(height * min(width, 960) / width))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    kernel = max(15, round(small.shape[1] * 0.035)) | 1
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((1, kernel), np.uint8))
    white = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 125)
    yellow = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 115)
    return np.where(white | yellow, contrast, 0).astype(np.uint8)


def estimate_horizon(response: np.ndarray) -> float:
    """Estimate a road vanishing row from opposing marking segments, if visible."""
    height, width = response.shape
    mask = (response > 18).astype(np.uint8) * 255
    mask[:round(0.55 * height)] = 0
    segments = cv2.HoughLinesP(mask, 1, np.pi / 180, 25,
                               minLineLength=width * 0.04, maxLineGap=width * 0.02)
    left, right, rows = [], [], []
    for x0, y0, x1, y1 in (segments.reshape(-1, 4) if segments is not None else []):
        if abs(y1 - y0) < 0.025 * height:
            continue
        slope = (x1 - x0) / (y1 - y0)
        intercept = x0 - slope * y0
        bottom = slope * height + intercept
        if 0.2 < abs(slope) < 3:
            if slope < 0 and bottom < width * 0.5:
                left.append((slope, intercept))
            elif slope > 0 and bottom > width * 0.5:
                right.append((slope, intercept))
    for slope, intercept in left:
        for other_slope, other_intercept in right:
            y = (other_intercept - intercept) / (slope - other_slope)
            x = slope * y + intercept
            if 0.35 * height < y < 0.85 * height and 0.2 * width < x < 0.8 * width:
                rows.append(y / height)
    # Short fragments on curbs/vehicles pull the median upward in the image.
    # Use the upper quartile of plausible vanishing rows; this also reduces
    # crop-selection changes caused by normal video compression.
    return float(np.quantile(rows, 0.75)) if len(rows) >= 8 else 0.55


def refine_lanes(lanes: list[dict], response: np.ndarray, width: int, height: int,
                 road_start: float = 0.52) -> tuple[list[dict], float]:
    """Fit proposals to nearby paint evidence and return a support score.

    Searches are bounded to 1.8% of image width. A quadratic requires several
    supported rows over a meaningful vertical span; no boundary is invented
    when paint is absent. The returned confidence remains the detector's value.
    """
    sy, sx = response.shape[0] / height, response.shape[1] / width
    radius = max(3, round(0.018 * response.shape[1]))
    minimum_span = 0.3 * (0.97 - road_start) * height
    refined, scores = [], []
    for lane in lanes:
        points = np.asarray(lane['points'], dtype=float)
        lo = max(points[0, 1], road_start * height)
        hi = min(points[-1, 1], 0.97 * height)
        if hi - lo < minimum_span:
            continue
        ys = np.linspace(lo, hi, 80)
        expected = np.interp(ys, points[:, 1], points[:, 0])
        # Search all rows at once: repeated Python/NumPy calls per row were
        # a substantial part of inference time on full-HD dashcam footage.
        image_rows = np.clip(np.rint(ys * sy).astype(int), 0, response.shape[0] - 1)
        centers = np.rint(expected * sx).astype(int)
        offsets = np.arange(-radius, radius + 1)
        columns = centers[:, None] + offsets
        in_bounds = (columns >= 0) & (columns < response.shape[1])
        bands = response[image_rows[:, None], np.clip(columns, 0, response.shape[1] - 1)].astype(float)
        candidate_scores = np.where(in_bounds, bands - np.abs(offsets) * 0.7, -np.inf)
        best = candidate_scores.argmax(axis=1)
        chosen = np.arange(len(ys)), best
        supported = in_bounds[chosen] & (bands[chosen] >= 18)
        if supported.sum() < 10:
            continue
        hits = np.column_stack((columns[chosen][supported] / sx, ys[supported]))
        keep = np.ones(len(hits), dtype=bool)
        for _ in range(3):
            if keep.sum() < 10:
                break
            coeff = np.polyfit(hits[keep, 1] / height, hits[keep, 0] / width, 2)
            residual = np.abs(np.polyval(coeff, hits[:, 1] / height) * width - hits[:, 0])
            keep = residual <= max(0.006 * width, 2.5 * float(np.median(residual)))
        if keep.sum() < 10 or np.ptp(hits[keep, 1]) < minimum_span:
            continue
        coeff = np.polyfit(hits[keep, 1] / height, hits[keep, 0] / width, 2)
        # Limit the visible curve to supported road rows; small gaps cover dashes.
        low = max(lo, hits[keep, 1].min())
        # A near-field dash gap can exceed 4% of the full image. Continue
        # along existing model samples across a bounded gap in the road ROI.
        # Never extend past the model's own observed domain.
        high = min(hi, hits[keep, 1].max() + minimum_span)
        rows = np.unique(np.rint(np.linspace(low, high, 40)).astype(int))
        base = np.interp(rows, points[:, 1], points[:, 0])
        supported_rows = np.clip(rows, hits[keep, 1].min(), hits[keep, 1].max())
        fitted = np.polyval(coeff, supported_rows / height) * width
        supported_base = np.interp(supported_rows, points[:, 1], points[:, 0])
        # Preserve the model's curve outside a narrow correction band. A fit
        # through intermittent dashes must not bend onto another boundary.
        xs = np.rint(base + np.clip(fitted - supported_base, -0.012 * width, 0.012 * width)).astype(int)
        inside = (xs >= 0) & (xs < width)
        pts = np.column_stack((xs[inside], rows[inside])).tolist()
        if len(pts) < 2:
            continue
        refined.append({**lane, 'points': pts})
        scores.append(float(keep.sum() / len(ys)) * float(np.ptp(hits[keep, 1]) / height))
    refined.sort(key=lambda lane: np.interp(0.85 * height, np.asarray(lane['points'])[:, 1], np.asarray(lane['points'])[:, 0]))
    # Converging markings become indistinguishable near the vanishing point.
    # Keep only their separated lower portions instead of drawing an X there.
    for left, right in zip(refined, refined[1:]):
        pl, pr = np.asarray(left['points']), np.asarray(right['points'])
        lo, hi = max(pl[0, 1], pr[0, 1]), min(pl[-1, 1], pr[-1, 1])
        if hi <= lo:
            continue
        rows = np.unique(np.concatenate((pl[:, 1], pr[:, 1])))
        rows = rows[(rows >= lo) & (rows <= hi)]
        gap = np.interp(rows, pr[:, 1], pr[:, 0]) - np.interp(rows, pl[:, 1], pl[:, 0])
        ambiguous = rows[gap <= 0.004 * width]
        if len(ambiguous) and ambiguous[-1] < lo + 0.2 * (hi - lo):
            start = int(ambiguous[-1]) + 1
            for lane, pts in ((left, pl), (right, pr)):
                first = [int(round(np.interp(start, pts[:, 1], pts[:, 0]))), start]
                lane['points'] = [first] + pts[pts[:, 1] > start].tolist()
    for index, lane in enumerate(refined):
        lane['lane_index'] = index
    return refined, sum(sorted(scores, reverse=True)[:2])

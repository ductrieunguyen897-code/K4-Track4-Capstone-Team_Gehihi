"""Lane visualization and frame annotation module for LaneGuard AI.

Follows API Contract specified in PROJECT_OVERVIEW.md (Section 5.5).
"""

from typing import Any, Dict, List
import cv2
import numpy as np


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
    gap_length: int = 8,
    phase_offset: float = 0.0,
) -> None:
    """Draw a dashed line between two points."""
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1e-3:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    curr_dist = -(phase_offset % (dash_length + gap_length))
    while curr_dist < dist:
        end_dist = min(curr_dist + dash_length, dist)
        if end_dist > 0:
            start_dist = max(curr_dist, 0.0)
            start_pt = (int(pt1[0] + dx * start_dist), int(pt1[1] + dy * start_dist))
            end_pt = (int(pt1[0] + dx * end_dist), int(pt1[1] + dy * end_dist))
            cv2.line(img, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
        curr_dist += dash_length + gap_length


def _interpolate_x_at_y(points: list[list[int]], target_y: float) -> float | None:
    """Linearly interpolate x coordinate at target_y given sorted points [[x, y], ...]."""
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: p[1])
    min_y = pts[0][1]
    max_y = pts[-1][1]
    if target_y < min_y or target_y > max_y:
        return None

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if y1 <= target_y <= y2:
            if y2 == y1:
                return float(x1)
            t = (target_y - y1) / float(y2 - y1)
            return float(x1 + t * (x2 - x1))
    return None


def render_frame(
    frame_bgr: np.ndarray,
    tracking: dict | None = None,
    analysis: dict | None = None,
) -> np.ndarray:
    """Render lane lines, centers, and departure warning overlays on a video frame.

    Args:
        frame_bgr: Input BGR frame as uint8 numpy array of shape (H, W, 3).
        tracking: Tracking output containing 'lanes', 'frame_width', 'frame_height'.
        analysis: Geometry and warning analysis containing 'valid_lane', 'warning', etc.

    Returns:
        A NEW BGR uint8 numpy array with identical shape (H, W, 3).
        Original input is never modified in-place.
    """
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        raise ValueError("frame_bgr must be a 3D numpy array (H, W, 3)")

    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    tracking = tracking or {}
    analysis = analysis or {}

    lanes: List[Dict[str, Any]] = tracking.get("lanes", [])
    valid_lane = bool(analysis.get("valid_lane", False))
    warning = str(analysis.get("warning", "UNKNOWN")).upper()
    if warning not in {"SAFE", "CAUTION", "WARNING", "UNKNOWN"}:
        warning = "UNKNOWN"

    reference_y = analysis.get("reference_y")
    if reference_y is None or not (0 <= reference_y < h):
        reference_y = int(round(0.85 * (h - 1)))
    else:
        reference_y = int(round(reference_y))

    lane_center_x = analysis.get("lane_center_x")
    vehicle_center_x = analysis.get("vehicle_center_x")
    offset_norm = analysis.get("offset_normalized")
    curve = str(analysis.get("curve", "UNKNOWN"))
    left_track_id = analysis.get("left_track_id")
    right_track_id = analysis.get("right_track_id")

    # 1. Draw Lane Polylines
    # Track dictionary for fast lookup when highlighting left/right tracks
    lane_by_id: dict[int, list[list[int]]] = {}

    for lane in lanes:
        pts = lane.get("points", [])
        if not pts or len(pts) < 2:
            continue

        pts_clean = []
        for p in pts:
            if len(p) >= 2 and not (np.isnan(p[0]) or np.isnan(p[1])):
                pts_clean.append([int(round(p[0])), int(round(p[1]))])

        if len(pts_clean) < 2:
            continue

        track_id = lane.get("track_id")
        if track_id is not None:
            lane_by_id[track_id] = pts_clean

        is_predicted = bool(lane.get("predicted", False))
        pts_arr = np.array(pts_clean, dtype=np.int32).reshape((-1, 1, 2))

        if is_predicted:
            # Predicted lane: Amber/Orange with dashed / thinner polyline
            color = (0, 165, 255)  # BGR orange
            # Mark retained history clearly, since it is not a current observation.
            phase = 0.0
            for start, end in zip(pts_clean, pts_clean[1:]):
                _draw_dashed_line(canvas, tuple(start), tuple(end), color=color, thickness=2,
                                  dash_length=6, gap_length=8, phase_offset=phase)
                phase += float(np.hypot(end[0] - start[0], end[1] - start[1]))
            tag = f"Trk#{track_id} (P)" if track_id is not None else "Lane (P)"
        else:
            # Observed lane: Bright Cyan
            color = (255, 220, 0)  # BGR Cyan-ish
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)
            tag = f"Trk#{track_id}" if track_id is not None else "Lane"

        # Label track ID near lowest y (closest to camera/bottom)
        label_pt = pts_clean[-1]
        cv2.putText(
            canvas,
            tag,
            (label_pt[0] - 15, min(h - 10, label_pt[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw Lane Region / Corridor if valid_lane is True
    # Fill subtle transparent corridor between left and right tracks
    if valid_lane and left_track_id in lane_by_id and right_track_id in lane_by_id:
        left_pts = sorted(lane_by_id[left_track_id], key=lambda p: p[1])
        right_pts = sorted(lane_by_id[right_track_id], key=lambda p: p[1])

        # Find overlapping y range
        min_y = max(left_pts[0][1], right_pts[0][1])
        max_y = min(left_pts[-1][1], right_pts[-1][1])

        if max_y - min_y >= 20:
            y_samples = np.linspace(min_y, max_y, num=15)
            poly_left = []
            poly_right = []
            for y_s in y_samples:
                xl = _interpolate_x_at_y(left_pts, y_s)
                xr = _interpolate_x_at_y(right_pts, y_s)
                if xl is not None and xr is not None and xr > xl:
                    poly_left.append([int(round(xl)), int(round(y_s))])
                    poly_right.append([int(round(xr)), int(round(y_s))])

            if len(poly_left) >= 2 and len(poly_right) >= 2:
                poly_pts = np.array(poly_left + poly_right[::-1], dtype=np.int32)
                overlay = canvas.copy()
                # Color tint based on warning status
                corridor_color = (0, 200, 0) if warning == "SAFE" else ((0, 215, 255) if warning == "CAUTION" else (0, 0, 230))
                cv2.fillPoly(overlay, [poly_pts], corridor_color)
                cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # 3. Draw Reference Y and Centerline Markers (ONLY at reference_y per contract)
    if valid_lane and lane_center_x is not None:
        lc_x = int(round(lane_center_x))
        # Draw lane width span at reference_y if left and right lanes cross it
        xl_ref = _interpolate_x_at_y(lane_by_id.get(left_track_id, []), reference_y) if left_track_id else None
        xr_ref = _interpolate_x_at_y(lane_by_id.get(right_track_id, []), reference_y) if right_track_id else None

        if xl_ref is not None and xr_ref is not None and xr_ref > xl_ref:
            # Reference baseline at reference_y
            _draw_dashed_line(
                canvas,
                (int(round(xl_ref)), reference_y),
                (int(round(xr_ref)), reference_y),
                color=(200, 200, 200),
                thickness=1,
            )

        # Draw Lane Center marker at reference_y (Crosshair + Circle)
        cv2.circle(canvas, (lc_x, reference_y), 7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (lc_x, reference_y), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        # Draw Vehicle Center marker at reference_y if provided
        if vehicle_center_x is not None:
            vc_x = int(round(vehicle_center_x))
            cv2.drawMarker(canvas, (vc_x, reference_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(canvas, (vc_x, reference_y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            # Line connecting vehicle center and lane center
            cv2.line(canvas, (vc_x, reference_y), (lc_x, reference_y), (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "Lane Center",
            (lc_x - 35, reference_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 4. Top Telemetry Banner & Warning HUD
    badge_colors = {
        "SAFE": (0, 160, 0),         # Green
        "CAUTION": (0, 180, 230),     # Amber / Orange-Yellow
        "WARNING": (0, 0, 220),       # Red
        "UNKNOWN": (80, 80, 80),      # Neutral Gray
    }
    badge_bg = badge_colors.get(warning, (80, 80, 80))

    # Top HUD background panel (semi-transparent)
    hud_h = 74
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(hud_overlay, 0.75, canvas, 0.25, 0, canvas)

    # Title
    cv2.putText(
        canvas,
        "LaneGuard AI",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "LDW Demo",
        (160, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Status Badge
    badge_text = warning if valid_lane else "LANE UNKNOWN"
    badge_w = 140 if not valid_lane else 110
    badge_x1 = w - badge_w - 16
    badge_y1 = 12
    badge_x2 = w - 16
    badge_y2 = 46
    cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
    cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

    text_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = badge_x1 + (badge_w - text_size[0]) // 2
    ty = badge_y1 + (badge_y2 - badge_y1 + text_size[1]) // 2
    cv2.putText(canvas, badge_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry Line: Offset, Tracks, Curve
    if valid_lane and offset_norm is not None:
        direction = "Right" if offset_norm > 0 else ("Left" if offset_norm < 0 else "Center")
        offset_str = f"Offset: {offset_norm:+.3f} ({direction})"
    else:
        offset_str = "Offset: N/A"

    l_id_str = f"#{left_track_id}" if left_track_id is not None else "-"
    r_id_str = f"#{right_track_id}" if right_track_id is not None else "-"
    tracks_str = f"Bounds: L[{l_id_str}] R[{r_id_str}]"
    curve_str = f"Curve: {curve}"

    metrics_text = f"{offset_str}   |   {tracks_str}   |   {curve_str}"
    cv2.putText(
        canvas,
        metrics_text,
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 5. Bottom Disclaimer Overlay (Required by Contract: "Lane estimate — demo only")
    disclaimer = "Lane estimate — demo only | Not for actual vehicle guidance"
    d_size = cv2.getTextSize(disclaimer, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    dx = (w - d_size[0]) // 2
    dy = h - 10
    # Small dark backing behind disclaimer
    cv2.rectangle(canvas, (dx - 8, dy - d_size[1] - 4), (dx + d_size[0] + 8, dy + 4), (10, 10, 10), -1)
    cv2.putText(canvas, disclaimer, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    return canvas

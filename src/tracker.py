"""CPU lane association, EMA smoothing, and bounded hold-last tracking.

Uses NumPy and SciPy's assignment solver. Call update in frame order
and reset between videos. No detector, model, or future frame is accessed.
"""

from numbers import Integral, Real
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment


class LaneTracker:
    """Track lane polylines by geometry, independently of detector lane_index.

    Global one-to-one matching uses mean horizontal distance / frame width
    at eight evenly spaced y samples in each pair's observed overlap. Pairs
    need two distinct shared y values and distance <= 0.08. EMA uses the union
    of polyline vertices within the current observation, never extrapolating.
    Newly observed rows are retained without smoothing against missing history. Missing
    tracks hold their last smoothed points, without motion prediction or
    confidence inflation. IDs increase until reset; output is ordered by ID.

    This stateful tracker is intended for one sequential video stream.
    Matching can swap IDs when lanes cross or move abruptly. The
    threshold needs validation on real video; it is an MVP default.
    """

    def __init__(self, alpha: float = 0.35, max_missed: int = 3) -> None:
        """Configure smoothing and retention, then initialize an empty stream.

        Args:
            alpha: Observation weight in [0, 1]; zero holds prior x, one uses
                observed x. The default balances smoothing and responsiveness.
            max_missed: Nonnegative number of missing frames to retain tracks.

        Raises:
            ValueError: If a parameter is nonnumeric, nonfinite, boolean, or
                outside its allowed range, or max_missed is not an integer.
        """
        if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                or not np.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("alpha must be a finite number in [0, 1]")
        if (isinstance(max_missed, bool) or not isinstance(max_missed, Integral)
                or max_missed < 0):
            raise ValueError("max_missed must be a nonnegative integer")
        self._alpha = float(alpha)
        self._max_missed = int(max_missed)
        self.reset()

    def reset(self) -> None:
        """Clear all tracks, forget frame size, and restart IDs at 1."""
        self._tracks: dict[int, dict] = {}
        self._next_id = 1
        self._frame_size: tuple[int, int] | None = None

    def update(self, detection: dict) -> dict:
        """Associate current observations, smooth matches, and age missing tracks.

        Args:
            detection: Detector dictionary with lanes, frame_width and
                frame_height. Each lane supplies at least two finite [x, y]
                pixel points with distinct y values, inside the original
                frame, and confidence in [0, 1] or None. Points are sorted by
                y internally. lane_index and inference_ms are not used.

        Returns:
            Contract dictionary containing lanes, original frame_width and
            frame_height, and elapsed tracking_ms measured by perf_counter.
            Each lane has track_id, points, confidence, predicted, and missed.
            Input data is unchanged; returned points do not alias state.

        Raises:
            ValueError: For malformed detection or changed frame dimensions.
                Validation failures leave existing state unchanged. Call
                reset before a new video or a different frame size.
        """
        started = perf_counter()
        width, height, observations = self._validate(detection)
        if self._frame_size is not None and self._frame_size != (width, height):
            raise ValueError("Frame dimensions changed; call reset() for a new video")

        candidates = []
        for track_id, track in self._tracks.items():
            previous = track["points"]
            for index, observation in enumerate(observations):
                current = observation["points"]
                lo = max(previous[0, 1], current[0, 1])
                hi = min(previous[-1, 1], current[-1, 1])
                if lo >= hi:
                    continue
                ys = np.linspace(lo, hi, 8)
                distance = float(np.mean(np.abs(
                    np.interp(ys, previous[:, 1], previous[:, 0])
                    - np.interp(ys, current[:, 1], current[:, 0])
                )) / width)
                if distance <= 0.08:
                    candidates.append((distance, track_id, index))

        matched_tracks, matched_observations = set(), set()
        # Minimize the total association cost after maximizing valid matches.
        # Greedy nearest-pair selection can consume the only observation that
        # another track can match, creating a ghost and a duplicate ID.
        track_ids = sorted(self._tracks)
        assignments = []
        if track_ids and observations:
            positions = {track_id: i for i, track_id in enumerate(track_ids)}
            unmatched_cost = float(len(track_ids) + len(observations) + 1)
            costs = np.full((len(track_ids), len(observations) + len(track_ids)), unmatched_cost)
            costs[:, :len(observations)] = unmatched_cost * 2
            for distance, track_id, index in candidates:
                costs[positions[track_id], index] = distance
            row_indices, column_indices = linear_sum_assignment(costs)
            assignments = [(track_ids[row], col) for row, col in zip(row_indices, column_indices)
                           if col < len(observations) and costs[row, col] < unmatched_cost]
        for track_id, index in assignments:
            track = self._tracks[track_id]
            observation = observations[index]
            track["points"] = self._smooth(track["points"], observation["points"])
            track["confidence"] = observation["confidence"]
            track["missed"] = 0
            matched_tracks.add(track_id)
            matched_observations.add(index)

        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                self._tracks[track_id]["missed"] += 1
                if self._tracks[track_id]["missed"] > self._max_missed:
                    del self._tracks[track_id]

        for index, observation in enumerate(observations):
            if index not in matched_observations:
                self._tracks[self._next_id] = {**observation, "missed": 0}
                self._next_id += 1

        self._frame_size = (width, height)
        lanes = [
            {"track_id": track_id, "points": track["points"].tolist(),
             "confidence": track["confidence"], "predicted": track["missed"] > 0,
             "missed": track["missed"]}
            for track_id, track in sorted(self._tracks.items())
        ]
        return {"lanes": lanes, "frame_width": width, "frame_height": height,
                "tracking_ms": (perf_counter() - started) * 1000.0}

    def _smooth(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Apply EMA in the overlap and retain newly visible observed rows.

        Args:
            previous: Sorted Nx2 smoothed pixel points.
            current: Sorted Mx2 observed pixel points with positive y overlap.

        Returns:
            New sorted pixel points covering the current observation.
        """
        lo = max(previous[0, 1], current[0, 1])
        hi = min(previous[-1, 1], current[-1, 1])
        ys = np.unique(np.concatenate((previous[:, 1], current[:, 1])))
        ys = ys[(ys >= current[0, 1]) & (ys <= current[-1, 1])]
        xs = np.interp(ys, current[:, 1], current[:, 0])
        overlap = (ys >= lo) & (ys <= hi)
        xs[overlap] = (self._alpha * xs[overlap]
                       + (1 - self._alpha) * np.interp(ys[overlap], previous[:, 1], previous[:, 0]))
        return np.column_stack((xs, ys))

    @staticmethod
    def _validate(detection: dict) -> tuple[int, int, list[dict]]:
        """Validate and copy a complete frame before changing tracker state.

        Args:
            detection: Detector output to validate.

        Returns:
            Width, height, and copied lanes with sorted float64 points and
            Python float/None confidence. Metadata unused by tracking is ignored.

        Raises:
            ValueError: If required fields, dimensions, points, or confidence
                violate the tracking contract. Duplicate y values are rejected
                because they do not define a single-valued x(y) polyline.
        """
        if not isinstance(detection, dict):
            raise ValueError("detection must be a dictionary")
        dimensions = []
        for name in ("frame_width", "frame_height"):
            value = detection.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            dimensions.append(int(value))
        width, height = dimensions
        if not isinstance(detection.get("lanes"), list):
            raise ValueError("lanes must be a list (use [] for no detections)")
        observations = []
        for index, lane in enumerate(detection["lanes"]):
            if not isinstance(lane, dict):
                raise ValueError(f"lane {index} must be a dictionary")
            try:
                points = np.array(lane.get("points"), dtype=np.float64, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"lane {index}: points must be numeric Nx2") from exc
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all()):
                raise ValueError(f"lane {index}: points must be finite Nx2 with N >= 2")
            if (np.any(points < 0) or np.any(points[:, 0] > width - 1)
                    or np.any(points[:, 1] > height - 1)):
                raise ValueError(f"lane {index}: points must lie inside the frame")
            points = points[np.argsort(points[:, 1], kind="stable")]
            if np.any(np.diff(points[:, 1]) <= 0):
                raise ValueError(f"lane {index}: points need distinct y values")
            confidence = lane.get("confidence")
            if confidence is not None:
                if (isinstance(confidence, bool) or not isinstance(confidence, Real)
                        or not np.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError(f"lane {index}: confidence must be None or in [0, 1]")
                confidence = float(confidence)
            observations.append({"points": points, "confidence": confidence})
        return width, height, observations

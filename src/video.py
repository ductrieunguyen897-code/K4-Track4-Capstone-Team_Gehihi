"""Video processing pipeline for LaneGuard AI.

Follows API Contract specified in PROJECT_OVERVIEW.md (Section 5.5).
Executes pipeline in exact order:
LaneDetector.detect() -> LaneTracker.update() -> analyze_lanes() -> render_frame()
"""

import csv
from functools import partial
import logging
import math
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional
import cv2

from src.visualizer import render_frame


def _find_ffmpeg_executable() -> Optional[str]:
    """Find a usable ffmpeg executable from PATH or imageio_ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe and os.path.exists(exe):
        return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return None


def _transcode_to_h264(raw_mp4_path: str, target_mp4_path: str) -> bool:
    """Transcode a raw MP4 file to browser-compatible H.264 (yuv420p).

    Uses subprocess without shell=True for security.
    """
    ffmpeg_exe = _find_ffmpeg_executable()
    if not ffmpeg_exe:
        return False

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        raw_mp4_path,
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        target_mp4_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result.returncode == 0 and os.path.exists(target_mp4_path) and os.path.getsize(target_mp4_path) > 0
    except Exception:
        return False


def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    camera_center_ratio: float = 0.5,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    detector: Any = None,
    tracker: Any = None,
    analyzer: Any = None,
    road_crop_ratio: Optional[float] = None,
    device: str = "auto",
    start_time_s: float = 0.0,
    reference_y_ratio: float = 0.9,
) -> Dict[str, Any]:
    """Process an input MP4 video through detection, tracking, analysis, and rendering.

    Args:
        input_path: Path to the input MP4 video file.
        output_path: Target path for the annotated output video.
        model_path: Path to the UFLD ONNX model weights.
        camera_center_ratio: Horizontal ratio for vehicle camera center (default 0.5).
        max_frames: Optional limit on the number of frames to process.
        progress_callback: Optional callback(fraction, desc) for progress updates.
        detector: Optional pre-instantiated LaneDetector (for testing / dependency injection).
        tracker: Optional pre-instantiated LaneTracker (for testing / dependency injection).
        analyzer: Optional analyze_lanes callable (for testing / dependency injection).
        road_crop_ratio: None for automatic framing; otherwise crop 0 to 50% from the top.
        device: auto selects available CUDA with CPU fallback; cpu forces CPU.
        start_time_s: Start analysis at this source timestamp, preserving CSV timestamps.
        reference_y_ratio: Image row used for vehicle position; default is near the vehicle.

    Returns:
        Dictionary following the contract:
        {
            "output_video": str,
            "csv_path": str,
            "summary": {
                "frames_processed": int,
                "processing_seconds": float,
                "processing_fps": float,
                "video_fps": float,
                "avg_inference_ms": float,
                "valid_lane_frames": int,
                "warning_frames": int,
                "failed_frames": int
            }
        }
    """
    # 1. Validate input video
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer or None")
    if isinstance(start_time_s, bool) or not isinstance(start_time_s, (int, float)) or not math.isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("start_time_s must be a finite nonnegative number")
    if isinstance(reference_y_ratio, bool) or not isinstance(reference_y_ratio, (int, float)) or not math.isfinite(reference_y_ratio) or not 0 < reference_y_ratio < 1:
        raise ValueError("reference_y_ratio must be a finite number in (0, 1)")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0 or (video_fps != video_fps):
        video_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    start_frame = int(round(start_time_s * video_fps))
    if total_video_frames > 0 and start_frame >= total_video_frames:
        cap.release()
        raise ValueError("Start time is outside the video duration")
    calibration_frame = None
    if start_frame:
        ok, calibration_frame = cap.read()
        if not ok or not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            cap.release()
            raise ValueError("Could not seek to the requested start time")

    # Determine frame limit within the remaining source interval.
    target_frames = total_video_frames - start_frame if total_video_frames > 0 else None
    if max_frames is not None and max_frames > 0:
        target_frames = min(target_frames, max_frames) if target_frames else max_frames

    # Prepare output paths
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    intermediate_video_path = os.path.splitext(output_path)[0] + "_raw.mp4"

    # 2. Initialize or inject modules
    if detector is None:
        if not os.path.exists(model_path):
            cap.release()
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'. "
                "Please ensure the UFLD ONNX model is downloaded and placed at this path."
            )
        try:
            from src.detector import LaneDetector
            detector = LaneDetector(model_path=model_path, device=device, road_crop_ratio=road_crop_ratio)
        except ImportError as err:
            cap.release()
            raise RuntimeError(
                "LaneDetector module (src.detector) is not yet available in this branch. "
                "Ensure src/detector.py is available or provide a mock detector for testing."
            ) from err

    if tracker is None:
        try:
            from src.tracker import LaneTracker
            tracker = LaneTracker(alpha=0.35, max_missed=3)
        except ImportError as err:
            cap.release()
            raise RuntimeError(
                "LaneTracker module (src.tracker) is not yet available in this branch. "
                "Ensure src/tracker.py is available or provide a mock tracker for testing."
            ) from err

    if analyzer is None:
        try:
            from src.geometry import analyze_lanes
            analyzer = partial(analyze_lanes, reference_y_ratio=reference_y_ratio)
        except ImportError as err:
            cap.release()
            raise RuntimeError(
                "analyze_lanes function (src.geometry) is not yet available in this branch. "
                "Ensure src/geometry.py is available or provide a mock analyzer for testing."
            ) from err

    # Reset tracker state at the start of a new video
    if hasattr(detector, "reset") and callable(detector.reset):
        detector.reset()
    if calibration_frame is not None and callable(getattr(detector, 'calibrate', None)):
        detector.calibrate(calibration_frame)
    if hasattr(tracker, "reset") and callable(tracker.reset):
        tracker.reset()

    # 3. Setup VideoWriter & CSV Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(intermediate_video_path, fourcc, video_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize cv2.VideoWriter for {intermediate_video_path}")

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "frame_index",
        "time_s",
        "detector_ms",
        "tracker_ms",
        "offset_normalized",
        "warning",
        "curve",
        "left_track_id",
        "right_track_id",
        "valid_lane",
    ]
    csv_writer.writerow(csv_header)

    frames_processed = 0
    valid_lane_frames = 0
    warning_frames = 0
    failed_frames = 0
    total_detector_ms = 0.0
    frame_errors = []

    start_wall_clock = time.time()

    try:
        while cap.isOpened():
            if max_frames is not None and frames_processed >= max_frames:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx = start_frame + frames_processed
            time_s = frame_idx / video_fps

            det_ms = 0.0
            trk_ms = 0.0

            try:
                # PIPELINE ORDER: detect -> update -> analyze -> render
                detection = detector.detect(frame)
                det_ms = float(detection.get("inference_ms", 0.0))
                total_detector_ms += det_ms

                tracking = tracker.update(detection)
                trk_ms = float(tracking.get("tracking_ms", 0.0))

                analysis = analyzer(tracking, camera_center_ratio=camera_center_ratio)

                rendered = render_frame(frame, tracking, analysis)
                writer.write(rendered)

                # Record statistics
                is_valid = bool(analysis.get("valid_lane", False))
                warn_state = str(analysis.get("warning", "UNKNOWN")).upper()

                if is_valid:
                    valid_lane_frames += 1
                if warn_state == "WARNING":
                    warning_frames += 1

                offset_val = analysis.get("offset_normalized")
                offset_str = f"{offset_val:.4f}" if offset_val is not None else ""
                left_id = analysis.get("left_track_id")
                left_id_str = str(left_id) if left_id is not None else ""
                right_id = analysis.get("right_track_id")
                right_id_str = str(right_id) if right_id is not None else ""
                curve_str = str(analysis.get("curve", "UNKNOWN"))

                csv_writer.writerow([
                    frame_idx,
                    f"{time_s:.4f}",
                    f"{det_ms:.2f}",
                    f"{trk_ms:.2f}",
                    offset_str,
                    warn_state,
                    curve_str,
                    left_id_str,
                    right_id_str,
                    is_valid,
                ])

            except Exception as exc:
                failed_frames += 1
                if len(frame_errors) < 5:
                    frame_errors.append({"frame_index": frame_idx, "error": str(exc)})
                    logging.getLogger(__name__).exception("Lane processing failed at frame %s", frame_idx)
                # Fallback: write original frame with error overlay, record UNKNOWN
                fallback_analysis = {
                    "valid_lane": False,
                    "warning": "UNKNOWN",
                    "curve": "UNKNOWN",
                    "offset_normalized": None,
                }
                fallback_rendered = render_frame(frame, {"lanes": []}, fallback_analysis)
                writer.write(fallback_rendered)
                csv_writer.writerow([
                    frame_idx,
                    f"{time_s:.4f}",
                    0.0,
                    0.0,
                    "",
                    "UNKNOWN",
                    "UNKNOWN",
                    "",
                    "",
                    False,
                ])

            frames_processed += 1

            if progress_callback is not None:
                fraction = frames_processed / target_frames if target_frames else 0.5
                progress_callback(
                    min(fraction, 1.0),
                    f"Frame {frames_processed}/{target_frames or '?'} ({video_fps:.1f} FPS)",
                )

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    processing_seconds = max(time.time() - start_wall_clock, 1e-6)
    processing_fps = frames_processed / processing_seconds
    avg_inference_ms = (total_detector_ms / frames_processed) if frames_processed > 0 else 0.0

    # 4. Transcode to browser-compatible H.264
    transcoded = _transcode_to_h264(intermediate_video_path, output_path)
    if transcoded:
        if os.path.exists(intermediate_video_path):
            try:
                os.remove(intermediate_video_path)
            except OSError:
                pass
        final_video_path = output_path
    else:
        # If transcoding failed or ffmpeg not present, move intermediate to output_path
        if os.path.exists(intermediate_video_path):
            shutil.move(intermediate_video_path, output_path)
        final_video_path = output_path

    summary = {
        "frames_processed": frames_processed,
        "processing_seconds": round(processing_seconds, 3),
        "processing_fps": round(processing_fps, 2),
        "video_fps": round(video_fps, 2),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "valid_lane_frames": valid_lane_frames,
        "warning_frames": warning_frames,
        "failed_frames": failed_frames,
        "unknown_lane_frames": frames_processed - valid_lane_frames,
        "frame_errors": frame_errors,
        "execution_provider": getattr(detector, "execution_provider", "external"),
        "model_kind": getattr(detector, "model_kind", "external"),
        "road_crop_ratio": getattr(detector, "_selected_crop", road_crop_ratio),
        "source_start_seconds": start_frame / video_fps,
        "reference_y_ratio": reference_y_ratio,
    }

    return {
        "output_video": final_video_path,
        "csv_path": csv_path,
        "summary": summary,
    }

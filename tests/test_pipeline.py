"""Integration and unit tests for LaneGuard AI Web and Video Pipeline.

Tests visualizer, video processing pipeline, and Gradio app wiring
using mock modules conforming strictly to PROJECT_OVERVIEW.md API contract.
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from src.visualizer import render_frame
from src.video import process_video
from app import build_app, analyze_video_callback


class MockLaneDetector:
    """Mock detector adhering to PROJECT_OVERVIEW.md Section 5.2 contract."""

    def __init__(self, model_path: str = "mock.onnx", device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.detect_call_count = 0

    def detect(self, frame_bgr: np.ndarray) -> dict:
        self.detect_call_count += 1
        h, w = frame_bgr.shape[:2]
        return {
            "lanes": [
                {
                    "lane_index": 0,
                    "points": [[int(w * 0.28), int(h * 0.6)], [int(w * 0.25), int(h * 0.88)]],
                    "confidence": None,
                },
                {
                    "lane_index": 1,
                    "points": [[int(w * 0.72), int(h * 0.6)], [int(w * 0.75), int(h * 0.88)]],
                    "confidence": None,
                },
            ],
            "frame_width": w,
            "frame_height": h,
            "inference_ms": 14.5,
        }


class MockLaneTracker:
    """Mock tracker adhering to PROJECT_OVERVIEW.md Section 5.3 contract."""

    def __init__(self, alpha: float = 0.35, max_missed: int = 3) -> None:
        self.alpha = alpha
        self.max_missed = max_missed
        self.reset_called = False
        self.update_call_count = 0

    def reset(self) -> None:
        self.reset_called = True

    def update(self, detection: dict) -> dict:
        self.update_call_count += 1
        w = detection.get("frame_width", 640)
        h = detection.get("frame_height", 480)
        return {
            "lanes": [
                {
                    "track_id": 1,
                    "points": [[int(w * 0.28), int(h * 0.6)], [int(w * 0.25), int(h * 0.88)]],
                    "confidence": None,
                    "predicted": False,
                    "missed": 0,
                },
                {
                    "track_id": 2,
                    "points": [[int(w * 0.72), int(h * 0.6)], [int(w * 0.75), int(h * 0.88)]],
                    "confidence": None,
                    "predicted": False,
                    "missed": 0,
                },
            ],
            "frame_width": w,
            "frame_height": h,
            "tracking_ms": 0.9,
        }


def mock_analyze_lanes_safe(tracking: dict, camera_center_ratio: float = 0.5) -> dict:
    """Mock geometry analyzer returning SAFE status."""
    w = tracking.get("frame_width", 640)
    h = tracking.get("frame_height", 480)
    ref_y = int(round(0.85 * (h - 1)))
    return {
        "valid_lane": True,
        "reference_y": ref_y,
        "vehicle_center_x": camera_center_ratio * (w - 1),
        "lane_center_x": (w - 1) * 0.5,
        "left_track_id": 1,
        "right_track_id": 2,
        "offset_normalized": 0.012,
        "warning": "SAFE",
        "curve": "STRAIGHT",
        "lane_type_left": "SOLID",
        "lane_type_right": "DASHED",
    }


def mock_analyze_lanes_warning(tracking: dict, camera_center_ratio: float = 0.5) -> dict:
    """Mock geometry analyzer returning WARNING status."""
    w = tracking.get("frame_width", 640)
    h = tracking.get("frame_height", 480)
    ref_y = int(round(0.85 * (h - 1)))
    return {
        "valid_lane": True,
        "reference_y": ref_y,
        "vehicle_center_x": camera_center_ratio * (w - 1),
        "lane_center_x": (w - 1) * 0.4,
        "left_track_id": 1,
        "right_track_id": 2,
        "offset_normalized": 0.285,
        "warning": "WARNING",
        "curve": "CURVED",
        "lane_type_left": "SOLID",
        "lane_type_right": "SOLID",
    }


def create_synthetic_video(video_path: str, num_frames: int = 60, width: int = 640, height: int = 480) -> None:
    """Generate a test MP4 video file with road lines."""
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Draw simulated road surface
        cv2.line(frame, (int(width * 0.3), height), (int(width * 0.45), int(height * 0.55)), (200, 200, 200), 5)
        cv2.line(frame, (int(width * 0.7), height), (int(width * 0.55), int(height * 0.55)), (200, 200, 200), 5)
        writer.write(frame)
    writer.release()


class TestVisualizer(unittest.TestCase):
    """Unit tests for src/visualizer.py."""

    def test_render_frame_creates_new_image(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = render_frame(frame, tracking=None, analysis=None)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        # Verify it returns a new array and did not modify the input
        self.assertIsNot(out, frame)

    def test_render_frame_empty_lanes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = render_frame(frame, tracking={"lanes": []}, analysis={"valid_lane": False, "warning": "UNKNOWN"})
        self.assertEqual(out.shape, (480, 640, 3))

    def test_render_frame_predicted_lane(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking = {
            "lanes": [
                {
                    "track_id": 9,
                    "points": [[150, 300], [140, 420]],
                    "predicted": True,
                    "confidence": None,
                }
            ]
        }
        out = render_frame(frame, tracking=tracking, analysis=None)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_render_frame_warning_badges(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for warn in ["SAFE", "CAUTION", "WARNING", "UNKNOWN"]:
            analysis = {
                "valid_lane": warn != "UNKNOWN",
                "warning": warn,
                "lane_center_x": 320.0,
                "vehicle_center_x": 320.0,
                "reference_y": 400,
                "offset_normalized": 0.05 if warn != "UNKNOWN" else None,
            }
            out = render_frame(frame, tracking={"lanes": []}, analysis=analysis)
            self.assertEqual(out.shape, (480, 640, 3))

    def test_render_frame_invalid_dimension(self):
        with self.assertRaises(ValueError):
            render_frame(np.zeros((100, 100), dtype=np.uint8))


class TestVideoPipeline(unittest.TestCase):
    """Integration tests for src/video.py using mock modules."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="laneguard_test_")
        cls.synthetic_video = os.path.join(cls.test_dir, "synthetic_input.mp4")
        create_synthetic_video(cls.synthetic_video, num_frames=60, width=640, height=480)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_pipeline_execution_e2e(self):
        output_video = os.path.join(self.test_dir, "output_e2e.mp4")
        detector = MockLaneDetector()
        tracker = MockLaneTracker()

        result = process_video(
            input_path=self.synthetic_video,
            output_path=output_video,
            model_path="dummy_model.onnx",
            camera_center_ratio=0.5,
            detector=detector,
            tracker=tracker,
            analyzer=mock_analyze_lanes_safe,
        )

        # 1. Check returned structure
        self.assertIn("output_video", result)
        self.assertIn("csv_path", result)
        self.assertIn("summary", result)

        summary = result["summary"]
        self.assertEqual(summary["frames_processed"], 60)
        self.assertEqual(summary["valid_lane_frames"], 60)
        self.assertEqual(summary["warning_frames"], 0)
        self.assertEqual(summary["failed_frames"], 0)
        self.assertGreater(summary["processing_fps"], 0.0)
        self.assertEqual(summary["video_fps"], 30.0)
        self.assertAlmostEqual(summary["avg_inference_ms"], 14.5, places=1)

        # 2. Check output video
        self.assertTrue(os.path.exists(output_video))
        cap = cv2.VideoCapture(output_video)
        self.assertTrue(cap.isOpened())
        read_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(read_frames, 60)
        cap.release()

        # 3. Check CSV format & contents
        csv_path = result["csv_path"]
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            self.assertEqual(
                header,
                [
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
                ],
            )
            # 60 data rows + 1 header = 61 rows
            self.assertEqual(len(reader), 61)
            first_row = reader[1]
            self.assertEqual(first_row[0], "0")
            self.assertEqual(first_row[5], "SAFE")
            self.assertEqual(first_row[9], "True")

        # 4. Verify tracker reset called
        self.assertTrue(tracker.reset_called)

    def test_pipeline_max_frames_limit(self):
        output_video = os.path.join(self.test_dir, "output_max20.mp4")
        detector = MockLaneDetector()
        tracker = MockLaneTracker()

        result = process_video(
            input_path=self.synthetic_video,
            output_path=output_video,
            model_path="dummy_model.onnx",
            max_frames=20,
            detector=detector,
            tracker=tracker,
            analyzer=mock_analyze_lanes_safe,
        )

        summary = result["summary"]
        self.assertEqual(summary["frames_processed"], 20)

        with open(result["csv_path"], mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 21)

    def test_pipeline_warning_counter(self):
        output_video = os.path.join(self.test_dir, "output_warn.mp4")
        detector = MockLaneDetector()
        tracker = MockLaneTracker()

        result = process_video(
            input_path=self.synthetic_video,
            output_path=output_video,
            model_path="dummy_model.onnx",
            max_frames=15,
            detector=detector,
            tracker=tracker,
            analyzer=mock_analyze_lanes_warning,
        )

        summary = result["summary"]
        self.assertEqual(summary["warning_frames"], 15)

    def test_pipeline_missing_input_file(self):
        with self.assertRaises(FileNotFoundError):
            process_video(
                input_path="non_existent_file.mp4",
                output_path="out.mp4",
                model_path="dummy.onnx",
            )

    def test_processing_failures_are_reported_and_keep_video_length(self):
        detector = MockLaneDetector()
        detector.detect = lambda frame: (_ for _ in ()).throw(RuntimeError("inference unavailable"))
        result = process_video(
            input_path=self.synthetic_video,
            output_path=os.path.join(self.test_dir, "output_errors.mp4"),
            model_path="dummy.onnx",
            max_frames=7,
            detector=detector,
            tracker=MockLaneTracker(),
            analyzer=mock_analyze_lanes_safe,
        )
        summary = result['summary']
        self.assertEqual(summary['failed_frames'], 7)
        self.assertEqual(summary['unknown_lane_frames'], 7)
        self.assertEqual(summary['valid_lane_frames'], 0)
        self.assertEqual(len(summary['frame_errors']), 5)
        self.assertEqual(summary['frame_errors'][0], {'frame_index': 0, 'error': 'inference unavailable'})
        cap = cv2.VideoCapture(result['output_video'])
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        self.assertEqual(count, 7)

    def test_pipeline_missing_model_file(self):
        with self.assertRaises(FileNotFoundError):
            process_video(
                input_path=self.synthetic_video,
                output_path="out.mp4",
                model_path="non_existent_model.onnx",
            )


class TestGradioApp(unittest.TestCase):
    """Unit tests for app.py."""

    def test_build_app_instantiation(self):
        demo = build_app()
        self.assertIsNotNone(demo)

    def test_callback_forwards_framing_and_reports_partial_failure(self):
        summary = dict(frames_processed=10, processing_seconds=1, processing_fps=10,
                       video_fps=30, avg_inference_ms=50, valid_lane_frames=7,
                       warning_frames=0, failed_frames=2)
        result = dict(output_video='preview.mp4', csv_path='preview.csv', summary=summary)
        for crop, expected in ((-1.0, None), (0.35, 0.35)):
            with self.subTest(crop=crop), tempfile.TemporaryDirectory() as directory:
                video_path = os.path.join(directory, 'input.mp4')
                model_path = os.path.join(directory, 'model.onnx')
                for path in (video_path, model_path):
                    open(path, 'wb').close()
                with patch('app.process_video', return_value=result) as process:
                    video, _, _, message = analyze_video_callback(
                        video_path, 0.5, model_path, 10, False, road_crop_ratio=crop)
                self.assertEqual(process.call_args.kwargs['road_crop_ratio'], expected)
                self.assertEqual(video, 'preview.mp4')
                self.assertIn('2 frame', message)
                self.assertNotIn('hoàn tất thành công', message)

    def test_analyze_callback_missing_video(self):
        video, summary, csv_file, msg = analyze_video_callback(
            video_path=None,
            camera_center_ratio=0.5,
            model_path="dummy.onnx",
            max_frames_input=300,
            process_all_frames=False,
        )
        self.assertIsNone(video)
        self.assertIn("Vui lòng tải lên một video MP4", msg)

    def test_analyze_callback_missing_model(self):
        # Create a temp video
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            video, summary, csv_file, msg = analyze_video_callback(
                video_path=f.name,
                camera_center_ratio=0.5,
                model_path="non_existent_model_file.onnx",
                max_frames_input=300,
                process_all_frames=False,
            )
            self.assertIsNone(video)
            self.assertIn("Không tìm thấy model weights", msg)


if __name__ == "__main__":
    unittest.main()

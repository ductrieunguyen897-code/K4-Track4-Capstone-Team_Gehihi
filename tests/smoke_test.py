#!/usr/bin/env python
"""LaneGuard AI smoke tests (owner: Thân).

These are *smoke* checks for the integrated product, not unit tests of the AI
modules (those live in each member's ``tests/test_*.py``).

Two modes that work **after** the branches are merged on the integration
machine, plus one clearly-labelled mock mode for the parallel phase:

    python tests/smoke_test.py --check-files
        Check that the essential files exist (pipeline, UI, model weights).

    python tests/smoke_test.py --check-output outputs/demo_processed.mp4
        Open a produced video with OpenCV, count frames, and verify the matching
        CSV has data rows and the required keys.

    python tests/smoke_test.py --mock
        Run a tiny **synthetic** pipeline (mock detector/tracker/geometry) to
        check wiring only. Clearly labelled MOCK - it is NOT evidence that the
        UFLD model works and NOT a benchmark.

Exit code is 0 on success, 1 on failure. Missing files that belong to other
members are reported, but only count as failures when ``--strict`` is given
(during the parallel phase a missing peer module is expected, not a Than bug).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Files that must exist once all branches are integrated (PROJECT_OVERVIEW.md §4).
ESSENTIAL_FILES = [
    "app.py",
    "src/detector.py",
    "src/tracker.py",
    "src/geometry.py",
    "src/video.py",
    "src/visualizer.py",
    "models/ufld_tusimple.onnx",
]

# CSV columns required of the pipeline output (PROJECT_OVERVIEW.md §5.5).
REQUIRED_CSV_COLUMNS = [
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


def check_files(strict: bool = False) -> bool:
    """Return True if all essential files exist. Missing peers are warnings."""
    print("Checking essential files (post-integration):")
    missing = []
    for path in ESSENTIAL_FILES:
        exists = os.path.exists(path)
        print(f"  [{'ok' if exists else 'MISSING'}] {path}")
        if not exists:
            missing.append(path)

    if not missing:
        print("All essential files present.")
        return True

    print()
    print(f"{len(missing)} file(s) missing.")
    if strict:
        print("--strict given: treating missing files as failure.")
        return False
    print("(Parallel phase: modules owned by other members may not exist yet.")
    print(" Re-run with --strict on the integration machine.)")
    return True


def check_output(video_path: str) -> bool:
    """Validate a produced video + its CSV. Returns True on success."""
    ok = True

    if not os.path.exists(video_path):
        print(f"ERROR: output video not found: {video_path}")
        return False

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV (cv2) is required for --check-output.")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: OpenCV cannot open {video_path}")
        return False
    frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frames += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Output video: {video_path}")
    print(f"  frames={frames}, {w}x{h}, fps={fps:.2f}")
    if frames <= 0:
        print("  [FAIL] video has no readable frames")
        ok = False
    else:
        print("  [ok] video is readable by OpenCV")

    # CSV sits next to the video with the same stem.
    csv_path = os.path.splitext(video_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        print(f"  [FAIL] expected CSV not found: {csv_path}")
        return False

    print(f"CSV: {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = sum(1 for _ in reader)

    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in fieldnames]
    if missing_cols:
        print(f"  [FAIL] CSV missing columns: {missing_cols}")
        ok = False
    else:
        print("  [ok] CSV has all required columns")

    if rows <= 0:
        print("  [FAIL] CSV has no data rows")
        ok = False
    else:
        print(f"  [ok] CSV has {rows} data rows")

    return ok


def mock_pipeline() -> bool:
    """Run a synthetic in-memory pipeline to check wiring. NOT a benchmark.

    Uses mock objects that satisfy the API contracts in PROJECT_OVERVIEW.md
    sections 5.2-5.4. This proves nothing about the real model; it only shows
    the data shapes flow through detect -> track -> analyze -> render.
    """
    print("=" * 70)
    print("MOCK MODE - synthetic data. This is NOT a real pipeline and NOT a benchmark.")
    print("It only checks that contract-shaped dicts flow end to end.")
    print("=" * 70)

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required for --mock.")
        return False

    H, W = 480, 640
    ref_y = round(0.85 * (H - 1))

    # --- mock detector output (PROJECT_OVERVIEW.md §5.2) ---
    detection = {
        "lanes": [
            {"lane_index": 0, "points": [[180, 200], [175, 300], [168, ref_y]], "confidence": None},
            {"lane_index": 1, "points": [[470, 200], [478, 300], [486, ref_y]], "confidence": None},
        ],
        "frame_width": W,
        "frame_height": H,
        "inference_ms": 16.4,
    }

    # --- mock tracker output (§5.3): add track_id / predicted / missed ---
    tracking = {
        "lanes": [
            {**lane, "track_id": i + 1, "predicted": False, "missed": 0}
            for i, lane in enumerate(detection["lanes"])
        ],
        "frame_width": W,
        "frame_height": H,
        "tracking_ms": 0.5,
    }

    # --- mock geometry output (§5.4) ---
    xL, xR = 168.0, 486.0
    xV = 0.5 * (W - 1)
    xC = (xL + xR) / 2
    offset = (xV - xC) / (xR - xL)
    analysis = {
        "valid_lane": True,
        "reference_y": ref_y,
        "vehicle_center_x": xV,
        "lane_center_x": xC,
        "left_track_id": 1,
        "right_track_id": 2,
        "offset_normalized": round(offset, 3),
        "warning": "SAFE" if abs(offset) < 0.15 else "CAUTION",
        "curve": "UNKNOWN",
        "lane_type_left": "UNKNOWN",
        "lane_type_right": "UNKNOWN",
    }

    # --- validate the shapes we depend on ---
    checks = []
    checks.append(("detection has 2 lanes", len(detection["lanes"]) == 2))
    checks.append(("tracker assigned unique ids",
                   len({lane["track_id"] for lane in tracking["lanes"]}) == 2))
    checks.append(("all lane points are [x, y] pairs",
                   all(len(p) == 2 for lane in tracking["lanes"] for p in lane["points"])))
    checks.append(("analysis offset is a float", isinstance(analysis["offset_normalized"], float)))
    checks.append(("analysis warning is valid",
                   analysis["warning"] in {"SAFE", "CAUTION", "WARNING", "UNKNOWN"}))

    # --- mock render (draw a copy; real renderer is Triệu's) ---
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for lane in tracking["lanes"]:
        for x, y in lane["points"]:
            if 0 <= y < H and 0 <= x < W:
                canvas[y, x] = (0, 255, 255)
    checks.append(("render output keeps shape/dtype",
                   canvas.shape == (H, W, 3) and canvas.dtype == np.uint8))

    print()
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print()
    print(f"Mock analysis: offset={analysis['offset_normalized']}, warning={analysis['warning']}")
    print("(Mock run - not evidence the model works; run the real pipeline for that.)")

    return all(passed for _, passed in checks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LaneGuard AI smoke tests (owner: Thân).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check-files", action="store_true",
                        help="check that essential integrated files exist")
    parser.add_argument("--check-output", metavar="VIDEO",
                        help="validate a produced video and its CSV")
    parser.add_argument("--mock", action="store_true",
                        help="run a synthetic mock pipeline (NOT a benchmark)")
    parser.add_argument("--strict", action="store_true",
                        help="with --check-files: treat missing files as failure")
    args = parser.parse_args(argv)

    ran_any = False
    ok = True

    if args.check_files:
        ran_any = True
        ok = check_files(strict=args.strict) and ok
        print()

    if args.check_output:
        ran_any = True
        ok = check_output(args.check_output) and ok
        print()

    if args.mock:
        ran_any = True
        ok = mock_pipeline() and ok
        print()

    if not ran_any:
        parser.print_help()
        print("\nNothing to do: pass --check-files, --check-output VIDEO and/or --mock.")
        return 0

    print("=" * 70)
    print(f"SMOKE TEST RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Download the short lane-detection sample videos used by LaneGuard AI.

Owner: Than (data / benchmark / QA).

Source
------
Udacity "CarND-LaneLines-P1" test videos, released under the **MIT License**:
    https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_videos

The clips are small (a few MB each) and free to use and redistribute. They are
**not committed to git** (see ``data/README.md``): every machine that needs them
runs this script, or the equivalent ``curl`` commands listed in that README.

Usage
-----
    python scripts/download_samples.py                 # download all three clips
    python scripts/download_samples.py --out data       # choose output folder
    python scripts/download_samples.py --max-seconds 15 # trim each clip (needs OpenCV)

If OpenCV is unavailable, ``--max-seconds`` is ignored and a warning is printed;
the raw clips are still fine to use.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos"

# name -> (source file on GitHub, role in the test matrix)
SAMPLES = [
    ("sample_lines.mp4", "solidWhiteRight.mp4", "A - highway, solid white right lane"),
    ("sample_yellow.mp4", "solidYellowLeft.mp4", "B - solid yellow left lane"),
    ("sample_challenge.mp4", "challenge.mp4", "C - curve / shadows (hard case)"),
]


def _download(url: str, dest: str) -> None:
    """Stream ``url`` to ``dest`` with a small progress indicator."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            pct = 100.0 * done / total
            sys.stdout.write(f"\r    {os.path.basename(dest)}: {pct:5.1f}%")
            sys.stdout.flush()

    req = urllib.request.Request(url, headers={"User-Agent": "LaneGuard-AI/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        blocks = 0
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            blocks += 1
            _hook(blocks, 64 * 1024, total)
    sys.stdout.write("\r" + " " * 60 + "\r")


def _trim(path: str, max_seconds: float) -> bool:
    """Trim ``path`` in place to ``max_seconds``. Returns True if trimmed.

    Uses OpenCV only; if OpenCV is missing, returns False and leaves the file.
    """
    try:
        import cv2  # noqa: WPS433 (lazy import on purpose)
    except ImportError:
        print("    [warn] OpenCV not installed - skipping trim")
        return False

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        print(f"    [warn] cannot open {path} for trimming")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = int(max_seconds * fps)
    if keep >= n_total:
        cap.release()
        return False

    tmp = path + ".trim.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while written < keep:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    os.replace(tmp, path)
    print(f"    trimmed to {written} frames (~{written / fps:.1f}s)")
    return True


def _describe(path: str) -> str:
    """Return a short 'WxH @ fps, N frames' string, or '' if OpenCV is missing."""
    try:
        import cv2
    except ImportError:
        return ""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return ""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return f"{w}x{h} @ {fps:.2f}fps, {n} frames ({n / fps:.1f}s)" if fps else f"{w}x{h}, {n} frames"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the LaneGuard AI sample videos (Udacity, MIT License)."
    )
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="optionally trim each clip to at most this many seconds (needs OpenCV)",
    )
    args = parser.parse_args(argv)

    print("Source: Udacity CarND-LaneLines-P1 test_videos (MIT License)")
    print(f"Target: {os.path.abspath(args.out)}")
    print()

    failures = 0
    for local_name, remote_name, role in SAMPLES:
        dest = os.path.join(args.out, local_name)
        print(f"[{local_name}] {role}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"    already present ({os.path.getsize(dest) / 1e6:.2f} MB)")
        else:
            url = f"{BASE_URL}/{remote_name}"
            try:
                _download(url, dest)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"    [error] download failed: {exc}")
                failures += 1
                continue
            print(f"    downloaded {os.path.getsize(dest) / 1e6:.2f} MB")

        if args.max_seconds:
            _trim(dest, args.max_seconds)

        info = _describe(dest)
        if info:
            print(f"    -> {info}")
        print()

    if failures:
        print(f"Done with {failures} failure(s).")
        return 1
    print("Done. All sample clips are ready in", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

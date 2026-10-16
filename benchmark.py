#!/usr/bin/env python
"""Benchmark the LaneGuard AI video pipeline (owner: Thân).

This script **does not** run its own detector/tracker. It calls the single
project pipeline ``src.video.process_video`` (contract in
``PROJECT_OVERVIEW.md`` §5.5) and reports what that pipeline actually measured.
Everything printed as a metric comes from real data: the pipeline's returned
``summary`` and the CSV it writes.

Metrics reported
----------------
- ``processing_fps``  : frames / wall-clock seconds (throughput of *this* machine)
- ``video_fps``       : source clip FPS (NOT a performance number)
- ``avg_inference_ms``: mean per-frame detector latency
- ``frames_processed``, ``failed_frames``
- ``valid_lane_frames`` and ``valid_lane_ratio`` (share of frames with usable
  geometry — this is **not** an accuracy/F1)
- ``warning_frames``  : frames whose instantaneous warning is ``WARNING``
  (``CAUTION`` is counted separately, never folded in)
- ``safe`` / ``caution`` / ``unknown`` frame counts

Usage
-----
    python benchmark.py --input data/sample_lines.mp4 --model models/ufld_tusimple.onnx --max-frames 300
    python benchmark.py --input data/sample_lines.mp4 --model models/ufld_tusimple.onnx --output results/run_lines
    python benchmark.py --self-test          # exercise metric/report code on synthetic data

``--self-test`` runs the metric and report machinery on a **synthetic** CSV so
you can verify the benchmark code before the pipeline exists. It is clearly
labelled and never writes to ``results/``. It is NOT a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

# Columns the pipeline CSV is required to contain (PROJECT_OVERVIEW.md §5.5).
CSV_COLUMNS = [
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

_TRUTHY = {"1", "true", "yes", "y", "t"}


# --------------------------------------------------------------------------- #
# CSV parsing / metric computation (pure functions - unit-testable)
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "" or text in {"none", "nan", "null"}:
        return None
    if text in _TRUTHY:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def read_csv_metrics(csv_path: str) -> Dict[str, Any]:
    """Read a pipeline CSV and compute metrics from the rows themselves.

    Returns a dict of measured values. Missing columns are tolerated (they are
    reported as ``None`` / ``0``) so a partial pipeline still benchmarks.
    """
    metrics: Dict[str, Any] = {
        "rows": 0,
        "detector_ms_values": [],
        "tracker_ms_values": [],
        "valid_lane_frames": 0,
        "warning_frames": 0,
        "caution_frames": 0,
        "safe_frames": 0,
        "unknown_frames": 0,
        "offset_none_frames": 0,
        "missing_columns": [],
    }
    if not os.path.exists(csv_path):
        metrics["error"] = f"CSV not found: {csv_path}"
        return metrics

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        metrics["missing_columns"] = [c for c in CSV_COLUMNS if c not in fieldnames]
        for row in reader:
            metrics["rows"] += 1
            d_ms = _to_float(row.get("detector_ms"))
            if d_ms is not None:
                metrics["detector_ms_values"].append(d_ms)
            t_ms = _to_float(row.get("tracker_ms"))
            if t_ms is not None:
                metrics["tracker_ms_values"].append(t_ms)

            valid = _to_bool(row.get("valid_lane"))
            if valid is True:
                metrics["valid_lane_frames"] += 1

            warning = (row.get("warning") or "").strip().upper()
            if warning == "WARNING":
                metrics["warning_frames"] += 1
            elif warning == "CAUTION":
                metrics["caution_frames"] += 1
            elif warning == "SAFE":
                metrics["safe_frames"] += 1
            elif warning == "UNKNOWN":
                metrics["unknown_frames"] += 1

            if _to_float(row.get("offset_normalized")) is None:
                metrics["offset_none_frames"] += 1
    return metrics


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_metrics(summary: Dict[str, Any], csv_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the pipeline summary with independently-parsed CSV metrics.

    ``summary`` values are trusted for wall-clock numbers; the CSV is used for
    the distributions (detector latency mean, warning breakdown) and as a
    cross-check on frame counts.
    """
    frames = int(summary.get("frames_processed", csv_metrics.get("rows", 0)) or 0)
    valid = int(summary.get("valid_lane_frames", csv_metrics.get("valid_lane_frames", 0)) or 0)
    warning = int(summary.get("warning_frames", csv_metrics.get("warning_frames", 0)) or 0)

    avg_inference = summary.get("avg_inference_ms")
    if avg_inference is None:
        avg_inference = _mean(csv_metrics.get("detector_ms_values", []))

    return {
        "frames_processed": frames,
        "csv_rows": int(csv_metrics.get("rows", 0)),
        "processing_seconds": float(summary.get("processing_seconds", 0.0) or 0.0),
        "processing_fps": float(summary.get("processing_fps", 0.0) or 0.0),
        "video_fps": float(summary.get("video_fps", 0.0) or 0.0),
        "avg_inference_ms": float(avg_inference or 0.0),
        "mean_tracker_ms": _mean(csv_metrics.get("tracker_ms_values", [])),
        "valid_lane_frames": valid,
        "valid_lane_ratio": (valid / frames) if frames else 0.0,
        "warning_frames": warning,
        "warning_ratio": (warning / frames) if frames else 0.0,
        "caution_frames": int(csv_metrics.get("caution_frames", 0)),
        "safe_frames": int(csv_metrics.get("safe_frames", 0)),
        "unknown_frames": int(csv_metrics.get("unknown_frames", 0)),
        "offset_none_frames": int(csv_metrics.get("offset_none_frames", 0)),
        "failed_frames": int(summary.get("failed_frames", 0) or 0),
        "csv_missing_columns": csv_metrics.get("missing_columns", []),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _machine_info() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "(unavailable)",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count() or "?"),
    }
    # A friendlier CPU name on Windows/Linux if we can get one cheaply.
    try:
        import multiprocessing

        info["cpu_count"] = str(multiprocessing.cpu_count())
    except Exception:  # noqa: BLE001
        pass
    return info


def format_report(
    clip: str,
    model: str,
    max_frames: Optional[int],
    metrics: Dict[str, Any],
    machine: Dict[str, str],
    when: str,
) -> str:
    """Render a Markdown report. Contains ONLY measured values."""
    lines: List[str] = []
    lines.append("# LaneGuard AI — Benchmark report")
    lines.append("")
    lines.append(f"- **Run (UTC):** {when}")
    lines.append(f"- **Clip:** `{clip}`")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Max frames:** {max_frames if max_frames is not None else 'all'}")
    lines.append(f"- **Platform:** {machine['platform']}")
    lines.append(f"- **Processor:** {machine['processor']}")
    lines.append(f"- **CPU count:** {machine['cpu_count']}")
    lines.append(f"- **Python:** {machine['python']}")
    lines.append("")
    lines.append("> All numbers below were measured by running the pipeline. Nothing here is estimated.")
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Frames processed | {metrics['frames_processed']} |")
    lines.append(f"| Processing time (s) | {metrics['processing_seconds']:.2f} |")
    lines.append(f"| **Processing FPS** (frames / wall-clock) | **{metrics['processing_fps']:.2f}** |")
    lines.append(f"| Source video FPS (not a perf number) | {metrics['video_fps']:.2f} |")
    lines.append(f"| Avg detector latency (ms/frame) | {metrics['avg_inference_ms']:.2f} |")
    lines.append(f"| Avg tracker time (ms/frame) | {metrics['mean_tracker_ms']:.3f} |")
    lines.append(f"| Failed frames | {metrics['failed_frames']} |")
    lines.append("")
    lines.append("## Lane / warning coverage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Valid-lane frames | {metrics['valid_lane_frames']} "
        f"({metrics['valid_lane_ratio'] * 100:.1f}%) |"
    )
    lines.append(
        f"| WARNING frames | {metrics['warning_frames']} "
        f"({metrics['warning_ratio'] * 100:.1f}%) |"
    )
    lines.append(f"| CAUTION frames | {metrics['caution_frames']} |")
    lines.append(f"| SAFE frames | {metrics['safe_frames']} |")
    lines.append(f"| UNKNOWN frames | {metrics['unknown_frames']} |")
    lines.append(f"| Frames with no offset | {metrics['offset_none_frames']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- *Valid-lane ratio* is the share of frames where geometry had enough evidence. "
                 "It is **not** accuracy, precision or F1 — there is no ground truth here.")
    lines.append("- *WARNING frames* counts only `warning == WARNING`; `CAUTION` is reported separately.")
    lines.append("- *Processing FPS* is throughput on the machine above, not the source clip's FPS, "
                 "and excludes Gradio upload time.")
    if metrics["csv_missing_columns"]:
        lines.append(f"- ⚠ CSV is missing expected columns: {metrics['csv_missing_columns']}")
    lines.append("")
    return "\n".join(lines)


def write_summary_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def maybe_plot(csv_path: str, out_prefix: str) -> Optional[str]:
    """Optional chart of detector latency per frame + warning distribution."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [chart] skipped (matplotlib unavailable: {exc})")
        return None

    frames: List[int] = []
    det: List[float] = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = _to_float(row.get("detector_ms"))
                if v is not None:
                    frames.append(len(frames))
                    det.append(v)
    except OSError:
        return None

    if not det:
        print("  [chart] skipped (no detector_ms values)")
        return None

    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    ax.plot(frames, det, linewidth=0.8, color="#2e548a")
    ax.set_xlabel("frame index")
    ax.set_ylabel("detector_ms")
    ax.set_title("Detector latency per frame (measured)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = f"{out_prefix}_latency.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [chart] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Self-test (synthetic - NOT a benchmark)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    """Exercise metric + report code on a synthetic CSV. Not a benchmark."""
    import tempfile

    print("=" * 70)
    print("SELF-TEST - synthetic data. This is NOT a benchmark.")
    print("It only checks that the metric and report code run correctly.")
    print("=" * 70)

    rows = []
    for i in range(100):
        if i % 10 == 0:
            warning, valid, offset = "UNKNOWN", "False", ""
        elif i % 7 == 0:
            warning, valid, offset = "WARNING", "True", "0.31"
        elif i % 5 == 0:
            warning, valid, offset = "CAUTION", "True", "0.18"
        else:
            warning, valid, offset = "SAFE", "True", "0.03"
        rows.append(
            {
                "frame_index": i,
                "time_s": f"{i / 25.0:.3f}",
                "detector_ms": f"{18 + (i % 5)}.0",
                "tracker_ms": "0.8",
                "offset_normalized": offset,
                "warning": warning,
                "curve": "UNKNOWN",
                "left_track_id": 1 if valid == "True" else "",
                "right_track_id": 2 if valid == "True" else "",
                "valid_lane": valid,
            }
        )

    tmpdir = tempfile.mkdtemp(prefix="laneguard_selftest_")
    csv_path = os.path.join(tmpdir, "synthetic.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    csv_metrics = read_csv_metrics(csv_path)
    fake_summary = {
        "frames_processed": 100,
        "processing_seconds": 4.0,
        "processing_fps": 25.0,
        "video_fps": 25.0,
        "avg_inference_ms": 20.0,
        "valid_lane_frames": csv_metrics["valid_lane_frames"],
        "warning_frames": csv_metrics["warning_frames"],
        "failed_frames": 0,
    }
    metrics = build_metrics(fake_summary, csv_metrics)

    print("\nParsed from synthetic CSV:")
    print(f"  rows                : {csv_metrics['rows']}")
    print(f"  valid_lane_frames   : {csv_metrics['valid_lane_frames']}")
    print(f"  warning_frames      : {csv_metrics['warning_frames']}")
    print(f"  caution_frames      : {csv_metrics['caution_frames']}")
    print(f"  unknown_frames      : {csv_metrics['unknown_frames']}")
    print(f"  offset_none_frames  : {csv_metrics['offset_none_frames']}")

    ok = (
        csv_metrics["rows"] == 100
        and csv_metrics["missing_columns"] == []
        and metrics["valid_lane_ratio"] > 0
    )
    print(f"\nSelf-test result: {'PASS' if ok else 'FAIL'}")
    print("(Synthetic run - do not report these numbers as a benchmark.)")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_benchmark(args: argparse.Namespace) -> int:
    # Lazy import: keeps `--help` and `--self-test` working before integration.
    try:
        from src.video import process_video  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print("ERROR: cannot import src.video.process_video.", file=sys.stderr)
        print(f"       {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "       The pipeline module is owned by Triệu and must exist for a real benchmark.\n"
            "       Run `python benchmark.py --self-test` to check the benchmark code alone.",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(args.input):
        print(f"ERROR: input video not found: {args.input}", file=sys.stderr)
        return 2
    if not os.path.exists(args.model):
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        print("       See models/README.md for how to download the UFLD ONNX weights.", file=sys.stderr)
        return 2

    os.makedirs("outputs", exist_ok=True)
    out_prefix = args.output or os.path.join(
        "results", "benchmark_" + os.path.splitext(os.path.basename(args.input))[0]
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    processed_video = os.path.join("outputs", os.path.basename(out_prefix) + "_processed.mp4")

    print(f"Running pipeline on {args.input} ...")
    t0 = time.perf_counter()
    result = process_video(
        input_path=args.input,
        output_path=processed_video,
        model_path=args.model,
        camera_center_ratio=0.5,
        max_frames=args.max_frames,
    )
    wall = time.perf_counter() - t0

    if not isinstance(result, dict) or "summary" not in result:
        print("ERROR: process_video did not return the expected dict with 'summary'.", file=sys.stderr)
        return 3

    summary = result.get("summary", {})
    csv_path = result.get("csv_path", "")
    csv_metrics = read_csv_metrics(csv_path) if csv_path else {"rows": 0, "missing_columns": CSV_COLUMNS}
    metrics = build_metrics(summary, csv_metrics)

    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    machine = _machine_info()

    # ---- terminal output ----
    print("\n" + "-" * 60)
    print("RESULTS (measured)")
    print("-" * 60)
    print(f"  output video        : {result.get('output_video', processed_video)}")
    print(f"  csv                 : {csv_path or '(none)'}")
    print(f"  frames processed    : {metrics['frames_processed']}")
    print(f"  processing seconds  : {metrics['processing_seconds']:.2f}  (harness wall-clock {wall:.2f}s)")
    print(f"  processing FPS      : {metrics['processing_fps']:.2f}")
    print(f"  source video FPS    : {metrics['video_fps']:.2f}")
    print(f"  avg detector ms     : {metrics['avg_inference_ms']:.2f}")
    print(f"  avg tracker ms      : {metrics['mean_tracker_ms']:.3f}")
    print(f"  valid-lane frames   : {metrics['valid_lane_frames']} ({metrics['valid_lane_ratio'] * 100:.1f}%)")
    print(f"  WARNING frames      : {metrics['warning_frames']} ({metrics['warning_ratio'] * 100:.1f}%)")
    print(f"  CAUTION / SAFE / UNKNOWN : {metrics['caution_frames']} / {metrics['safe_frames']} / {metrics['unknown_frames']}")
    print(f"  failed frames       : {metrics['failed_frames']}")
    print(f"  machine             : {machine['processor']} | {machine['cpu_count']} CPUs | py {machine['python']}")
    print("-" * 60)

    # ---- artefacts ----
    row = {
        "run_utc": when,
        "clip": args.input,
        "model": args.model,
        "max_frames": args.max_frames if args.max_frames is not None else "",
        **{k: v for k, v in metrics.items() if k != "csv_missing_columns"},
        "processor": machine["processor"],
        "cpu_count": machine["cpu_count"],
        "python": machine["python"],
    }
    summary_csv = out_prefix + "_summary.csv"
    write_summary_csv(summary_csv, row)
    print(f"wrote {summary_csv}")

    report_md = out_prefix + "_report.md"
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write(format_report(args.input, args.model, args.max_frames, metrics, machine, when))
    print(f"wrote {report_md}")

    maybe_plot(csv_path, out_prefix)

    # Also drop a JSON blob next to the outputs for programmatic use.
    with open(out_prefix + "_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "machine": machine, "run_utc": when}, fh, indent=2)

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the LaneGuard AI pipeline (calls src.video.process_video).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", help="input video path (e.g. data/sample_lines.mp4)")
    parser.add_argument("--model", help="UFLD ONNX model path (e.g. models/ufld_tusimple.onnx)")
    parser.add_argument("--max-frames", type=int, default=300, help="cap frames processed")
    parser.add_argument(
        "--output",
        default=None,
        help="output prefix for report/chart (default: results/benchmark_<clip>)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run metric/report code on synthetic data (NOT a benchmark)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.input or not args.model:
        parser.error("--input and --model are required (or use --self-test)")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())

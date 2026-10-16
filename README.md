# LaneGuard AI

**LaneGuard AI** is a local web app that analyzes a recorded dashcam video and
shows an illustrative estimate of the lane the vehicle is in. It runs a
pretrained **UFLD ONNX** lane-detection model (inference only), tracks the lane
markings across frames with OpenCV, estimates the vehicle's position relative to
the lane center in **image coordinates**, and renders an overlay video plus a CSV
of per-frame metrics.

> **Demo only.** LaneGuard AI is not a car safety system and must not be used to
> make driving decisions. It does not report distance in metres, road curvature
> in 1/m, or validated ADAS-grade lane-departure warnings — the camera is not
> calibrated and the outputs are not validated against ground truth.

This README is the entry point for the merged project. The API contract that
the detector, tracker and video modules expose is documented in their Python
docstrings and tests. The original hackathon brief page is
[`index.html`](index.html).

---

## 1. What it does (features)

Planned / required for the 2-hour MVP (P0):

- Accept an `.mp4` dashcam clip and produce a browser-playable output video.
- Run a **pretrained UFLD ONNX** model per frame (no training / fine-tuning).
- Track lane polylines across frames with stable IDs and EMA smoothing.
- Find the two lane boundaries bracketing the camera axis, draw the lane center.
- Compute a **normalized** offset (not metres) and a demo warning
  `SAFE` / `CAUTION` / `WARNING` / `UNKNOWN`.
- Report measured FPS, inference latency, and the share of frames with valid
  lanes; export a per-frame CSV.
- A Gradio UI: upload → Analyze → video + metrics + CSV.

> Status note: the modules described here are integrated on `integration-final`
> and verified end-to-end with the real pretrained ONNX model and real video
> (see §7). Benchmark numbers in this README are measured, not estimated.

---

## 2. Architecture

```text
Video (mp4)
   │
   ▼
UFLD ONNX detector ──► Tracker ──► Geometry ──► Overlay / UI
 (src/detector.py)   (src/tracker.py) (src/geometry.py) (src/visualizer.py)
                                                 │
                                                 ▼
                                        Gradio app (app.py)
                                        CSV + summary metrics
```

Text flow: `Video → UFLD ONNX → Tracker → Geometry → Overlay/UI`.

- **detector** (`src/detector.py`) — `LaneDetector.detect(frame_bgr) -> dict`
- **tracker** (`src/tracker.py`) — `LaneTracker.update(detection) -> dict`
- **geometry** (`src/geometry.py`) — `analyze_lanes(tracking, camera_center_ratio) -> dict`
- **visualizer / video / app** (`src/visualizer.py`, `src/video.py`, `app.py`)
  — `render_frame(...)`, `process_video(...)`, `build_app()`

Full signatures and output schemas: the module docstrings in `src/`.

---

## 3. Project structure

```text
K4-Track4-Capstone-Team_Gehihi/
├── app.py                     # Gradio web app              (Triệu)
├── requirements.txt           # Python dependencies         (Triệu)
├── benchmark.py               # performance benchmark CLI   (Thân)
├── README.md                  # this file                   (Thân)
├── index.html                 # hackathon brief page
├── src/
│   ├── detector.py            # UFLD ONNX inference         (Quang)
│   ├── tracker.py             # temporal lane association   (Việt Anh)
│   ├── geometry.py            # offset + warning rules      (Diện)
│   ├── video.py               # video pipeline              (Triệu)
│   └── visualizer.py          # overlay rendering           (Triệu)
├── models/
│   ├── README.md              # weights source & setup       (Quang)
│   └── ufld_tusimple.onnx     # model (NOT committed)
├── data/
│   ├── README.md              # sample clip sources/license  (Thân)
│   └── sample_*.mp4           # clips (NOT committed)
├── docs/
│   └── presentation.md        # pitch slides + demo script   (Thân)
├── scripts/
│   └── download_samples.py    # fetch sample videos          (Thân)
├── tests/
│   ├── test_detector.py       # (Quang)
│   ├── test_tracker.py        # (Việt Anh)
│   ├── test_geometry.py       # (Diện)
│   ├── test_pipeline.py       # (Triệu)
│   └── smoke_test.py          # integration smoke tests      (Thân)
├── outputs/                   # runtime outputs (NOT committed)
└── results/                   # small measured reports (may commit)
```

---

## 4. Quick start

Requires **Python 3.10+** (tested on 3.13). Windows, macOS or Linux.

```bash
# 1. clone and enter
git clone <REPO_URL>
cd K4-Track4-Capstone-Team_Gehihi

# 2. virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. download the pretrained UFLD ONNX weights into models/ufld_tusimple.onnx
#    (source + exact steps: models/README.md)

# 5. (optional) download the sample test clips into data/
python scripts/download_samples.py

# 6. run the app
python app.py
# then open the local URL it prints (default http://127.0.0.1:7860)
```

> `requirements.txt` and `app.py` are owned by Triệu; if a dependency is missing,
> tell Triệu rather than editing those files yourself.

### Camera framing and unsupported detections

The detector now checks its proposals against nearby white/yellow marking
contrast and applies a bounded correction to the visible curve. Unsupported
proposals are removed; missing boundaries produce `UNKNOWN`. This check can
also miss faint markings, and is not a calibrated confidence score.
UFLDv2 proposals that fold back through column anchors or jump abruptly between
nearby image rows are rejected before tracking to prevent zigzag overlays.

**Vùng ảnh nhận diện đường** in the advanced settings selects camera framing.
Automatic mode estimates the road horizon on the first frame and, for cameras
showing a large sky area, selects a crop of the upper 20%, 35% or 45% of the
image. The selection stays fixed during the video. Use an explicit crop if the
first frame is obscured or the automatic selection is unsuitable. Coordinates
are mapped back onto the full original frame. `None` enables automatic framing
in `LaneDetector(..., road_crop_ratio=None)` and `process_video(...)`; `0.0`
selects the full image. The tracker restores newly visible rows after an
occlusion instead of permanently shortening a boundary.

The UI reuses the last loaded model in a sequential analysis queue, resetting
camera and tracking state for each video. This avoids loading large ONNX
weights on every analysis. Replacing the weight file or changing device/crop
loads a new session. Use **Bắt đầu từ giây** to inspect later sections without
processing the entire prefix; CSV frame indices and timestamps retain the
source-video position. Camera framing is calibrated from the original first
frame even when analysis starts later, so seeking does not change calibration.

Position measurement defaults to the 90% image row near the vehicle, adjustable
under **Vị trí đo gần xe**. `analyze_lanes()` keeps its 85% API default for
existing callers and accepts `reference_y_ratio`. No boundary is extrapolated
to reach the measurement row. Association now maximizes one-to-one matches
before minimizing total distance, reducing ID loss when close candidates compete.

The optional UFLDv2 CULane model and GPU setup are described in
[`models/README.md`](models/README.md). CPU remains supported. The UI reports
the actual execution provider; explicit GPU requests fail clearly if unavailable.

The summary includes `unknown_lane_frames`, `failed_frames` and up to five
`frame_errors`; the UI flags processing errors instead of reporting unqualified
success. A frame with two accepted boundaries is **not** an accuracy measurement.
The summary also records model family, execution provider, camera crop,
measurement row and source start time. See
[`results/lane_v2_validation.md`](results/lane_v2_validation.md) for current validation.

### Benchmark command

```bash
python benchmark.py --input data/sample_lines.mp4 --model models/ufld_tusimple.onnx --max-frames 300
```

Writes `results/benchmark_<clip>_report.md`, `_summary.csv`, `_metrics.json` and
(if matplotlib is present) `_latency.png`. Before the pipeline exists you can
still check the benchmark code itself:

```bash
python benchmark.py --self-test     # synthetic data - NOT a benchmark
```

### Smoke tests

```bash
python tests/smoke_test.py --check-files               # post-integration file check
python tests/smoke_test.py --check-output outputs/demo_processed.mp4
python tests/smoke_test.py --mock                      # synthetic wiring check only
```

---

## 5. Getting the pretrained model

LaneGuard AI uses a **pretrained, inference-only** UFLD ONNX model. No training
or fine-tuning is performed.

- Reference inference code: https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference
- ONNX model collection: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection
- Original UFLD: https://github.com/cfzd/Ultra-Fast-Lane-Detection

Place the downloaded file at **`models/ufld_tusimple.onnx`**. The runtime filename
is internal — verify the real input/output shapes and the license before use.
Exact download steps and checksums (when available) live in **`models/README.md`**
(owned by Quang). Model weights are **not committed** to git.

---

## 6. Sample data

Three short, free test clips (Udacity `CarND-LaneLines-P1`, **MIT License**):
highway, yellow-lane and a curve/shadow "challenge" clip. Download with
`python scripts/download_samples.py`. Sources, sizes and the test matrix are in
**[`data/README.md`](data/README.md)**. The `.mp4` files are not committed.

---

## 7. Benchmarks

Measured results go in `results/`. **No numbers are published here until they
are measured on real hardware** — see `results/benchmark_<clip>_report.md` after
a run.

Historical measurements before the image-support filter and camera-framing
changes below; these numbers do not describe the current detector. Current
verification results are in [`results/lane_fix_validation.md`](results/lane_fix_validation.md).

Measured on the real pretrained `models/ufld_tusimple.onnx` with the full
`detect → track → analyze → render` pipeline (`benchmark.py`, ONNX Runtime CPU):

| Clip | Frames | Processing FPS | Avg inference (ms) | Valid-lane ratio | WARNING frames | Machine |
|---|---|---|---|---|---|---|
| `data/sample_lines.mp4` | 221 | 12.82 | 66.71 | 221 / 221 (100.0%) | 0 | AMD64 12-CPU, Python 3.13.13 |
| `data/sample_challenge.mp4` | 251 | 16.01 | 49.56 | 251 / 251 (100.0%) | 0 | AMD64 12-CPU, Python 3.13.13 |

Both clips are highway clips, which is the domain the pretrained UFLD model was
trained on, so it finds two boundaries on essentially every frame — a 100%
valid-lane ratio here is **not** evidence of accuracy on harder roads (see §8).
Per-run artefacts: `results/benchmark_<clip>_report.md`, `_metrics.json`,
`_latency.png` (and `_summary.csv`).

Interpretation rules (kept honest):

- *Processing FPS* is throughput on the test machine — **not** the source clip FPS.
- *Valid-lane ratio* is the share of frames with usable geometry — **not** accuracy / F1.
- *WARNING frames* counts only `warning == WARNING`; `CAUTION` is separate.

---

## 8. Limitations

- Camera is **not calibrated**: offsets are normalized image coordinates, not metres.
- Warnings are **demo rules**, not validated lane-departure warnings.
- Model is a pretrained highway-lane model; it may fail on **Vietnamese urban
  roads**, heavy rain, night, glare, worn markings, or complex junctions.
- Valid-lane ratio is not accuracy — there is no ground-truth labelling.
- Output video codec depends on the environment (OpenCV `mp4v` or FFmpeg H.264);
  browser playback should be verified on the demo machine.

---

## 9. Team & branches

| Member | Branch | Owns |
|---|---|---|
| Quang | `quang` | `src/detector.py`, `models/README.md`, `tests/test_detector.py` |
| Việt Anh | `viet-anh` | `src/tracker.py`, `tests/test_tracker.py` |
| Diện | `dien` | `src/geometry.py`, `tests/test_geometry.py` |
| Triệu | `trieu` | `app.py`, `src/video.py`, `src/visualizer.py`, `requirements.txt`, `.gitignore`, `tests/test_pipeline.py` |
| Thân | `than` | `benchmark.py`, `README.md`, `data/README.md`, `docs/presentation.md`, `scripts/download_samples.py`, `tests/smoke_test.py` |

Each member pushes their own branch. Integration happens on the
`integration-final` branch, which is then fast-forwarded into `main`.

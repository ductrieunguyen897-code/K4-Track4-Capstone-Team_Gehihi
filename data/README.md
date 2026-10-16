# Sample data — LaneGuard AI

Owner: **Thân** (data / benchmark / QA). Do not commit large video files.

This folder holds the short lane clips the team uses to test lane detection,
tracking and geometry. **The `.mp4` files are not committed to git** — every
machine downloads them locally with `scripts/download_samples.py` (see below).

---

## 1. Source & license

All three clips come from the Udacity self-driving car course repository
**`CarND-LaneLines-P1`**, folder `test_videos`:

- Repository: https://github.com/udacity/CarND-LaneLines-P1
- License: **MIT License** (https://github.com/udacity/CarND-LaneLines-P1/blob/master/LICENSE)

The MIT License permits use, copy, modification and redistribution with
attribution. We keep this README as the attribution record. **Re-check the
license before any public redistribution.**

> These clips are **US highway footage**. They are a convenient, small, free
> test set for a 2-hour MVP — they are **not** evidence that the demo works on
> Vietnamese urban roads, in rain, or at night. Treat any Vietnamese-road clip
> as an extra stress test only, and only with a clear right to use it.

---

## 2. The three clips

| Local name | Upstream file | Role in the test matrix | Resolution | FPS | Frames | Duration | Size |
|---|---|---|---|---|---|---|---|
| `sample_lines.mp4` | `solidWhiteRight.mp4` | **A** — highway, solid white right lane (easy) | 960×540 | 25.0 | 221 | 8.8 s | 2.64 MB |
| `sample_yellow.mp4` | `solidYellowLeft.mp4` | **B** — solid yellow left lane (color / contrast) | 960×540 | 25.0 | 681 | 27.2 s | 8.01 MB |
| `sample_challenge.mp4` | `challenge.mp4` | **C** — curve + shadows (hard case, expect more `UNKNOWN`) | 1280×720 | 25.0 | 251 | 10.0 s | 15.09 MB |

Metadata measured locally with OpenCV on download (2026-10-16). Re-measure if
upstream files change.

**How the clips are used**

- **A** — check that ONNX detection finds lanes, the center marker is drawn and
  the video renders.
- **B** — check colour/contrast normalisation and tracking stability.
- **C** — check the `UNKNOWN` guard and warnings under shadows / curvature;
  expect a lower *valid-lane ratio* here, which is the honest result, not a bug.

---

## 3. Downloading the clips

```bash
# from the repository root
python scripts/download_samples.py
```

Options:

```bash
python scripts/download_samples.py --out data            # choose folder
python scripts/download_samples.py --max-seconds 15      # trim each clip (needs OpenCV)
```

Equivalent manual commands (if you prefer `curl`):

```bash
cd data
curl -L -o sample_lines.mp4     https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4
curl -L -o sample_yellow.mp4    https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidYellowLeft.mp4
curl -L -o sample_challenge.mp4 https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/challenge.mp4
```

---

## 4. What is NOT committed

Per the project rules, the following are **local artefacts** and must not be
pushed to git:

- `data/*.mp4`, `data/*.avi` — sample videos (download with the script)
- `models/*.onnx` — model weights (see `models/README.md` once Quang adds it)
- `outputs/*` — processed videos and CSV produced at runtime
- `results/*.mp4`, `results/*.csv` — large benchmark outputs

Small, text-only benchmark summaries (`results/benchmark_report.md`,
`results/benchmark_summary.csv`) **may** be committed once they contain real,
measured numbers — never fabricated ones.

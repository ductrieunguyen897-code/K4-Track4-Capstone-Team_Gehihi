# UFLDv2 and tracking improvements - verification (2026-10-16)

## Implemented behavior

- Support the official UFLDv2 CULane ResNet18 ONNX output heads, training crop and hybrid row/column anchors. Retain UFLDv1 compatibility.
- Reject folded outer-column predictions and abrupt per-row jumps before tracking, preventing zigzag overlays and phantom boundaries caused by reordering noisy anchors.
- Provide a reproducible inference-only exporter with strict checkpoint loading, pinned checkpoint SHA-256, atomic output replacement and ONNX graph validation.
- Support optional NVIDIA CUDA with explicit provider reporting and CPU fallback in auto mode; keep FP32 as the default.
- Replace greedy association with maximum-cardinality, minimum-cost assignment.
- Scale paint-support requirements to the visible road ROI. Keep existing model samples across bounded dash gaps without extrapolating the correction polynomial.
- Reduce preprocessing copies and batch the local grid and marking searches.
- Measure at the 90% image row near the vehicle by default in the video pipeline; keep the geometry API 85% default for compatibility. Both are configurable.
- Add source-time seeking with source timestamps preserved and camera framing calibrated from the original first frame.
- Cache one model session in the sequential UI queue. Invalidate the cache when device, crop or weight file changes.
- Mark retained history with dashed orange lines. Distinguish processing failures and missing lane results in the UI. Use H.264 veryfast encoding.

## Checks

- Full test suite: **96 passed, 98 subtests passed**.
- After the encoding change: pipeline tests **18 passed, 7 subtests passed**.
- After final UI changes: Gradio callback tests **5 passed, 2 subtests passed**.
- `git diff --check` passed.
- The reproducible exporter loaded the official checkpoint strictly and passed the ONNX checker. Its four outputs matched the independently exported reference exactly (max absolute difference 0.0) on a zero tensor and a real road frame, on CPU.
- Seven H.264 outputs were fully decoded and all expected frame counts matched.
- Two real Gradio upload/analyze API requests processed 30 frames each with zero failures, used CUDA and preserved the requested 1s/2s CSV source timestamps. Observed request wall times were 22.618s and 4.526s; the second request reused the model session.

## Measured frame availability

The baseline is the previously published lane-fix report at `b5ab1a5`. These are complete-pipeline comparisons: model, evidence filter, association and measurement row have changed. They are not isolated model benchmarks and are **not accuracy/F1 measurements**. No independent ground-truth lane annotations are available.

| Input | Frames | Baseline with two boundaries | New with two boundaries | UNKNOWN | Processing failures | Pipeline FPS |
|---|---:|---:|---:|---:|---:|---:|
| hanoi_segments | 540 | 191 | 346 | 194 | 0 | 5.22 |
| sample_lines | 221 | 205 | 221 | 0 | 0 | 8.59 |
| sample_challenge | 251 | 231 | 250 | 1 | 0 | 7.33 |
| sample_yellow | 681 | 668 | 677 | 4 | 0 | 8.23 |
| hanoi_first10s | 300 | 127 | 298 | 2 | 0 | 4.6 |
| hanoi_180s | 60 | not measured | 60 | 0 | 0 | 5.43 |
| hanoi_300s | 60 | not measured | 9 | 51 | 0 | 6.74 |

Total processed across the seven runs: **2113 frames**, zero processing failures. These runs include overlapping source intervals. Pipeline FPS excludes model initialization and the final transcode, and is specific to this machine and run; it is below source-video FPS. Source playback timestamps are retained, regardless of processing throughput.

All runs use UFLDv2 FP32, CUDAExecutionProvider, automatic camera crop and reference_y_ratio=0.9. The three sample clips use crop 0.0; the Hanoi clips use crop 0.45. Hardware: GTX 1650, 4 GB VRAM; ONNX Runtime GPU 1.21.1 with PyTorch 2.7.1+cu126 CUDA DLLs.

The 540-frame Hanoi input is the same re-encoded montage used by the baseline: 45 frames at source seconds 0, 5, 10, 20, 40, 60, 120, 180, 300, 420, 540 and 700. The other Hanoi runs read the original 1920x1080, 30 FPS file: first 300 frames, frames 5400-5459 and frames 9000-9059. The original has 21,799 frames; the full original video was not processed.

## Precision and model provenance

An optional FP16 model was tested on 12 original-video timestamps. Maximum logit difference was below 0.013 and existence-class predictions matched in all probes. On this GTX 1650, median ONNX execution was 35.69 ms for FP32 and 103.75 ms for FP16 in the component probe, so FP16 is not automatically selected. These are model-only timings and not end-to-end throughput.

Official checkpoint SHA-256: `956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967`. The exporter-produced ONNX checked for exact inference equivalence had SHA-256 `6f5026fcae0d9cb9215da609dbca4c9264671d1c601a0d7084b40412f22a8ad4`; the original author-code export used for the video runs had SHA-256 `c76d3b0dd8f104f2c8da418a86186a455443d8249bc009945e4699256d7faaac`. Serialized bytes differ while the checked outputs match.

Sources: [author model/configuration](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2), [official decoder](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/demo.py), [ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Remaining limits

Urban junctions, vehicle occlusion, worn markings and bright non-marking details can still produce missing or incorrect boundaries. Some accepted pairs can still represent the wrong lane, especially on curves or near a lane change. Availability of two boundaries and model existence probabilities do not prove correctness. Image-coordinate WARNING/SAFE rules remain uncalibrated; warning-count changes are not evidence of safer decisions. A valid measurement is never extended beyond the model polyline domain. Automatic framing relies on the original first frame; manual crop remains available.

## Local review files

- `outputs/lane_v2_release/hanoi_first10s.mp4`: continuous original-source preview.
- `outputs/lane_v2_release/hanoi_segments.mp4`: the full 12-section montage.
- `outputs/lane_v2_release/hanoi_180s.mp4` and `hanoi_300s.mp4`: source seek checks.
- `outputs/lane_v2_release/comparison.jpg`: baseline/new overlay on identical montage frames, 20 frames after each sampled section starts.
- `outputs/lane_v2_release/samples_review.jpg`: first/middle/last snapshots of all three complete sample runs.
- `outputs/lane_v2_release/results.json`, `decode_check.json`, `ui_check.json` and CSVs: measured outputs and checks.
- `outputs/lane_v2/profile.json` and `export_check.json`: component probe and exporter equivalence.

Weights, downloaded source and runtime wheels, videos and previews remain local under ignored paths.

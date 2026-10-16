# LaneGuard AI — Benchmark report

- **Run (UTC):** 2026-10-16 09:39:52
- **Clip:** `data/sample_challenge.mp4`
- **Model:** `models/ufld_tusimple.onnx`
- **Max frames:** 251
- **Platform:** Windows-11-10.0.26200-SP0
- **Processor:** AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **CPU count:** 12
- **Python:** 3.13.13

> All numbers below were measured by running the pipeline. Nothing here is estimated.

## Throughput

| Metric | Value |
|---|---|
| Frames processed | 251 |
| Processing time (s) | 15.68 |
| **Processing FPS** (frames / wall-clock) | **16.01** |
| Source video FPS (not a perf number) | 25.00 |
| Avg detector latency (ms/frame) | 49.56 |
| Avg tracker time (ms/frame) | 0.530 |
| Failed frames | 0 |

## Lane / warning coverage

| Metric | Value |
|---|---|
| Valid-lane frames | 251 (100.0%) |
| WARNING frames | 0 (0.0%) |
| CAUTION frames | 0 |
| SAFE frames | 251 |
| UNKNOWN frames | 0 |
| Frames with no offset | 0 |

## Notes

- *Valid-lane ratio* is the share of frames where geometry had enough evidence. It is **not** accuracy, precision or F1 — there is no ground truth here.
- *WARNING frames* counts only `warning == WARNING`; `CAUTION` is reported separately.
- *Processing FPS* is throughput on the machine above, not the source clip's FPS, and excludes Gradio upload time.

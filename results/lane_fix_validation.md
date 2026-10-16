# Lane recognition fixes - verification (2026-10-16)

## Changes

- Select and retain camera framing from marking-segment vanishing rows; map crop coordinates back to the original frame. Manual crop is available in the UI.
- Check model proposals against local white/yellow contrast, bound curve corrections, and reject short or unsupported proposals. Trim ambiguous convergence near the vanishing point.
- Restore newly observed polyline rows after a temporary occlusion; smooth only where history is available.
- Expose frame errors and distinguish missing lane geometry from pipeline failures.

## Verification

`python -m pytest -q`: **71 passed, 93 subtests passed**. `git diff --check` passed.
The running Gradio app was exercised through its upload/analyze API (30 real sample frames, zero processing failures).

| Input | Frames | Two accepted boundaries | UNKNOWN | Processing failures |
|---|---:|---:|---:|---:|
| sample_lines | 221 | 205 | 16 | 0 |
| sample_challenge | 251 | 231 | 20 | 0 |
| sample_yellow | 681 | 668 | 13 | 0 |
| Hanoi: first 10 seconds from original file | 300 | 127 | 173 | 0 |
| Hanoi: 12 sampled segments, re-encoded | 540 | 191 | 349 | 0 |

All output videos were decoded to verify frame counts; output codec is H.264. The Hanoi source is 1920x1080 at 30 FPS, 21,799 frames. Only the first 300 original frames and 12 short sampled sections were processed; this is not a full-video validation. Sample times (seconds): 0, 5, 10, 20, 40, 60, 120, 180, 300, 420, 540, 700. Each sampled section is 45 frames. Sampling introduces scene jumps and re-encoding changes the pixels.

## Limits

These counts measure availability of geometry, **not detection accuracy**. There are no independent ground-truth annotations. The existing UFLD model still misses markings on the Hanoi urban footage, especially at junctions and during vehicle occlusion. The contrast filter also rejects some faint markings. A road edge or vehicle detail can still be mistaken for a marking. WARNING is an instantaneous image-coordinate heuristic and has not been validated against vehicle position.

## Local review artifacts

- `outputs/lane_fix/hanoi_first10s.mp4`: continuous original-source preview.
- `outputs/lane_fix/hanoi_review.mp4`: 12 sampled sections.
- `outputs/lane_fix/hanoi_comparison.jpg`: before/after on identical original-source frames at 0, 60, 180, 300, 420 and 700 seconds, with independent tracker state at each timestamp. The new detector retains its first-frame camera calibration across timestamps.
- `outputs/lane_fix/samples_review.jpg`: full-run sample-video snapshots.
- `outputs/lane_fix/*_results.json` and CSVs: measured counts and per-frame records.

Artifacts stay local under ignored `outputs/`; they are not committed.

# LaneGuard AI — Presentation & Demo Script

Owner: **Thân**. Target: 3–5 slides, pitch **60–90 seconds**.

> **Honesty rule for this deck:** every number on a slide must come from a real,
> logged benchmark (`results/`). If a metric has not been measured, the slide
> says **"Not measured"**. Never put 95%, 15 FPS, 99%, or "80% fewer alerts" on a
> slide without a log to back it up.

---

## Slide 1 — Problem

**Title: Watching lane behaviour by hand does not scale**

- Dashcam/lane video is abundant, but judging lane-marking stability and
  false/again-missing warnings by eye is slow and inconsistent.
- Teams want a quick, local way to *see* whether a lane-detection model behaves
  and *measure* how fast it runs.
- Scope is deliberately a **prototype**: an illustrative lane estimate from
  recorded video, not a car safety system.

**Speaker line:** "We built a local tool that turns a dashcam clip into an
overlay video plus a metrics CSV, so lane-model behaviour can be seen and
measured quickly."

---

## Slide 2 — Our solution

**Title: Upload a clip → get an overlay + metrics**

- Web upload of an `.mp4` dashcam clip (Gradio, runs locally).
- **UFLD pretrained ONNX** lane detection (inference only — no training).
- OpenCV tracking across frames + geometry for a **normalized** lane offset.
- Output: overlay video, a per-frame **CSV**, and a summary of measured speed.
- Warnings are **illustrative**: `SAFE` / `CAUTION` / `WARNING`, and `UNKNOWN`
  when there is not enough evidence.

**Speaker line:** "One upload, one click, and you get the annotated video and the
numbers — no cloud, no training."

---

## Slide 3 — Architecture & technology

**Title: Small, honest pipeline**

```text
Video → UFLD ONNX → Tracker → Geometry → Overlay/UI → CSV + summary
         (detector)   (EMA)    (offset)   (Gradio)
```

- **ONNX Runtime (CPU)** + UFLD pretrained weights (inference only).
- **OpenCV / NumPy** for video I/O, tracking and geometry; **Gradio** for the UI.
- Design choices that keep it honest:
  - `UNKNOWN` guard when fewer than two lane boundaries are observed.
  - Offset is normalized image coordinates — **never** metres.
  - Wall-clock FPS reported separately from the source clip FPS.

**Speaker line:** "The pipeline is deliberately small: detect, track, compute
offset, draw. Every stage has a contract, and it says 'unknown' instead of
guessing."

---

## Slide 4 — Results

**Title: What we measured**

Fill this table **only** from `results/benchmark_report.md` after a real run.
Until then it stays as shown.

| Clip | Processing FPS | Avg latency (ms) | Valid-lane ratio | WARNING frames | Machine |
|---|---|---|---|---|---|
| `sample_lines.mp4` | Not measured | Not measured | Not measured | Not measured | — |
| `sample_yellow.mp4` | Not measured | Not measured | Not measured | Not measured | — |
| `sample_challenge.mp4` | Not measured | Not measured | Not measured | Not measured | — |

- Show 1–3 output clips.
- Quote **only** measured FPS / latency and the machine they were measured on.
- Show one **`UNKNOWN` / failure case** — it is evidence of honest behaviour, not
  a defect to hide.

**Speaker line:** "These are real numbers from our machine, not estimates. Notice
the challenge clip produces more `UNKNOWN` frames — that is the guard working."

---

## Slide 5 — Limitations & future work

**Title: What this is not (yet)**

- Camera is **not calibrated** → offsets are normalized, not metres; no physical
  curvature, no time-to-lane-crossing.
- Validated only on a few short highway clips; **not** on Vietnamese urban roads,
  rain, night or glare.
- No ground-truth labels → no accuracy / precision / F1 claims.
- Future work: a labelled test set + F1, camera calibration, a Vietnamese urban
  stress-test set, solid/dashed classification, merge/split detection.

**Speaker line:** "It is a working prototype with clear limits. Next steps are a
labelled set for real metrics and calibration for physical units."

---

## Demo script (60–90 seconds)

| Time | Action | Say |
|---|---|---|
| 0–10 s | App open, title visible | "LaneGuard AI — lane analysis on a dashcam clip." |
| 10–20 s | Upload a 5–10 s clip, set camera center, click **Analyze** | "Upload, one click. It runs a pretrained ONNX model locally." |
| 20–45 s | Play the overlay: lane polylines, track IDs, center marker, warning | "It tracks the lane, draws the center, and shows an illustrative warning." |
| 45–60 s | Show the metrics / CSV | "Here are the measured FPS and latency for this machine." |
| 60–75 s | Point at an `UNKNOWN` frame | "When it cannot see two boundaries, it says UNKNOWN instead of guessing." |
| 75–90 s | Close on limitations slide | "It is a prototype — normalized offsets, no calibration — with a clear path forward." |

**Always have a fallback:** a pre-recorded screen capture of the output video in
case live inference is slow during the pitch.

---

## One-line summary

> LaneGuard AI turns a dashcam clip into an annotated lane-overlay video and a
> metrics CSV using a pretrained ONNX model — a fast, honest prototype that
> reports `UNKNOWN` rather than inventing confidence.

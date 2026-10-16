# TASK — TRIỆU | Web App, Video Processing & Visualization

**Branch:** `trieu`  
**Vai trò:** Tạo giao diện Gradio chạy local, điều phối pipeline 3 module AI/tracking/geometry, vẽ video và xuất CSV. **Triệu là đầu mối tích hợp local** sau khi các thành viên push riêng.  
**Contract:** `PROJECT_OVERVIEW.md` mục **5.1–5.5**.  
**Mốc:** Web chạy với mock trước phút 35; sau phút 60 tích hợp model thật; demo chạy trước phút 100.

## 1. Definition of Done

Chạy được lệnh `python app.py` rồi mở local URL Gradio. Upload video MP4, bấm Analyze, xem video có polyline, tâm làn và trạng thái cảnh báo, tải CSV và đọc thống kê thời gian xử lý. Dùng đúng `LaneDetector.detect`, `LaneTracker.update`, `analyze_lanes`, không tự viết lại detector/tracker/geometry.

### File thuộc quyền sở hữu Triệu

- `app.py`
- `src/video.py`
- `src/visualizer.py`
- `requirements.txt`
- `.gitignore`
- `src/__init__.py` (nếu cần)
- `tests/test_pipeline.py`

**Không sửa:** `src/detector.py`, `src/tracker.py`, `src/geometry.py`, `benchmark.py`, README chung hoặc files người khác. Nếu thiếu module khi phát triển độc lập, mock bằng monkeypatch/stub **chỉ trong test**, không commit bản giả của module người khác.

## 2. Cấu trúc kiến trúc

```text
Gradio app.py
  └── process_video(input_path, output_path, model_path, camera_center_ratio, max_frames)
        ├── cv2.VideoCapture()
        ├── LaneDetector(model_path, device="cpu")   # khởi tạo 1 lần mỗi video
        ├── LaneTracker(alpha=0.35, max_missed=3)   # khởi tạo 1 lần mỗi video
        └── per frame:
            ├── detection = detector.detect(frame_bgr)
            ├── tracking = tracker.update(detection)
            ├── analysis = analyze_lanes(tracking, camera_center_ratio)
            ├── annotated = render_frame(frame_bgr, tracking, analysis)
            ├── writer.write(annotated)
            └── write csv row + aggregate stats
```

Cố gắng dùng import kiểu `from src.detector import LaneDetector`, tương tự tracker/geometry. Trong lúc code riêng trên branch trieu chưa có các file đó, có thể **đưa các import tích hợp bên trong `process_video`** để `app.py` vẫn có thể khởi chạy giao diện, và kiểm thử một bản `process_video` mock trong test. Khi tích hợp local, imports phải chạy được trực tiếp.

## 3. API contract Triệu cung cấp

### `src/visualizer.py`

```python
import numpy as np

def render_frame(frame_bgr: np.ndarray, tracking: dict, analysis: dict) -> np.ndarray:
    """Return a NEW BGR uint8 image of exactly the same H×W×3."""
```

Yêu cầu:

- `tracking["lanes"]` chứa `track_id`, `points`, `confidence`, `predicted`, `missed`.
- Vẽ đường làn quan sát bằng polyline màu xanh/cyan; lane `predicted` dùng màu khác hoặc nét khác để phân biệt dự đoán.
- Chỉ vẽ tâm làn/vùng làn đã chọn nếu `analysis["valid_lane"] == True`; vị trí tâm **chỉ có tại `reference_y`** trong contract, nên vẽ marker tại hàng đó và thông tin trên màn hình; không tự bịa cả đường centerline dài khi chỉ biết 1 điểm.
- Overlay `offset_normalized` và `warning` (SAFE/CAUTION/WARNING/UNKNOWN). Khi UNKNOWN, ghi rõ `LANE UNKNOWN` không tô vùng làn xanh như đã biết chắc.
- Text "Lane estimate — demo only" hoặc tương đương (không mô tả là hệ thống bảo đảm an toàn).
- Không chỉnh ảnh gốc inplace: `canvas = frame_bgr.copy()`.
- Trường hợp `lanes=[]`, các trường None, points ngắn => không crash.

### `src/video.py`

```python
def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    camera_center_ratio: float = 0.5,
    max_frames: int | None = None
) -> dict: ...
```

**Return schema chính xác:**

```python
{
    "output_video": str,
    "csv_path": str,
    "summary": {
        "frames_processed": int,
        "processing_seconds": float,
        "processing_fps": float,
        "video_fps": float,
        "avg_inference_ms": float,
        "valid_lane_frames": int,
        "warning_frames": int,
        "failed_frames": int
    }
}
```

**CSV schema và thứ tự đề xuất:**

```text
frame_index,time_s,detector_ms,tracker_ms,offset_normalized,warning,curve,left_track_id,right_track_id,valid_lane
```

- `frame_index` đếm từ 0; `time_s=frame_index / video_fps` nếu fps hợp lệ, nếu không fps fallback 30.0 kèm chú thích fallback.
- `detector_ms = detection["inference_ms"]`, `tracker_ms = tracking["tracking_ms"]`.
- `offset_normalized` ghi trống khi None; `warning` lấy giá trị tức thời geometry, không đổi thành SAFE khi UNKNOWN.
- `warning_frames`: số frame có `warning=="WARNING"`, không cộng CAUTION; tài liệu đọc chỉ số phải ghi rõ.
- `processing_seconds`: thời gian wall-clock từ trước xử lý frame đầu đến sau khi đóng writer/CSV (có thể gồm init model nếu đo cả hàm, phải nhất quán khi ghi README).
- `processing_fps=frames_processed/max(processing_seconds, 1e-9)`; **khác** `video_fps`, vốn là fps của nguồn video. FPS benchmark chưa tính thời gian Gradio upload từ mạng (local).
- `avg_inference_ms`: trung bình `inference_ms` ở các frame detector chạy thật; nếu không có frame => 0.0 + thông báo lỗi nếu video rỗng.

### `app.py`

```python
def build_app():
    """Return gr.Blocks with inputs video path and camera center ratio."""

if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False)
```

Input UI: `gr.Video(label="Input Dashcam Video", sources=["upload"], format="mp4")`; slider camera center ratio trong `[0.4, 0.6]`, mặc định 0.5. Output UI: `gr.Video(label="Analyzed Video")`, `gr.JSON(label="Metrics")`, `gr.File(label="CSV")`, thông báo lỗi.

Hàm callback có thể `return output_video, summary, csv_path, "Success"`; không trả nội dung bytes video lớn vào Gradio. Mỗi lượt dùng folder `outputs/<unique_id>/`, không ghi đè file giữa các lượt. Chỉ nhận video local đã upload, không nhận URL từ người dùng để download tùy ý.

## 4. Triển khai Video Pipeline — từng bước

### A. Khởi tạo video + writer

- Validate `input_path` tồn tại & đọc được (`cap.isOpened()`).
- `W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))`, `H`, `source_fps`; nếu FPS nguồn 0/NaN, fallback 30 và đánh dấu trong log.
- `output_path` nằm trong `outputs/`, tạo parent directory.
- Tạo `cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))`, kiểm tra `.isOpened()`. OpenCV `mp4v` có thể không phát ở một số trình duyệt; cần kiểm tra với Gradio, nếu có FFmpeg thì chuyển sang H.264 bằng subprocess **không cho phép shell=True với input tùy ý**, ví dụ `ffmpeg -y -i intermediate.mp4 -vcodec libx264 -pix_fmt yuv420p output.mp4`.
- Ghi CSV bằng `csv.DictWriter` của stdlib (không cần pandas).

### B. Xử lý frame

```python
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    detection = detector.detect(frame)
    tracking = tracker.update(detection)
    analysis = analyze_lanes(tracking, camera_center_ratio)
    rendered = render_frame(frame, tracking, analysis)
    writer.write(rendered)
    # write CSV row / aggregate metrics
```

Dùng `try/finally` để đóng `cap`, `writer`, CSV dù lỗi. Nên **fail rõ** ở lỗi thiếu model/decoder thay vì âm thầm xuất video toàn UNKNOWN trông như AI đã chạy. Trường hợp lỗi frame đơn lẻ nếu vẫn muốn bỏ qua thì đánh dấu failed frame với UNKNOWN, giữ fps bằng ghi ảnh gốc + overlay lỗi; phải log số lỗi.

### C. Giảm tải CPU

Không tự ý resize frame toàn bộ rồi quên scale output: Quang đã resize input model nội bộ và đưa tọa độ về frame gốc. Chỉ cân nhắc giảm độ phân giải **video đầu vào** khi thực sự cần, với xử lý đồng bộ W/H, theo quyết định nhóm. `max_frames` cho benchmark/demo ngắn, mặc định None.

### D. Video playback

Nếu MP4 output không phát trong UI, kiểm tra bằng `ffprobe` hoặc `ffmpeg`, thử H.264 với pixel format yuv420p. Nếu FFmpeg không cài được, Gradio có thể thử transcode, nhưng phải **kiểm tra thật trên browser**; không khẳng định tương thích mọi máy.

## 5. `requirements.txt` và `.gitignore`

Đây là danh sách package khởi đầu, điều chỉnh theo import thực trong module của Quang; **chỉ Triệu sửa file**:

```text
numpy
opencv-python
onnxruntime
gradio
scipy
matplotlib
```

Không cài cả `opencv-python` và `opencv-python-headless` trong cùng môi trường. Không cài `torch`/`torchvision` khi đã chạy ONNX. Có thể dùng `pip freeze` để ghi lại version sau khi setup; nếu cần môi trường tái lập, pin version chạy được **sau khi kiểm chứng**.

`.gitignore` ít nhất gồm:

```gitignore
.venv/
__pycache__/
*.pyc
outputs/
results/*.mp4
results/*.csv
models/*.onnx
data/*.mp4
data/*.avi
.gradio/
```

Tránh `.gitignore` che README trong `models/` hoặc `data/`.

## 6. Unit test trong khi chưa có module người khác

`tests/test_pipeline.py`:

- Tạo video nhân tạo 60 frame bằng cv2.VideoWriter (đường và vạch vẽ lên ảnh), dùng mock detector/tracker/geometry theo đúng contract để test **wiring, video output, CSV, UI callback**. Đây là test wiring, **không phải minh chứng UFLD hoạt động**.
- Test `render_frame` với lane đủ điểm và lane rỗng; không thay input, đúng shape và dtype.
- Test `process_video` xử lý đúng `max_frames`, `summary.frames_processed`, số dòng CSV, file output được tạo và mở bằng OpenCV.
- Test lỗi file video không có, model path không có; phải trả thông báo hợp lý.
- Có thể monkeypatch `sys.modules` hoặc patch các import trong function để tránh việc phải commit bản fake `src/detector.py` của Quang. Nếu quá lâu, tạo test chỉ cho `render_frame` và để kiểm tra pipeline integration sau.

**Không mock detector ở bản demo chấm điểm**. Mọi video đầu ra được trình bày là demo AI phải dùng weights pretrained thật.

## 7. Giai đoạn tích hợp local (sau push, trong phiên 120 phút)

Triệu là người giữ **máy chạy demo**, không cần push branch tích hợp lên GitHub. Sau khi 5 người đã push:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c integration-local
git merge --no-ff origin/quang
git merge --no-ff origin/viet-anh
git merge --no-ff origin/dien
git merge --no-ff origin/trieu
git merge --no-ff origin/than
python -m pip install -r requirements.txt
python app.py
```

**Lưu ý:** Lệnh tạo `integration-local` chỉ làm ở clone tích hợp (một máy riêng hoặc checkout sạch); không chạy trong working tree branch `trieu` đang có thay đổi chưa commit. Nếu repo main thực sự rỗng thì bootstrap trước khi tạo các branch, theo overview.

Khi bắt đầu tích hợp, đối chiếu chính xác **signature** và output schema trước tiên, không chữa tạm bằng hard-code khác contract.

## 8. Timeline chi tiết của Triệu

| Phút | Nhiệm vụ | Output |
|---|---|---|
| 0–10 | Setup Gradio, requirements, mock contract | App có upload control |
| 10–25 | `render_frame`, video reader/writer, CSV | Test visualizer trên frame giả |
| 25–35 | Callback Gradio với mock, video sample ngắn | Có video output thử |
| 35–50 | Bắt lỗi, output codec, hoàn thiện `process_video` | Tests đơn vị cơ bản |
| 50–60 | Commit/push `trieu` | Branch sẵn sàng merge |
| 60–80 | Tạo `integration-local`; merge 5 branch, chạy end-to-end | AI thật xử lý ≥1 video |
| 80–100 | Cùng cả nhóm fix & test clip | UI có video, CSV, summary |
| 100–120 | Freeze, giúp Thân quay demo, tập pitch | App ổn định |

## 9. Checklist bàn giao

- [ ] `python app.py` khởi động Gradio trên `127.0.0.1`, không yêu cầu cloud/public share.
- [ ] Upload MP4 → processing → output video + CSV + summary.
- [ ] `process_video` trả đúng schema, CSV đúng cột và offset None không đổi thành 0.
- [ ] Rendered BGR cùng size, predicted lane phân biệt bằng màu; UNKNOWN hiển thị trung thực.
- [ ] Đo `processing_fps` theo đồng hồ thực, không tráo với fps video gốc.
- [ ] OpenCV/FFmpeg output mở được trong trình duyệt demo.
- [ ] Khi chưa có detector thật chỉ test mock; demo chính thức buộc có ONNX.
- [ ] `requirements.txt` và `.gitignore` chuẩn; không sửa file người khác.
- [ ] Branch `trieu` đã push; merge tích hợp giữ local chưa push main.

## 10. Git commands trên nhánh cá nhân

```bash
git switch main
git switch -c trieu
# coding...
git add app.py src/video.py src/visualizer.py src/__init__.py requirements.txt .gitignore tests/test_pipeline.py
git commit -m "feat(app): add Gradio video processing and overlay"
git push -u origin trieu
```

Nếu không tạo `src/__init__.py` thì bỏ path khỏi lệnh `git add`.

## 11. Prompt cho Codex / Claude Code

> Đọc **PROJECT_OVERVIEW.md mục 5** và **TASK_TRIEU.md**. Chỉ làm `app.py`, `src/video.py`, `src/visualizer.py`, `requirements.txt`, `.gitignore`, `src/__init__.py` (nếu cần), `tests/test_pipeline.py`. Xây Gradio upload MP4, slider camera center ratio, output video/JSON metrics/CSV. Hàm `render_frame(frame_bgr,tracking,analysis)` trả BGR copy cùng shape; `process_video(input_path,output_path,model_path,camera_center_ratio=0.5,max_frames=None)` init LaneDetector ONNX, LaneTracker và gọi analyze_lanes theo contract, viết CSV đúng cột, summary đo wall-clock FPS thực. Module của người khác chưa có trong nhánh này nên dùng import lười và test bằng mock KHÔNG COMMIT module giả; demo cuối chạy pretrained thật. Video codec phải tương thích browser (cân nhắc ffmpeg H.264), output folder unique, `try/finally` release. Không đổi API và không sửa file ngoài quyền sở hữu. Trả danh sách file, lệnh chạy, trạng thái test thật vs mock, lỗi còn lại. Push branch `trieu`, chỉ merge local khi đồng đội push xong, không push main.

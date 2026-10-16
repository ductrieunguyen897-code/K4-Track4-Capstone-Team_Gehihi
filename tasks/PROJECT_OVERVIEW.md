# LaneGuard AI — PROJECT OVERVIEW & INTEGRATION CONTRACT

> **Mục tiêu:** MVP hackathon phân tích video camera hành trình để phát hiện/theo dõi làn đường và hiển thị cảnh báo *minh họa*. 5 người phát triển song song, thời hạn **120 phút**, ngân sách **0 đồng**.
>
> **Trạng thái:** Tài liệu đặc tả để triển khai; **chưa phải** báo cáo rằng ứng dụng đã hoạt động hoặc đã đạt benchmark.
>
> **Phân công mặc định (có thể đổi tên người trước khi bắt đầu):** Quang = AI; Việt Anh = tracking; Diện = geometry/cảnh báo; Triệu = web/pipeline; Thân = dữ liệu/benchmark/demo. Người dùng chưa cung cấp ánh xạ nhiệm vụ cụ thể, nên đây là đề xuất tạm thời.

## 1. Tuyên bố sản phẩm

**LaneGuard AI** là web app chạy **trên máy cá nhân**, tải một video `.mp4` quay từ phía trước xe, chạy model **UFLD pretrained ONNX** trên CPU kết hợp **OpenCV**, theo dõi các vạch làn qua frame, ước lượng vị trí của xe so với tâm làn trong **tọa độ ảnh** và xuất video có overlay cùng CSV chỉ số.

**Mục tiêu demo:** người xem mở web local → tải video → bấm Analyze → nhận video vẽ vạch và vùng làn → thấy nhãn `SAFE`, `CAUTION`, `WARNING` hoặc `UNKNOWN` → tải CSV/đọc FPS, thời gian xử lý.

**Không phải** hệ thống an toàn lắp trên ô tô; không dùng để ra quyết định lái xe. Không mô tả sai lệch làn bằng mét, độ cong bằng 1/m hoặc khả năng cảnh báo chuẩn ADAS khi chưa hiệu chuẩn camera và xác thực dữ liệu.

### Cam kết ràng buộc

| Hạng mục | Điều kiện | Cách thực hiện |
|---|---|---|
| Thời gian | 2 giờ, bao gồm tích hợp/demo | Chỉ làm P0 trước, chốt tính năng ở phút 80 |
| Nhân lực | 5 người hiểu AI/CV | Mỗi người một cụm file, branch riêng |
| Ngân sách | 0 | Chạy local, model công khai, video mẫu nhỏ |
| AI | **Bắt buộc pretrained** | UFLD ONNX, inference-only; **không train/fine-tune** |
| Xử lý | Ưu tiên OpenCV | OpenCV đọc/ghi video, geometry, overlay |
| Phần cứng | Laptop phổ thông | ONNX Runtime CPU; không bắt buộc CUDA |
| Tốc độ | Mục tiêu ~10–15 FPS | **Mục tiêu cần đo**, không cam kết trước |
| Kết quả | Web + GitHub + video demo + benchmark | Gradio + 5 branch, CSV + biểu đồ + trình diễn |

## 2. Phạm vi — làm đúng thứ tự ưu tiên

### P0: phải hoàn thành trong 120 phút

1. Nhận video MP4; đọc frame và xuất video có thể phát trong trình duyệt.
2. Chạy **UFLD pretrained ONNX** thật để phát hiện các polylines của vạch.
3. Theo dõi ID của các vạch qua các frame và làm mượt (association + EMA).
4. Tìm 2 ranh giới kẹp trục camera (nếu quan sát đủ); vẽ tâm làn.
5. Tính `offset_normalized` (không phải mét); cảnh báo `SAFE`/`CAUTION`/`WARNING`/`UNKNOWN`.
6. Báo FPS thực đo, thời gian inference, tỷ lệ frame có đủ 2 vạch; xuất CSV.

### P1: chỉ làm khi P0 đã chạy

- Phân loại đường thẳng/cong từ fit đa thức **trong ảnh** (chỉ là proxy).
- Gợi ý vạch liền/đứt từ phân tích cường độ ảnh ở vị trí vạch; khi không chắc => `UNKNOWN`.
- Biểu đồ và lựa chọn clip mẫu đẹp hơn.

### P2: loại khỏi cam kết 2 giờ

- Nhận biết **đáng tin cậy** làn nhập/tách, 3D lane geometry, hiệu chuẩn khoảng cách mét, lane change intention, cảnh báo an toàn ô tô thực tế. Có thể nêu là hướng phát triển nhưng **không ghi là hoàn thành**.

## 3. Công nghệ chuẩn (không thay đổi tùy ý)

- **Python ≥ 3.10**; ưu tiên một môi trường ảo chung.
- **ONNX Runtime (CPU)**, UFLD ONNX pretrained; không có training/fine-tune.
- **OpenCV**, **NumPy**, tùy chọn SciPy để dùng hậu xử lý của repo gốc.
- **Gradio** giao diện local; `gr.Video` nhận/trả đường dẫn file, `gr.File` cho CSV, `gr.JSON` hoặc `gr.Markdown` cho thống kê.
- **matplotlib** chỉ khi cần biểu đồ; không kéo thêm React, FastAPI, Docker, database, Redis.

**Nguồn model:** https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference ; repo ghi nhận mô hình UFLD ONNX input RGB 800×200, output điểm cho tối đa 4 làn, trỏ tới model chuyển đổi tại https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection . **Người 1 cần kiểm tra chính xác file ONNX tải được, input/output thực tế và giấy phép trước khi tích hợp.** File model ở runtime: `models/ufld_tusimple.onnx` (tên nội bộ, đổi tên file tải về phù hợp sau khi xác minh).

**Video test:** https://github.com/udacity/CarND-LaneLines-P1 (`test_videos`); ưu tiên clip nhỏ. Video đô thị Việt Nam chỉ là kiểm thử bổ sung nếu có quyền dùng; không buộc tải dataset lớn.

## 4. Cấu trúc source đích và quyền sở hữu file

```text
laneguard-ai/
├── app.py                     # Triệu
├── requirements.txt           # Triệu
├── .gitignore                 # Triệu
├── src/
│   ├── __init__.py            # Triệu (nếu cần)
│   ├── detector.py             # Quang
│   ├── tracker.py              # Việt Anh
│   ├── geometry.py             # Diện
│   ├── video.py                # Triệu
│   └── visualizer.py          # Triệu
├── benchmark.py               # Thân
├── README.md                  # Thân
├── models/
│   └── README.md              # Quang; *.onnx không commit mặc định
├── data/
│   └── README.md              # Thân; video không commit mặc định
├── docs/
│   └── presentation.md        # Thân
├── tests/
│   ├── test_detector.py       # Quang
│   ├── test_tracker.py        # Việt Anh
│   ├── test_geometry.py       # Diện
│   ├── test_pipeline.py       # Triệu
│   └── smoke_test.py          # Thân
├── outputs/                   # sinh khi chạy; không commit
└── results/                   # sinh khi chạy; chỉ commit số liệu nhỏ nếu cần
```

**Không sửa file do thành viên khác sở hữu**, không đổi API contract mà không báo nhóm. Người dùng yêu cầu **mỗi người chỉ push branch riêng**; nhóm chỉ tích hợp **sau khi các module sẵn sàng**, trên branch local `integration-local`, không tự động push `main`.

## 5. API CONTRACT — PHẦN BẮT BUỘC ĐỌC TRƯỚC KHI CODE

### 5.1 Quy ước toàn cục

- Input frame: `np.ndarray`, `uint8`, shape `(H, W, 3)`, **BGR theo OpenCV**.
- Các `points`: `[[x, y], ...]` là **tọa độ pixel frame GỐC**, `x` sang phải, `y` xuống dưới; sắp từ **y nhỏ đến y lớn**, giá trị hợp lệ trong ảnh, không chứa NaN; cần **ít nhất 2 điểm** cho mỗi lane.
- `lane_index`: thứ tự detection trong frame, **không phải ID bền vững**. `track_id`: ID bền vững do tracker cấp trong video; không gán cố định ID cho trái/phải xe.
- `confidence`: `float` 0–1 nếu model có ý nghĩa xác định; nếu không => **`None`**, không chế ra xác suất.
- Thời gian: `*_ms` là millisecond, FPS là frame/giây thực đo (wall-clock).
- Empty detection => `lanes: []`, không trả None cả kết quả, không crash.
- Không ngầm lấy dữ liệu tương lai, không giả định xe đặt camera chính giữa: UI có `camera_center_ratio` mặc định 0.5.

### 5.2 Quang → Detector

**File:** `src/detector.py`

```python
class LaneDetector:
    def __init__(self, model_path: str, device: str = "cpu") -> None: ...
    def detect(self, frame_bgr: "np.ndarray") -> dict: ...
```

**Output bắt buộc**:

```python
{
    "lanes": [
        {"lane_index": 0, "points": [[180, 280], [175, 340], [163, 420]], "confidence": None},
        {"lane_index": 1, "points": [[460, 280], [473, 340], [489, 420]], "confidence": None}
    ],
    "frame_width": 640,
    "frame_height": 480,
    "inference_ms": 16.4
}
```

Điểm ví dụ chỉ minh họa schema, **không phải kết quả mô hình**. Tạo 1 session ONNX duy nhất khi khởi tạo; không nạp model mỗi frame. Tự đảm nhiệm preprocess RGB/model shape, decode/postprocess và rescale điểm về frame gốc.

### 5.3 Việt Anh → Tracker

**File:** `src/tracker.py`

```python
class LaneTracker:
    def __init__(self, alpha: float = 0.35, max_missed: int = 3) -> None: ...
    def update(self, detection: dict) -> dict: ...
    def reset(self) -> None: ...
```

**Output bắt buộc**:

```python
{
    "lanes": [
        {"track_id": 1, "points": [[180, 280], [175, 340], [163, 420]],
         "confidence": None, "predicted": False, "missed": 0}
    ],
    "frame_width": 640,
    "frame_height": 480,
    "tracking_ms": 0.8
}
```

`predicted=True` khi lane không có observation frame hiện tại; `missed` là số frame liên tiếp bị mất. Chỉ dự đoán tối đa `max_missed` frame; sau đó xóa track. Tracking `confidence` không được tự nâng lên khi chỉ có dự đoán.

### 5.4 Diện → Geometry + Warning

**File:** `src/geometry.py`

```python
def analyze_lanes(tracking: dict, camera_center_ratio: float = 0.5) -> dict: ...
```

**Output bắt buộc**, kể cả khi không tìm được vạch:

```python
{
    "valid_lane": True,
    "reference_y": 408,
    "vehicle_center_x": 320.0,
    "lane_center_x": 313.0,
    "left_track_id": 1,
    "right_track_id": 2,
    "offset_normalized": 0.024,
    "warning": "SAFE",         # SAFE | CAUTION | WARNING | UNKNOWN
    "curve": "UNKNOWN",       # STRAIGHT | CURVED | UNKNOWN
    "lane_type_left": "UNKNOWN",  # SOLID | DASHED | UNKNOWN
    "lane_type_right": "UNKNOWN"
}
```

`reference_y = round(0.85*(H-1))`. Lấy hai lane gần nhất **kẹp** trục camera tại `reference_y` (nội suy trong miền quan sát; không ngoại suy tùy tiện). `xV=camera_center_ratio*(W-1)`, `xC=(xL+xR)/2`, `e=(xV-xC)/(xR-xL)`; **`e>0` = xe/camera nằm bên phải tâm làn**, `e<0` = nằm bên trái. Ngưỡng demo: `abs(e)<0.15` → SAFE, `[0.15, 0.25)` → CAUTION, `>=0.25` → WARNING.

**BẮT BUỘC UNKNOWN** nếu thiếu 1 trong 2 lane, width không hợp lệ, lane ngoài miền `reference_y`, hoặc chỉ có dự đoán cũ (`missed>1`), hoặc **không có ít nhất một quan sát thật** ở frame hiện tại. Khi UNKNOWN: `valid_lane=False`, `offset_normalized=None`, `lane_center_x=None`, track ID thiếu -> `None`, `warning='UNKNOWN'`. Đây **không phải** cảnh báo an toàn đã chứng thực.

### 5.5 Triệu → Visualization + Pipeline + Web

**Files:** `src/visualizer.py`, `src/video.py`, `app.py`

```python
def render_frame(frame_bgr: "np.ndarray", tracking: dict, analysis: dict) -> "np.ndarray": ...

def process_video(
    input_path: str, output_path: str, model_path: str,
    camera_center_ratio: float = 0.5,
    max_frames: int | None = None
) -> dict: ...

def build_app(): ...  # returns gr.Blocks
```

`render_frame` phải trả ảnh BGR mới cùng `shape`, không sửa input tại chỗ. Pipeline phải giữ thứ tự `detect→update→analyze→render`, init model/tracker **một lần mỗi video**, `reset` khi video mới.

**Output `process_video` bắt buộc:**

```python
{
    "output_video": "outputs/demo_processed.mp4",
    "csv_path": "outputs/demo_processed.csv",
    "summary": {
        "frames_processed": 180,
        "processing_seconds": 12.0,
        "processing_fps": 15.0,
        "video_fps": 30.0,
        "avg_inference_ms": 36.0,
        "valid_lane_frames": 150,
        "warning_frames": 10,
        "failed_frames": 0
    }
}
```

`summary` là ví dụ schema, **không phải benchmark**. CSV bắt buộc có cột: `frame_index,time_s,detector_ms,tracker_ms,offset_normalized,warning,curve,left_track_id,right_track_id,valid_lane`. `time_s=frame_index/source_video_fps` khi fps hợp lệ. `warning` là cảnh báo tức thời từ geometry; có thể bổ sung nhãn cảnh báo debounce trong overlay, nhưng không được ghi sai CSV.

Khi exception nghiêm trọng không xuất video rỗng giả thành công; thông báo lỗi rõ ràng. Khi lỗi tại một frame nhưng có thể tiếp tục, ghi `failed_frames` và frame nguyên bản/gắn UNKNOWN theo chính sách đã ghi trong README.

### 5.6 Thân → Benchmark + Demo

**File:** `benchmark.py`

```bash
python benchmark.py --input data/sample.mp4 --model models/ufld_tusimple.onnx --max-frames 300
```

Dùng trực tiếp `process_video(...)`, không viết một model inference thứ hai. Báo `processing_fps` đo thực, `video_fps`, `avg_inference_ms`, `valid_lane_frames/frames_processed`, tổng warning, CPU/OS (nếu truy xuất được), clip, độ phân giải, số frame. Xuất bảng Markdown/CSV; tùy chọn vẽ biểu đồ. Không bịa metrics khi chưa có phép đo. Test thủ công khoảng **10 frame/clip × 3 clips** (nếu kịp), đánh dấu phát hiện đúng hai ranh giới theo tiêu chuẩn nhóm định nghĩa.

## 6. Phân nhánh Git — không push main

| Thành viên | Branch ASCII chính xác | File sở hữu |
|---|---|---|
| Quang | `quang` | `src/detector.py`, `models/README.md`, `tests/test_detector.py` |
| Việt Anh | `viet-anh` | `src/tracker.py`, `tests/test_tracker.py` |
| Diện | `dien` | `src/geometry.py`, `tests/test_geometry.py` |
| Triệu | `trieu` | `app.py`, `src/video.py`, `src/visualizer.py`, `requirements.txt`, `.gitignore`, `src/__init__.py`, `tests/test_pipeline.py` |
| Thân | `than` | `benchmark.py`, `README.md`, `data/README.md`, `docs/presentation.md`, `tests/smoke_test.py` |

**Repository:** người dùng ghi `repo tro·` nhưng chưa cung cấp URL nhận diện được. Thay `<REPO_URL>` bằng URL thực, không đoán tên repo.

Nếu GitHub repository **thực sự rỗng, chưa có `main`**, một người được chọn bootstrap *một lần* (README tối thiểu hoặc commit đầu tiên) rồi `git push -u origin main`; **đây là ngoại lệ khởi tạo ban đầu được nhóm phê duyệt**, không phải merge code lên main. Nếu đã có main, bỏ qua bước này.

Lệnh cho từng người (thay `<branch>`):

```bash
git clone <REPO_URL>
cd <FOLDER>
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c <branch>
# Chỉ tạo/sửa file mình sở hữu
git add <file-cua-minh-1> <file-cua-minh-2>
git commit -m "feat: implement <module>"
git push -u origin <branch>
```

Không `git add .` nếu có video/weights. Không `force push`, không push `main`, không sửa chung `requirements.txt` (Triệu sở hữu). Nếu cần thêm dependency, gửi tên/version cho Triệu. Bảo đảm remote branch tồn tại bằng `git ls-remote --heads origin <branch>`.

**Tích hợp khi đã có đủ code:** tại máy điều phối, tạo branch **chỉ local** từ `main`:

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
# Không push integration-local hoặc main khi chưa được nhóm thống nhất
python app.py
```

Nếu merge xung đột: dừng, đối chiếu file ownership, báo đúng người sở hữu sửa, không lặng lẽ đổi API. Tích hợp **sau giai đoạn code độc lập nhưng vẫn trong phiên 120 phút** (mục tiêu từ phút 60). Mỗi thành viên chỉ đẩy code của chính mình lên branch riêng.

## 7. Timeline 120 phút & mốc Go/No-Go

| Phút | Cả nhóm | Điều kiện qua cổng |
|---|---|---|
| 00–10 | Chốt contract, bootstrap `main` nếu repo rỗng, tạo branch, cài dependency | Mọi người xác nhận đúng branch và quyền sở hữu file |
| 10–35 | Coding song song, dùng fake data để test giữa module | Quang phải chạy được ONNX 1 frame; các module còn lại có unit test mock |
| 35–60 | Hoàn thiện module, push lên 5 branch | Mỗi branch có code chạy độc lập; danh sách file không xung đột |
| 60–80 | Triệu/điều phối tích hợp local, sửa lỗi input/output | Chạy `python app.py`; xử lý tối thiểu 1 video MP4 |
| 80–100 | Test 2–3 clip, fix blockers, đo benchmark | Model thật + overlay + CSV + playback thành công |
| 100–115 | Chốt video demo, README, slide, ảnh/video minh chứng | Có clip quay màn hình dự phòng |
| 115–120 | Dry-run pitch và trình diễn | Toàn bộ flow chạy được; 5 branch giữ nguyên |

**Nếu phút 35 model chưa có:** Quang ưu tiên tải/kiểm tra file ONNX & decoder. Nhóm khác vẫn test bằng input dict giả. Không được đổi tên một heuristic thuần OpenCV thành pretrained AI.

**Nếu phút 60 chưa ra video:** bỏ P1/P2, xử lý duy nhất output frame + codec + Gradio.

**Nếu phút 80 chưa có AI inference end-to-end:** tập trung một đường xử lý video ngắn (5–10 giây) thay vì tối ưu FPS; tuyệt đối không bịa video đầu ra hoặc benchmark.

## 8. Test tích hợp tối thiểu

1. Detector trên frame thật: kết quả `dict`, 0–4 lane, tất cả điểm pixel nằm trong frame gốc, `inference_ms >= 0`.
2. Tracker qua 3 frame giả: lane gần nhau giữ nguyên `track_id`; có thể xử lý `lanes=[]`; xóa track sau `max_missed`.
3. Geometry: hai lane kẹp tâm → `SAFE` gần tâm; xe dịch ngang → CAUTION/WARNING; thiếu một lane → UNKNOWN; offset sign đúng.
4. Video: đầu vào 5–15 giây; xuất video có thể mở bằng browser, giữ fps hợp lý, CSV đủ cột.
5. UI: upload → Analyze → output + summary + CSV; lỗi thiếu model thân thiện.
6. Benchmark: phân biệt **processing FPS** (thực đo) và **source video FPS**; ghi model/CPU/clip.

## 9. Tiêu chí nghiệm thu và cách trình bày trung thực

- **Bắt buộc:** 1 video hoàn chỉnh xử lý bằng ONNX pretrained, UI phát được, overlay có ≥2 vạch khi model thấy đủ, offset cảnh báo dùng UNKNOWN khi thiếu dữ liệu, CSV thống kê.
- **Mong muốn:** 3 video, 10–15 FPS trên laptop nhóm (**phải đo**), giảm rung thấy được, chart minh họa giá trị đo.
- **Không cam kết:** hoạt động mọi tuyến đường Việt Nam, hỗ trợ lái xe an toàn, dự đoán nhập/tách làn chính xác, tracking 3D, mét hoặc m/s chuẩn.
- **Checklist bàn giao:** 5 branch đã push; 1 máy tích hợp local; demo video; README; benchmark ghi điều kiện; slide 3–5 trang.

## 10. Tài liệu/tham chiếu kỹ thuật

- UFLD ONNX demo: https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference
- Nguồn ONNX: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection
- UFLD gốc: https://github.com/cfzd/Ultra-Fast-Lane-Detection
- Udacity lane clips: https://github.com/udacity/CarND-LaneLines-P1
- Gradio Video: https://www.gradio.app/docs/gradio/video
- ONNX Runtime: https://onnxruntime.ai/docs/

> **Quy tắc cuối:** Mọi file TASK tham chiếu **đúng contract trong file này**. Nếu phát hiện mâu thuẫn, xem PROJECT_OVERVIEW.md là nguồn chân lý duy nhất, báo nhóm trước khi thay đổi chữ ký hàm.

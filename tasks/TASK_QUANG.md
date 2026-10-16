# TASK — QUANG | UFLD ONNX Lane Detection

**Branch:** `quang`  
**Vai trò:** Chủ module nhận diện vạch làn bằng model pretrained; không phụ trách tracking/geometry/web.  
**Nguồn chân lý:** `PROJECT_OVERVIEW.md`, mục **5.1–5.2**.  
**Deadline:** Module chạy độc lập trên 1 ảnh ở phút 35; code đã push trước phút 60.

## 1. Mục tiêu cuối cùng (Definition of Done)

Tạo `src/detector.py` cho phép một máy laptop CPU nhận frame BGR bất kỳ từ OpenCV, chạy inference bằng **model UFLD pretrained ONNX** và trả về danh sách làn dưới dạng **tọa độ pixel ảnh GỐC** theo contract. Cung cấp smoke test chạy được trên một frame thật; không huấn luyện/fine-tune.

### File chỉ Quang được sửa

- `src/detector.py` — class `LaneDetector` chính.
- `tests/test_detector.py` — unit tests/smoke tests của detector.
- `models/README.md` — nguồn weights, link tải, tên file, checksum nếu có, cấu hình input/output, lưu ý license.

**Không sửa:** `src/tracker.py`, `src/geometry.py`, `src/video.py`, `app.py`, `requirements.txt`, `README.md`, các task của người khác. Nếu cần dependency mới, báo Triệu.

## 2. Tài liệu và nguồn model

- ONNX inference reference: https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference
- ONNX model collection: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection
- Kiến trúc gốc: https://github.com/cfzd/Ultra-Fast-Lane-Detection

Repo inference tham khảo nêu input model RGB **800×200** và output các điểm tối đa **4 làn**. **Đó là thông tin theo repo, không phải bảo đảm mọi file ONNX giống nhau**; đọc `session.get_inputs()/get_outputs()` trước khi hardcode preprocess/postprocess. Ưu tiên biến thể UFLD ResNet-18 pretrained nếu có file tương thích và tải được; xác minh dạng `.onnx` thật, không dùng file `.pth` đưa thẳng vào ONNX Runtime. Trọng số phải xuất xứ rõ ràng, không đổi sang random initialization.

**File runtime chuẩn:** `models/ufld_tusimple.onnx`. Không commit weights dung lượng lớn theo mặc định; `models/README.md` chỉ rõ cách đặt file tải về đúng tên này. Nếu kho model không truy cập được, báo vấn đề với nhóm, không đưa kết quả giả ra demo.

## 3. API contract bắt buộc

```python
import numpy as np

class LaneDetector:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Load ONNX session 1 lần. Raise FileNotFoundError nếu thiếu weights."""

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return dict detector như bên dưới. Không sửa frame_bgr tại chỗ."""
```

```python
{
  "lanes": [
    {
      "lane_index": 0,
      "points": [[int_x0, int_y0], [int_x1, int_y1], ...],
      "confidence": None
    }
  ],
  "frame_width": int_W,
  "frame_height": int_H,
  "inference_ms": float_milliseconds
}
```

### Bất biến dữ liệu

- `frame_bgr.dtype == np.uint8`, shape `[H, W, 3]`, màu **BGR**.
- `points` theo `[x,y]`, pixel frame đầu vào **trước resize**. `x∈[0,W-1]`, `y∈[0,H-1]`, `y` tăng dần; bỏ điểm ngoài ảnh/invalid/duplicated y; lane phải ≥2 điểm.
- `lane_index` cấp lại 0,1,2,... theo thứ tự từ trái sang phải tại vùng giao nhau có thể quan sát (nếu có); không đồng nhất với `track_id`.
- `confidence` dùng `None` nếu output model không có confidence có thể diễn giải trực tiếp. Không tự gán 0.99.
- Dù không có vạch: output `{ "lanes": [], "frame_width": W, "frame_height": H, "inference_ms": ... }`.
- `inference_ms` đo thời gian preprocess + ONNX inference + decode cho **mỗi frame** (wall-clock) bằng `time.perf_counter()`.
- Không phụ thuộc import của `tracker/geometry/video`, để task chạy độc lập.

## 4. Thứ tự triển khai đề xuất

### Bước A — 0–10 phút: xác minh weights

1. Xác minh file ONNX tải được từ nguồn đã nêu. Kiểm tra kích thước file và đường dẫn. Tránh clone cả model zoo hoặc tải dataset lớn.
2. Tạo `ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])`.
3. In `input.name`, `input.shape`, `input.type`, danh sách `outputs` trên máy phát triển. Ghi lại vào `models/README.md`.
4. Kiểm tra output thực tế trên một ảnh thật. Nếu phiên bản dùng `NCHW` và kích thước cố định, bám đúng sơ đồ của repo inference; không suy đoán tensor shape.

### Bước B — 10–25 phút: inference adapter

1. Xây helper preprocess: BGR → RGB, resize/normalize/transpose phù hợp với đúng checkpoint. **Không thay preprocessing theo cảm tính**.
2. Gọi session `run(...)`, tái sử dụng session cho các frame.
3. Tái sử dụng logic giải mã UFLD từ repo tham chiếu, không thiết kế decoder khác rồi tự coi là đúng. Kiểm tra model có dùng softmax/argmax/grid anchors hay không.
4. Đổi tọa độ model về frame gốc: `x_orig = x_model * W / model_width`, `y_orig = y_model * H / model_height` **chỉ nếu tensor tọa độ đang ở hệ input model**. Nếu decoder đã trả tọa độ chuẩn hóa (0–1), phải dùng công thức phù hợp thay vì scale hai lần.
5. Bỏ điểm vô hiệu, sort theo `y`, cắt các lane có dưới 2 điểm; sắp xếp lane theo x tại cùng mốc tham chiếu có dữ liệu; gán `lane_index` tuần tự.

### Bước C — 25–35 phút: test ảnh thật

Trong `tests/test_detector.py`, chuẩn bị CLI nhỏ:

```bash
python tests/test_detector.py --model models/ufld_tusimple.onnx --image data/sample_frame.jpg
```

Test tự tạo một ảnh BGR 640×360 hoặc 1280×720; gọi detector trên frame đọc từ video hoặc ảnh mẫu. Kiểm tra:

- Giá trị output là `dict`; `frame_width/height` khớp frame.
- Tất cả điểm hợp lệ, y tăng dần và số điểm mỗi lane ≥2.
- Output không có `NaN`, không đổi kích thước frame input.
- In số lane, điểm lane, `inference_ms`; ảnh thực có thể không phát hiện được lane (được chấp nhận nhưng cần ghi rõ).
- Nếu có hình ảnh đường rõ, dùng OpenCV vẽ points để kiểm tra thủ công decoder không bị scale sai.

### Bước D — 35–60 phút: hoàn thiện tích hợp

- Kiểm tra 3 kích cỡ: 640×360, 1280×720, 640×480 (nếu có).
- Kiểm tra 3 frame liên tiếp có kết quả nhất quán về tọa độ.
- Dọn code, không sử dụng đường dẫn tuyệt đối theo máy Quang.
- Push branch và gửi cho Triệu một ví dụ output `detect(frame)` thực tế để ghép tracking.

## 5. Xử lý lỗi và hiệu năng

| Vấn đề | Cách xử lý |
|---|---|
| `model_path` không tồn tại | `FileNotFoundError` có đường dẫn & hướng dẫn tải |
| Invalid frame | `ValueError`, không tạo output giả |
| CPUExecutionProvider không hỗ trợ graph | Ghi lỗi gốc và version, kiểm tra đúng model/runtime |
| Output decoded sai vị trí | Kiểm tra scale/shape/anchors; đối chiếu script inference gốc |
| Model chạy chậm | Đo `inference_ms`; không tự ý bỏ frame trong detector (Triệu xử lý pipeline) |
| Không thấy vạch | `lanes=[]`, không vẽ vạch bịa |

Tối ưu nhỏ: ONNX session khởi tạo một lần, không convert frame qua PIL nếu không cần, tách logging/debug ngoài vòng inference. Không sử dụng GPU bắt buộc.

## 6. Giao tiếp với các thành viên

- **Việt Anh:** gửi một dict detector đúng contract và xác nhận `lane_index` chỉ là thứ tự hiện tại, ID tracking do Việt Anh tạo.
- **Diện:** xác nhận tọa độ thuộc ảnh gốc, điểm được sắp từ trên xuống, không tự quyết định camera center.
- **Triệu:** gửi tên model/path, dependency chính xác, ví dụ tạo `LaneDetector` và thời gian inference ước lượng đã đo trên phần cứng nhóm.
- **Thân:** gửi thông tin variant checkpoint và nơi đặt weights để README/benchmark không ghi nhầm model.

## 7. Checklist nghiệm thu

- [ ] Load thành công một ONNX pretrained thực, ghi nguồn và phiên bản.
- [ ] `LaneDetector(model_path).detect(frame)` khớp API contract 100%.
- [ ] Điểm lane trong tọa độ gốc, kiểm tra bằng ảnh overlay.
- [ ] Xử lý được `lanes=[]` mà không crash; không fake model output.
- [ ] Smoke test có lệnh chạy ngắn gọn.
- [ ] `models/README.md` mô tả nơi lấy weights & cách cài.
- [ ] Chỉ commit file sở hữu, branch `quang` đã push.

## 8. Git commands

```bash
git fetch origin
git switch main
git switch -c quang
# code...
git add src/detector.py tests/test_detector.py models/README.md
git commit -m "feat(detector): add UFLD ONNX inference adapter"
git push -u origin quang
```

Nếu branch đã tạo: `git switch quang`, không chạy lại `switch -c`. Không push main.

## 9. Prompt đưa trực tiếp cho Codex / Claude Code

> Bạn là engineer AI/CV, được giao **TASK_QUANG.md** trong dự án LaneGuard AI. Đọc **PROJECT_OVERVIEW.md mục 5.1–5.2** làm hợp đồng không thể thay đổi. Triển khai **chỉ** `src/detector.py`, `tests/test_detector.py`, `models/README.md`. Dùng UFLD pretrained ONNX qua ONNX Runtime CPU; không train/fine-tune và không giả lập kết quả model. Xác minh input/output ONNX bằng session, dùng preprocess và postprocess đúng checkpoint theo repo tham khảo, trả tọa độ `[x,y]` của frame BGR gốc. Session load 1 lần; xử lý lane rỗng và lỗi thiếu weights; có smoke test. Không sửa file người khác, không sửa API. Cuối cùng báo rõ file đã tạo, lệnh chạy, những gì đã test thật, lỗi tồn tại và nhánh `quang` để push. Nếu không tải được weights, vẫn viết adapter thật nhưng đánh dấu rõ chưa kiểm thử inference — không báo thành công giả.

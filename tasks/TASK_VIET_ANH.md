# TASK — VIỆT ANH | Lane Tracking & Temporal Smoothing

**Branch:** `viet-anh`  
**Vai trò:** Ghép các làn giữa nhiều frame, giữ ID ổn định, làm mượt và xử lý mất quan sát ngắn hạn.  
**Contract:** `PROJECT_OVERVIEW.md` mục **5.1 & 5.3**.  
**Deadline:** Unit test hoạt động trước phút 35; push branch trước phút 60.

## 1. Kết quả cần bàn giao

Xây dựng tracker hoạt động **không cần load model**. Input là kết quả Quang trả về (`dict` chứa `lanes`/`frame_width`/`frame_height`), output là `dict` lane có `track_id` duy trì qua các frame, điểm đã smoothing và cờ dự đoán. Không sử dụng DeepSORT/ByteTrack hay mô hình bổ sung: số lượng lane ít, matching đơn giản đã đủ cho MVP 2 giờ.

### File sở hữu

- `src/tracker.py`
- `tests/test_tracker.py`

Không chỉnh `src/detector.py`, `src/geometry.py`, `src/video.py`, `app.py`, `requirements.txt` và README chung. Nếu cần đề xuất thay đổi contract: thông báo trước, không tự ý sửa.

## 2. API contract — chính xác

```python
class LaneTracker:
    def __init__(self, alpha: float = 0.35, max_missed: int = 3) -> None: ...
    def update(self, detection: dict) -> dict: ...
    def reset(self) -> None: ...
```

Input mock từ detector:

```python
{
    "lanes": [
        {"lane_index": 0, "points": [[100, 100], [95, 200], [88, 300]], "confidence": None},
        {"lane_index": 1, "points": [[400, 100], [410, 200], [425, 300]], "confidence": None}
    ],
    "frame_width": 640,
    "frame_height": 360,
    "inference_ms": 20.0
}
```

Output (phải có đủ key):

```python
{
    "lanes": [
        {
            "track_id": 1,
            "points": [[100, 100], [95, 200], [88, 300]],
            "confidence": None,
            "predicted": False,
            "missed": 0
        }
    ],
    "frame_width": 640,
    "frame_height": 360,
    "tracking_ms": 0.5
}
```

- `track_id`: số nguyên dương, tăng đơn điệu trong một video, **không tái sử dụng** ID ngay sau khi track chết.
- `predicted=False` + `missed=0` khi có observation thật.
- `predicted=True` + `missed≥1` khi giữ track cũ không có observation mới.
- Khi `missed > max_missed`: xóa track khỏi output.
- `reset()`: xóa toàn bộ tracks và reset bộ đếm ID về 1 giữa hai video.
- `points`: pixel frame gốc, `[x,y]` theo thứ tự y tăng, không NaN/Inf, ≥2 điểm.
- `tracking_ms` đo bằng `time.perf_counter` mỗi lần `update`.
- `confidence` giữ nguyên nếu observation mới có; track dự đoán có thể giữ giá trị trước để tham khảo nhưng **không được coi là cập nhật tin cậy**; cờ `predicted`/`missed` mới là chính.

## 3. Thiết kế thuật toán đề xuất

### A. Chuẩn hóa hình dạng để so khớp

Không so trực tiếp từng điểm vì hai frame có thể có số keypoint khác nhau. Với mỗi lane, tạo một biểu diễn theo một nhóm mốc y chuẩn, ví dụ `y_ratios = [0.55, 0.65, 0.75, 0.85]`, lấy `y = round(ratio * (H-1))`:

1. Sort điểm theo y (detector đã đảm bảo nhưng vẫn có thể validate).
2. Nội suy `x` theo `y` bằng `np.interp` **chỉ trong vùng y quan sát được**; mốc nằm ngoài trả `None`/NaN (không ngoại suy).
3. Chỉ so hai lane khi có tối thiểu 2 mốc y chung hợp lệ.
4. Distance matching là mean `abs(x_old - x_new)/W` trên các mốc chung; dùng threshold khởi tạo ví dụ **0.08 chiều rộng ảnh**, phải test trên video mẫu.

### B. Data association

- Mỗi frame có tối đa khoảng 4 lane (theo model tham khảo) → có thể tạo tất cả candidate `(track, detection, distance)` rồi sort tăng và greedy 1–1.
- Chỉ nhận match nếu `distance <= match_threshold` (ví dụ 0.08, điều chỉnh thực nghiệm).
- Không gán track theo `lane_index`, cũng không mặc định “vạch trái không bao giờ chuyển thành vạch phải”; vị trí tương đối với xe thay đổi khi xe chuyển làn.
- Track chưa match tăng `missed` và `predicted=True`; detection chưa match tạo track mới.

### C. EMA smoothing

Để làm mượt lane matched: sử dụng mốc y chung/chuẩn ở phần giao nhau của 2 polyline, sau đó:

\[
 x_{smooth}(y) = \alpha x_{obs}(y) + (1-\alpha)x_{prev}(y)
\]

`alpha=0.35` → có ưu tiên ổn định; khi chuyển làn/góc cua mạnh có thể bị trễ. Chỉ tạo polyline điểm hữu hiệu trên miền y trùng nhau. Không dùng các mốc y không được quan sát để vẽ một đoạn lane bịa quá dài.

### D. Mất lane / occlusion

- Khi mất detection, bản MVP **giữ nguyên polyline gần nhất trong tối đa 3 frame** và đánh dấu `predicted=True`, tăng `missed`; đây là *hold-last observation*, **không** tuyên bố đã dự báo động học bằng Kalman.
- Khi có lane mới khớp lại, EMA cập nhật và `missed=0`.
- Sau 3 frame liên tiếp mất lane, xóa track; một lane xuất hiện sau đó có thể nhận ID mới.
- `Diện` sẽ áp dụng guard: `missed > 1` hoặc cả hai lane chỉ dự đoán → `UNKNOWN`.

### E. Biến thiên kích thước frame

Một video có kích thước không đổi. Nếu `frame_width`/`frame_height` đột ngột đổi, tốt nhất `reset()` tracker và khởi tạo lại, hoặc raise lỗi rõ ràng, không so sánh điểm khác hệ tọa độ.

## 4. Pseudocode triển khai

```python
def update(detection):
    validate_input(detection)
    start_timer()
    det_lanes = detection["lanes"]
    distances = build_candidate_distances(self.tracks, det_lanes)
    matched = greedy_one_to_one(distances, threshold=0.08)

    for track, obs in matched:
        track.points = smooth_overlap(track.points, obs.points, alpha=self.alpha)
        track.confidence = obs.get("confidence")
        track.missed = 0
        track.predicted = False

    for track in unmatched_tracks:
        track.missed += 1
        track.predicted = True
    remove_if_missed_greater_than(self.max_missed)

    for obs in unmatched_detections:
        create_new_track(obs)

    return contract_output_with_elapsed_ms()
```

Đây là pseudocode, không copy thành code chạy mà không viết helper/validate.

## 5. Tests bắt buộc bằng dữ liệu giả (không phụ thuộc Quang)

Tạo `tests/test_tracker.py`, có thể dùng `unittest` chuẩn để không cần pytest.

| Case | Input | Kỳ vọng |
|---|---|---|
| Khởi tạo | 2 lanes | 2 `track_id` khác nhau, `missed=0` |
| Nhiễu vị trí | Lane x dịch +3 pixel | Giữ ID, output ít dịch hơn obs khi EMA |
| Đảo thứ tự detection | `[A,B]` thành `[B,A]` | Không đổi ID của A/B |
| Mất một lane 1 frame | Lane A biến mất | Track A `predicted=True`, `missed=1` |
| Mất hết lanes > max_missed | `lanes=[]` lặp lại | Không còn track cũ |
| Lane mới xuất hiện | Một lane xa tracks hiện tại | Cấp ID mới |
| `reset` | Đang có 2 tracks rồi reset | Frame mới bắt đầu ID từ 1 |
| Edge cases | 0 lanes, 1 lane, invalid points | Không crash với 0/1; lỗi rõ khi input invalid |

Bảo đảm **không cần file ONNX hoặc video** để chạy unit test.

```bash
python -m unittest discover -s tests -p 'test_tracker.py' -v
```

## 6. Các mốc thời gian

- **0–10 phút:** đọc overview; tạo branch; viết class + tests stub + mock.
- **10–25 phút:** matcher 1–1, quản lý ID, `predicted/missed/reset`.
- **25–35 phút:** EMA, test đảo thứ tự và occlusion.
- **35–50 phút:** chỉnh edge cases; gửi ví dụ output cho Diện/Triệu.
- **50–60 phút:** chạy test, commit & push.

**Nếu thời gian ít:** matcher greedy + EMA + hold-last đủ, bỏ Kalman và đặc trưng hình dạng nâng cao.

## 7. Giao tiếp module

- Quang cung cấp **dict** điểm lane như overview; tracker không cần biết ONNX hay tên checkpoint.
- Diện nhận track đã smoothing, có `track_id`,`predicted`,`missed`; phải thống nhất `UNKNOWN` nếu chỉ có track dự đoán.
- Triệu tạo **một** LaneTracker mỗi video, gọi `update()` theo thứ tự frame, reset giữa video.
- Thân đo tỷ lệ continuity bằng video, không tự suy đoán ID đúng khi chưa có nhãn.

## 8. Checklist bàn giao

- [ ] File và API chuẩn, không import model/deep learning.
- [ ] Matching đúng dù lane_index đổi thứ tự.
- [ ] EMA cho track matched, dữ liệu vẫn ở pixel frame gốc.
- [ ] Dừng dự đoán sau `max_missed`, cờ predicted chính xác.
- [ ] `reset()` hoạt động, test chỉ dùng mock.
- [ ] Không sửa file thuộc người khác.
- [ ] Push lên branch `viet-anh`, gửi commit SHA cho nhóm.

## 9. Git

```bash
git switch main
git switch -c viet-anh
# code và test...
git add src/tracker.py tests/test_tracker.py
git commit -m "feat(tracking): add temporal lane association and EMA"
git push -u origin viet-anh
```

Nếu branch đã tồn tại, `git switch viet-anh`. Không push main.

## 10. Prompt cho Codex / Claude Code

> Đọc **PROJECT_OVERVIEW.md mục 5.1, 5.3** và toàn bộ **TASK_VIET_ANH.md**. Bạn chịu trách nhiệm duy nhất cho `src/tracker.py` và `tests/test_tracker.py`. Xây LaneTracker với `__init__(alpha=0.35,max_missed=3)`, `update(detection)->dict`, `reset()`. Input từ detector là dict các polylines pixel frame gốc; output track_id, points đã EMA, confidence, predicted, missed, frame_width/height, tracking_ms. Match 1–1 theo khoảng cách hình học trên các mốc y có dữ liệu, không dùng lane_index làm ID. Khi mất detection giữ vị trí cũ tối đa 3 frame và đánh dấu predicted, không gọi đó là Kalman. Viết unit tests hoàn toàn synthetic cho mất lane, đổi thứ tự, smoothing, reset; không cần weights. Không đổi contract, không sửa file thành viên khác, không thêm framework nặng. Báo lệnh test, file thay đổi, kết quả test thực tế, hạn chế và nhánh `viet-anh`.

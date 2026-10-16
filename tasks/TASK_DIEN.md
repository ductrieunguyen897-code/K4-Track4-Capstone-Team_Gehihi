# TASK — DIỆN | Lane Geometry, Offset & Warning Rules

**Branch:** `dien`  
**Vai trò:** Từ các làn được tracking, tính 2 ranh giới làn xe đang đi, tâm làn, độ lệch tương đối và cảnh báo minh họa.  
**Contract:** `PROJECT_OVERVIEW.md` mục **5.1 & 5.4**.  
**Mốc:** Code với dữ liệu giả hoạt động trước phút 35, push trước phút 60.

## 1. Mục tiêu bàn giao

Một hàm `analyze_lanes(tracking, camera_center_ratio=0.5)` thuần Python/NumPy; không cần video, ONNX, Gradio hoặc Torch. Output theo hợp đồng, xử lý tốt 0/1/2/4 lane, hình học sai và trường hợp `predicted`. Đây là phần quyết định **lúc nào có đủ bằng chứng để hiển thị offset**.

### File Diện sở hữu

- `src/geometry.py`
- `tests/test_geometry.py`

Không sửa code/requirement của các thành viên khác. Không tự ý thêm chức năng đòi hiệu chuẩn camera thật nếu không có dữ liệu.

## 2. API contract bắt buộc

```python
def analyze_lanes(tracking: dict, camera_center_ratio: float = 0.5) -> dict:
    """Analyze the currently observed ego-lane in image coordinates."""
```

**Input** (từ Việt Anh):

```python
{
    "lanes": [
        {"track_id": 1, "points": [[100, 160], [95, 250], [80, 350]],
         "confidence": None, "predicted": False, "missed": 0},
        {"track_id": 2, "points": [[530, 160], [545, 250], [560, 350]],
         "confidence": None, "predicted": False, "missed": 0}
    ],
    "frame_width": 640,
    "frame_height": 420,
    "tracking_ms": 0.8
}
```

**Output** (bắt buộc có đủ các field ngay cả khi UNKNOWN):

```python
{
    "valid_lane": bool,
    "reference_y": int,
    "vehicle_center_x": float,
    "lane_center_x": float | None,
    "left_track_id": int | None,
    "right_track_id": int | None,
    "offset_normalized": float | None,
    "warning": "SAFE" | "CAUTION" | "WARNING" | "UNKNOWN",
    "curve": "STRAIGHT" | "CURVED" | "UNKNOWN",
    "lane_type_left": "SOLID" | "DASHED" | "UNKNOWN",
    "lane_type_right": "SOLID" | "DASHED" | "UNKNOWN"
}
```

Để nhất quán, `reference_y = round(0.85 * (H - 1))` (hệ pixel ảnh), `vehicle_center_x = camera_center_ratio * (W - 1)`. Nếu `camera_center_ratio` không trong `[0,1]`, raise `ValueError`.

## 3. Thuật toán P0 — lựa chọn 2 vạch của làn đang đi

### Bước A — Xác định tọa độ tại một hàng ảnh chung

Với từng lane `points` sort theo y; chỉ chấp nhận lane có `min(y) <= reference_y <= max(y)`; tính `x_at_ref` bằng nội suy tuyến tính `np.interp(ref_y, ys, xs)`. Không ngoại suy ngoài miền quan sát; nếu trùng y, loại duplicate trước nội suy.

### Bước B — Chọn hai ranh giới kẹp trục camera

- `xV = camera_center_ratio * (W-1)`.
- Tập lane ở **bên trái**: `x_at_ref < xV`; lấy lane có x lớn nhất (gần trục camera nhất từ trái).
- Tập lane ở **bên phải**: `x_at_ref > xV`; lấy lane có x nhỏ nhất (gần trục camera nhất từ phải).
- Không chọn mặc định 2 vạch ngoài cùng (vì ảnh có tối đa 4 vạch). Nếu thiếu một phía => UNKNOWN.
- `lane_width_px = xR-xL`; từ chối nếu `lane_width_px <= 0` hoặc nhỏ vô lý (ví dụ `< 0.06*W`); đây chỉ là heuristic kiểm tra hình học. Có thể thêm giới hạn trên hợp lý bằng ratio nhưng không dùng để gạt bỏ tất cả trường hợp camera góc rộng.

### Bước C — Quy tắc thông tin đủ/không đủ

Nếu `missed > 1` ở một trong hai track chọn, hoặc **cả hai `predicted=True`**, trả UNKNOWN. Nếu thiếu field predicted/missed, hợp đồng đã yêu cầu có: báo ValueError rõ thay vì im lặng coi là lane quan sát. Một track có `missed==1` được phép dùng tạm nếu track còn lại có quan sát thật. Khi không có frame nào quan sát thực, **không thể trả SAFE**.

### Bước D — Tính offset bằng pixel, không quy đổi mét

\[
 x_C = (x_L+x_R)/2,
 \qquad e = \frac{x_V-x_C}{x_R-x_L}
\]

- `e > 0`: camera/xe nằm bên **phải** tâm làn.
- `e < 0`: camera/xe nằm bên **trái** tâm làn.
- `abs(e) < 0.15`: `SAFE` (ngưỡng demo).
- `0.15 <= abs(e) < 0.25`: `CAUTION` (ngưỡng demo).
- `abs(e) >= 0.25`: `WARNING` (ngưỡng demo).
- Khi invalid: `valid_lane=False`, `offset_normalized=None`, `lane_center_x=None`, `warning='UNKNOWN'`; `left_track_id/right_track_id` đặt `None` nếu không có cặp lane hợp lệ.

**Không gọi đây là time-to-lane-crossing hay cảnh báo lệch làn đã kiểm nghiệm**; từ một frame/không camera calibration không thể suy ra chính xác chuyển động ngang/quan hệ với mép bánh xe. Cũng không nên hiện "SAFE" theo nghĩa an toàn lái xe; trong UI có thể ghi "Centered (demo)".

## 4. Curve classification — P1 tùy chọn

Khi P0 chạy ổn, fit `x=f(y)` cho mỗi boundary/cặp centerline bằng `np.polyfit(y, x, deg=2)` nếu có đủ >=3 điểm trên miền y đáng kể và không ill-conditioned. Đo độ phi tuyến so với fit bậc nhất (ví dụ độ lệch trung bình chuẩn hóa theo W) trên miền y chung. Nếu dưới ngưỡng demo thì `STRAIGHT`, cao thì `CURVED`; không đủ dữ liệu/đường polyline ngắn → `UNKNOWN`.

**Không dùng curvature tính trong pixel để báo bán kính cong bằng mét.** Trong 2 tiếng có thể để `curve='UNKNOWN'` nếu chưa đủ bằng chứng/độ ổn định.

## 5. Solid/dashed classification — P1 sau curve (nếu kịp)

Việc phân biệt SOLID/DASHED từ chỉ các **điểm đường được UFLD ước lượng** là không đáng tin: model có thể trả điểm đều dọc vạch đứt, khiến chúng trông như vạch liền. Nếu thực hiện:

1. Cần frame ảnh gốc để đo thay đổi độ sáng dọc vạch; **API `analyze_lanes` hiện không nhận frame**, vì vậy **không được âm thầm thay chữ ký**.
2. Đối với MVP, mặc định `lane_type_left/right='UNKNOWN'`. Muốn phân loại thật, đề xuất interface bổ sung riêng **sau khi cả nhóm chấp thuận**, chẳng hạn helper `classify_lane_markings(frame_bgr, tracking)` ở file geometry; pipeline của Triệu phải gọi riêng nếu rảnh.
3. Không tính là hoàn thành nếu chỉ dựa vào số điểm UFLD để phán SOLID/DASHED.

## 6. Unit test bằng mock, không chờ AI

Tạo `tests/test_geometry.py` dùng `unittest`, tự sinh đường cong/đường thẳng trên ảnh 640×480. **Bắt buộc** kiểm tra:

| Test | Tình huống | Kỳ vọng |
|---|---|---|
| 2 lane cân xứng | L/R cách đều `xV` | `offset≈0`, `SAFE` |
| Xe lệch phải | `xV` ở phía phải tâm làn | offset dương |
| Xe lệch trái | `xV` ở phía trái tâm làn | offset âm |
| Vạch thứ 3, thứ 4 | Nhiều lane trên ảnh | Chọn 2 lane gần nhất **kẹp trục xe** |
| Chỉ một lane | Missing right | UNKNOWN và `offset_normalized is None` |
| Không lane | Empty input | UNKNOWN, không throw |
| Hai lane cũ | cả hai `predicted=True` | UNKNOWN |
| Một lane mất lâu | `missed=2` | UNKNOWN |
| Video khác width | 1280×720 | Tính dựa theo W/H được cung cấp |
| Invalid geometry | xR≤xL / lane quá hẹp | UNKNOWN |
| Invalid `camera_center_ratio` | -0.1 hoặc 1.2 | `ValueError` |

```bash
python -m unittest discover -s tests -p 'test_geometry.py' -v
```

Tránh lỗi kiểm thử: các điểm test phải thực sự **bao phủ `reference_y`**. Nếu `H=480`, `reference_y=407`, points y cuối phải ít nhất 407. Có thể dùng `y=[250,350,430,470]`.

## 7. Phân bổ thời gian đề xuất

- **Phút 0–10:** đọc overview; tạo nhánh và helper interpolate.
- **10–25:** logic chọn lane trái/phải, guard invalid, offset, warning.
- **25–35:** hoàn thành 10 tests synthetic; gửi contract mẫu cho Triệu.
- **35–50:** kiểm tra data thực từ Việt Anh khi có; chỉnh threshold hợp lý nhưng giữ công thức/sign.
- **50–60:** commit, push branch, báo kết quả.

Nếu không còn thời gian: **bỏ đường cong và liền/đứt**, ưu tiên không báo SAFE khi không thấy đủ hai lane.

## 8. Phụ thuộc và giao tiếp

- Nhận đầu vào của **Việt Anh**, không đọc trực tiếp output thô UFLD của Quang.
- **Triệu** gọi `analyze_lanes(tracking, camera_center_ratio)` mỗi frame, hiển thị đúng `UNKNOWN` khi `valid_lane=False` và xuất CSV.
- **Thân** có thể dùng output warning để benchmark tỷ lệ khung hình có đủ dữ liệu; không biến tỷ lệ này thành accuracy.
- Khi cần frame ảnh để phân loại SOLID/DASHED, thảo luận helper **riêng** trước khi thực hiện; không đổi API P0.

## 9. Checklist & Git

- [ ] Signature và tất cả output key chuẩn.
- [ ] Chọn vạch kẹp camera từ tối đa 4 đường.
- [ ] Offset sign đúng, phân ngưỡng đầy đủ.
- [ ] Empty/missing/stale/invalid → UNKNOWN.
- [ ] Không đưa khoảng cách mét, không nói cảnh báo an toàn thực tế.
- [ ] Test synthetic chạy pass.
- [ ] Chỉ thay đổi 2 file sở hữu; push `dien`.

```bash
git switch main
git switch -c dien
# coding và test...
git add src/geometry.py tests/test_geometry.py
git commit -m "feat(geometry): compute ego-lane offset and warnings"
git push -u origin dien
```

## 10. Prompt đưa cho Codex / Claude Code

> Đọc **PROJECT_OVERVIEW.md mục 5.1, 5.4** và **TASK_DIEN.md**. Chỉ viết `src/geometry.py`, `tests/test_geometry.py`. Triển khai hàm thuần `analyze_lanes(tracking:dict,camera_center_ratio:float=0.5)->dict`, chọn lane gần nhất mỗi bên kẹp vị trí camera tại `reference_y=round(0.85*(H-1))`, nội suy chỉ trên miền có dữ liệu, tính offset chuẩn hóa dấu dương lệch phải, warning theo các ngưỡng demo. Nếu thiếu 2 lane, không có observation thật, hoặc predicted quá cũ thì UNKNOWN + offset None. Trả đủ field `valid_lane,reference_y,vehicle_center_x,lane_center_x,left_track_id,right_track_id,offset_normalized,warning,curve,lane_type_left,lane_type_right`. `curve` và liền/đứt có thể UNKNOWN; không thay chữ ký để thêm ảnh. Viết unit test đầy đủ với 0/1/4 lanes, sign, predicted, no extrapolation. Không dùng training, không sửa file người khác. Báo kết quả test thật, mọi giới hạn và cách push branch `dien`.

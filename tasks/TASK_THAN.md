# TASK — THÂN | Video Samples, Benchmark, QA & Hackathon Demo

**Branch:** `than`  
**Vai trò:** Chuẩn bị dữ liệu nhỏ, kiểm thử định lượng/trình diễn, tài liệu người dùng và pitch. Không phụ trách model inference, tracker, geometry hoặc UI.  
**Contract:** `PROJECT_OVERVIEW.md` mục **5.1, 5.5, 5.6**.  
**Mốc:** Có clip mẫu + kế hoạch benchmark trước phút 35, push module kiểm thử/tài liệu trước phút 60; phối hợp demo từ phút 80.

## 1. Kết quả bàn giao

Chuẩn bị **tối đa 3 clip ngắn** (5–20 giây) đường có vạch rõ để cả nhóm test; tạo script benchmark gọi **pipeline chính** của Triệu, không xây inference/tracker thứ hai; ghi nhận số liệu đo thực; hoàn thiện README, nội dung slide và script demo 60–90 giây. Nếu chưa tích hợp được ở phút 60, phần script/tài liệu vẫn phải có thể dùng bằng mock test rõ ràng, không ghi benchmark giả.

### Các file Thân sở hữu

- `benchmark.py`
- `README.md`
- `data/README.md`
- `docs/presentation.md`
- `tests/smoke_test.py`

**Dữ liệu/artefact runtime:** `data/*.mp4`, `results/*`, `outputs/*` (video lớn/weights **không commit**). Tránh sửa `requirements.txt`, `src/*`, `app.py`, `.gitignore` do Triệu/đồng đội sở hữu.

## 2. Dữ liệu mẫu — tối thiểu và kiểm soát quyền sử dụng

**Nguồn khuyến nghị:** https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_videos (kiểm tra clip đang có và điều kiện giấy phép phù hợp trước khi tái phân phối).

Đề xuất ba case:

| Clip | Phân loại | Dùng kiểm tra |
|---|---|---|
| A — highway simple | Đường thẳng/vạch rõ | ONNX detection, tâm làn, render |
| B — yellow/white lines | Hai màu/độ tương phản khác | Chuẩn hóa output, tracking |
| C — challenging/curve | Đường hơi cong hoặc bóng đổ | UNKNOWN, cảnh báo, thất bại có kiểm soát |

Ví dụ tên các video trong các bản Udacity phổ biến: `solidWhiteRight.mp4`, `solidYellowLeft.mp4`, `challenge.mp4`. **Kiểm tra danh sách thực tế trên repo trước khi viết downloader; không giả định link raw chắc chắn tồn tại.** Nếu không tải được trong 5–10 phút, dùng video ngắn có sẵn trên laptop với quyền sử dụng rõ ràng, thông báo team nguồn và hạn chế.

### Chuẩn hóa video (nếu cần)

- Dùng FFmpeg (nếu có sẵn) để cắt **5–15 giây**, MP4/H.264/yuv420p, 640×360 hoặc 1280×720 (không bắt buộc nếu nguồn tương thích).
- Tên local: `data/sample_highway.mp4`, `data/sample_lines.mp4`, `data/sample_challenge.mp4`.
- Ghi nguồn, license, thời lượng, W×H, FPS, dung lượng vào `data/README.md`.
- **Không push video lớn** hoặc file chứa dữ liệu cá nhân/biển số chưa được phép chia sẻ. Máy tích hợp tải/copy video local theo README, không yêu cầu các nhánh có video.

Nếu thử dữ liệu đường đô thị Việt Nam: nên chọn cảnh hợp pháp/có quyền sử dụng; xem là **stress test bổ sung**, không được tuyên bố sản phẩm đã xử lý tốt mọi đường đô thị Việt Nam từ 1–2 clip.

## 3. `benchmark.py` — hợp đồng và cách chạy

Viết CLI bằng `argparse`:

```bash
python benchmark.py \
  --input data/sample_highway.mp4 \
  --model models/ufld_tusimple.onnx \
  --max-frames 300 \
  --output results/benchmark_sample
```

`--max-frames` mặc định 300; `--output` là tiền tố để lưu file đo/biểu đồ/bản ghi, không bắt buộc cho `process_video`. Lệnh có thể điều chỉnh cho Windows PowerShell thành một dòng.

### Không sao chép pipeline!

```python
from src.video import process_video

result = process_video(
    input_path=args.input,
    output_path="outputs/benchmark_processed.mp4",  # chọn tên unique theo clip
    model_path=args.model,
    camera_center_ratio=0.5,
    max_frames=args.max_frames,
)
summary = result["summary"]
# read result["csv_path"] và ghi metrics thực tế
```

Vì `src/video.py` chưa tồn tại trên nhánh `than` trong lúc phát triển song song, **import bên trong `main()` hoặc trong hàm chạy benchmark**, để `python benchmark.py --help` hoạt động độc lập. Có thể unit test parse CSV bằng một `sample.csv` tạo trong memory/temp folder; không commit `src/video.py` bản giả.

### Output mong muốn từ benchmark script

- In terminal: số frame, `video_fps`, `processing_fps`, `processing_seconds`, `avg_inference_ms`, `valid_lane_frames`, `% valid`, `warning_frames`, `failed_frames`.
- In clip resolution: đọc bằng `cv2.VideoCapture` hoặc metadata đã lưu; in thông số **CPU/OS/Python** (ví dụ `platform.platform()`, `platform.processor()` có thể không đầy đủ; không phán đoán CPU từ chuỗi rỗng).
- Xuất JSON hoặc CSV tổng hợp `results/benchmark_summary.csv` nhỏ và Markdown table `results/benchmark_report.md` nếu cần (báo cáo này chỉ sinh sau đo thật).
- Nếu có thời gian, biểu đồ hai thông số: `detector_ms` theo frame và tỷ lệ lane valid/invalid. Matplotlib có thể được dùng, không buộc phải có plot nếu làm chậm demo.
- Thời gian chạy tính wall-clock từ `process_video`; không gọi source `video_fps` là performance.
- Không tạo accuracy/precision/recall giả khi chưa có ground-truth.

### Metrics và cách diễn giải

| Metric | Cách tính | Cảnh báo diễn giải |
|---|---|---|
| Processing FPS | `frames_processed / processing_seconds` | Là throughput của máy đã test, không phải video fps |
| Mean inference latency | `mean(detector_ms)` từ CSV | Không bao gồm mọi bước nếu detector đo một phần; xem overview |
| Valid-lane ratio | `valid_lane_frames / frames_processed` | Tỷ lệ có dữ liệu hình học, **không phải accuracy** |
| WARNING ratio | `warning_frames / frames_processed` | Số frame bị rule kích hoạt, **không phải ground truth lỗi lái** |
| Tracker smoothness | Độ dao động x tại một mốc y qua frame | Có thể giảm do quá mượt dù dự đoán sai; ghi hạn chế |
| Manual check | 10 frame/clip × 3 clips (nếu kịp) | Chỉ là kiểm tra định tính/thủ công quy mô nhỏ |

Nếu đo trên 1 clip, ghi rõ một clip, không tuyên bố "benchmark toàn diện". Nếu tốc độ dưới 10 FPS, nêu số thật cùng CPU/độ phân giải và phương án tối ưu sau hackathon.

## 4. QA checklist (phải gửi nhóm từ sớm)

### Functional test

- [ ] UFLD weights tồn tại, load một lần, không sử dụng dummy AI trong sản phẩm cuối.
- [ ] Upload 1 video thật trong Gradio, kết quả xuất MP4.
- [ ] Browser phát được output MP4 (không chỉ `VideoCapture` đọc được).
- [ ] Tracker có ID ổn định tương đối giữa các frame đơn giản.
- [ ] Có vạch 2 bên: hiển thị center marker, normalized offset.
- [ ] Không đủ hai vạch: cảnh báo `UNKNOWN`, **không** SAFE giả.
- [ ] Khi model lỗi, app thông báo lỗi chứ không báo hoàn thành giả.
- [ ] CSV có dữ liệu đủ cột; NaN/None ghi trống đúng format.
- [ ] FPS được đo trên phần cứng thực tế; chỉ đưa số lên slide nếu đo được.

### UX test

- [ ] Tên sản phẩm dễ đọc, luồng upload → analyze → kết quả không quá 3 thao tác.
- [ ] Có một video mẫu để demo nhanh.
- [ ] Overlay đủ rõ trên màn hình 1366×768, không che hết đường.
- [ ] Output có cảnh báo khác màu và giải thích "demo only".
- [ ] Nếu người xem upload file lạ/không đọc được, hiển thị message dễ hiểu.

### Safety / honesty

- [ ] Không quảng cáo sản phẩm đạt tiêu chuẩn ADAS/lái xe thực tế.
- [ ] Không báo độ lệch bằng mét/độ cong vật lý khi không calibration.
- [ ] Không gọi tỷ lệ valid lane là F1/accuracy.
- [ ] Trong README nêu giới hạn đường đô thị Việt Nam/ánh sáng kém/làn phức tạp.

## 5. Smoke test độc lập của Thân

File `tests/smoke_test.py` nên có hai chế độ:

1. `--check-files`: kiểm tra các tệp thiết yếu tồn tại khi **đã tích hợp**: `app.py`, `src/detector.py`, `src/tracker.py`, `src/geometry.py`, `src/video.py`, `src/visualizer.py`, `models/ufld_tusimple.onnx`.
2. `--check-output outputs/...`: đọc video đầu ra bằng cv2, đếm frame, kiểm tra CSV tương ứng có hàng dữ liệu và key `warning`,`offset_normalized`. **Không bắt buộc tích hợp khi chỉ chạy `--help` hoặc `--check-output`.**

Không coi thiếu file từ nhánh đồng đội trong giai đoạn song song là lỗi của Thân. Sau khi integration-local hoàn thành, dùng checklist này để test tổng.

## 6. README.md — khung bắt buộc

Tạo `README.md` đủ để giám khảo clone **sau khi đã merge các nhánh**, nội dung:

1. **LaneGuard AI:** problem statement 3–5 câu, demo chỉ dùng video ghi sẵn, không điều khiển xe.
2. **Features P0:** pretrained lane detection, temporal tracking, offset, cảnh báo dạng rule, overlay, CSV.
3. **Architecture:** sơ đồ text `Video → UFLD ONNX → Tracker → Geometry → Overlay/UI`.
4. **Quick start:** Python 3.10+, `python -m venv .venv`, activate (Windows/Linux), `pip install -r requirements.txt`, tải weights vào `models/ufld_tusimple.onnx` (URL ở `models/README.md`), `python app.py`.
5. **Model/data sources & license:** dẫn repo gốc và video test; **không sao chép hoặc nêu license weights chưa xác minh**.
6. **API contracts:** liên kết `PROJECT_OVERVIEW.md` làm nguồn chân lý.
7. **Benchmarks:** chỉ điền bảng đo thực, nếu chưa có ghi `Not measured`.
8. **Limitations:** camera chưa calibration, offset chỉ normalized, mưa/vạch mờ/đô thị hỗn hợp có thể lỗi, cảnh báo demo.
9. **Team/branch:** Quang `quang`, Việt Anh `viet-anh`, Diện `dien`, Triệu `trieu`, Thân `than`.

**Quan trọng:** Bởi `PROJECT_OVERVIEW.md` và các `TASK_*.md` là tài liệu gửi riêng cho nhóm, README nguồn có thể tham chiếu chúng nếu đã copy vào repo. Nếu không copy, README phải tự đủ hướng dẫn chạy.

## 7. `docs/presentation.md` — pitch 3–5 slides (nội dung văn bản)

Tạo nội dung slide theo cấu trúc này:

1. **Problem:** Nhiều dữ liệu video; khó theo dõi độ ổn định vạch/cảnh báo sai lệch khi kiểm tra thủ công; giới hạn phạm vi prototype.
2. **Our solution:** Web upload MP4 → UFLD pretrained + OpenCV tracking/geometry → video overlay + CSV.
3. **Technology/demo:** ONNX Runtime CPU, Gradio, bộ tracking EMA, normalized offset và UNKNOWN guard.
4. **Results:** 1–3 clip output; *chỉ số FPS/latency và máy chạy đã đo thật*; ví dụ thất bại/UNKNOWN nếu có.
5. **Limitations & future work:** F1 trên bộ gán nhãn, hiệu chuẩn camera, bộ kiểm thử đường đô thị Việt Nam, nâng cấp dashed/solid, nhập/tách làn.

Không đưa số 95%, 15 FPS, 99% hoặc giảm cảnh báo 80% lên slide khi chưa có log benchmark đủ chứng cứ.

### Kịch bản demo 60–90 giây

- 0–10s: giới thiệu "LaneGuard AI — phân tích vạch làn trên dashcam video".
- 10–20s: upload video 5–10 giây; bấm Analyze.
- 20–45s: chiếu overlay vạch, ID, center marker, warning/UNKNOWN.
- 45–60s: xem metrics CSV/FPS thực.
- 60–75s: nêu hiệu năng đo và giới hạn an toàn.
- 75–90s: hướng phát triển tương lai.

Luôn có clip output offline/screen capture dự phòng nếu inference trong buổi trình diễn chậm.

## 8. Timeline công việc của Thân

| Phút | Việc cần làm | Đầu ra |
|---|---|---|
| 0–10 | Tạo branch, đọc overview; chốt format metric | Kế hoạch QA |
| 10–25 | Tìm/tải/cắt 2–3 video mẫu ngắn | Clip local + `data/README.md` |
| 25–35 | Viết `benchmark.py --help`, khung QA và README | Script không cần module thật khi import |
| 35–50 | Viết smoke test và slide outline | Test/report skeleton |
| 50–60 | Commit push branch `than` | Code và docs sẵn merge |
| 60–80 | Hỗ trợ Triệu tích hợp/test output | Danh sách blockers & clip chất lượng |
| 80–100 | Chạy benchmark thật trên máy tích hợp | FPS, latency, CSV, tỷ lệ valid |
| 100–115 | Quay/chốt video demo và slide | Assets hoàn thành |
| 115–120 | Dry-run pitch | Demo 60–90 giây |

Nếu chỉ đủ thời gian chuẩn bị **1 video mẫu tốt** và benchmark đúng, ưu tiên hơn số lượng video lớn không được kiểm chứng.

## 9. Giao tiếp phụ thuộc

- **Quang:** đề nghị link ONNX chính xác, input shape/độ lớn model và tình trạng chạy trên CPU.
- **Việt Anh:** khi đo giảm rung, xin định nghĩa `track_id`,`missed`, không đo continuity từ ID nếu chưa kiểm tra visual.
- **Diện:** giải thích sign offset, guard UNKNOWN, demo thresholds chỉ là heuristic.
- **Triệu:** chốt return schema `process_video`, địa chỉ output CSV/video, cách đọc metrics và cấu hình demo/browser.

## 10. Checklist bàn giao & Git

- [ ] Có ít nhất 1–3 video mẫu *local* ngắn, ghi nguồn trong `data/README.md`.
- [ ] `benchmark.py --help` chạy mà không cần module Triệu.
- [ ] `benchmark.py` gọi chung `process_video`, không nhân bản model.
- [ ] README có quickstart + dependency + model instructions + giới hạn.
- [ ] Smoke test kiểm tra thành phẩm sau khi tích hợp.
- [ ] Pitch 3–5 slides trong Markdown và demo script.
- [ ] Tất cả metrics trên slide có log chứng minh hoặc ghi chưa đo.
- [ ] Chỉ commit file sở hữu, branch `than` đã push.

```bash
git switch main
git switch -c than
# coding + docs...
git add benchmark.py README.md data/README.md docs/presentation.md tests/smoke_test.py
git commit -m "feat(benchmark): add smoke tests, docs and hackathon demo"
git push -u origin than
```

## 11. Prompt dùng trực tiếp với Codex / Claude Code

> Đọc **PROJECT_OVERVIEW.md mục 5.1, 5.5, 5.6** và **TASK_THAN.md**. Chỉ tạo `benchmark.py`, `README.md`, `data/README.md`, `docs/presentation.md`, `tests/smoke_test.py`. Trách nhiệm: chuẩn bị hướng dẫn 1–3 clip video ngắn có nguồn rõ, benchmark **sử dụng trực tiếp** `src.video.process_video` với CLI `--input --model --max-frames --output`, ghi FPS wall-clock, latency, valid lane ratio, WARNING frames, CPU/OS vào report nhỏ; tuyệt đối không làm giả benchmark và không gọi valid ratio là accuracy. README hướng dẫn setup Gradio/ONNX và giới hạn demo; slides 3–5 trang và kịch bản demo dưới 90 giây; smoke test kiểm tra output MP4/CSV. Vì module Triệu chưa có trên branch `than`, import bên trong hàm main và test bằng file giả, không commit module AI/UI giả. Không sửa file sở hữu thành viên khác, không tải dataset lớn vào Git. Trả file thay đổi, lệnh chạy, những gì đã đo thật/chưa đo, và lệnh push branch `than`.

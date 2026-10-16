# UFLD Pretrained ONNX Model Weights (`models/ufld_tusimple.onnx`)

## UFLDv2 CULane option

The app also supports the author's pretrained UFLDv2 CULane ResNet18 model.
It is inference-only and optional; the existing v1 model remains supported.
When `models/ufldv2_culane_res18.onnx` exists, the UI selects it by default.
The current measurements and remaining limits are in `results/lane_v2_validation.md`.

```powershell
# Export dependencies are optional and are not needed to run an existing ONNX.
.venv/Scripts/python.exe -m pip install torch torchvision onnx
.venv/Scripts/python.exe scripts/prepare_ufldv2.py
```

The script downloads the official checkpoint (~825 MB) to ignored `outputs/`,
checks SHA-256, loads every parameter strictly and exports a self-contained
ONNX file (~825 MB). The download is not a training dataset. Neither weights
nor videos are committed. To reuse a checkpoint, pass `--checkpoint PATH`.
The v2 network uses RGB, ImageNet normalization, resize to 1600x533 and the
bottom 320 rows, with four outputs: `loc_row`, `loc_col`, `exist_row`, `exist_col`.
Decoded anchors map to the original ROI, including the training crop.

Verified official checkpoint SHA-256:
`956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967`.
ONNX serialization checksums vary with export software; the script prints its
checksum and validates the graph. It refuses to replace a model unless `--force` is set.

Sources: [author's model and license](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2),
[CULane configuration](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/configs/culane_res18.py),
[reference decoding](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/demo.py).

### Optional NVIDIA runtime

CPU execution remains available. For compatible CUDA 12 / cuDNN 9 installations:

```powershell
.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
.venv/Scripts/python.exe -m pip install -r requirements-gpu.txt
```

The tested environment uses PyTorch 2.7.1+cu126 to supply CUDA DLLs, ONNX Runtime
GPU 1.21.1 and a GTX 1650. `auto` uses CUDA when available, with CPU fallback;
an explicit `cuda` request fails if CUDA cannot initialize. The summary reports
the actual provider. Do not install CPU and GPU ONNX Runtime distributions
together because they provide the same Python module.
See [ONNX Runtime's CUDA setup](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

An optional FP16 export (~413 MB) can be produced with `--fp16` after installing
`onnxconverter-common`. It keeps float32 input/output and needs GPU execution;
it is not automatically selected. Compare its outputs on your footage before use.

> **Phụ trách:** Quang (Branch `quang`)  
> **Chính sách Git:** File trọng số `*.onnx` / `*.pth` có dung lượng lớn (~233.5 MB) nên **không commit lên Git**. Đặt file đúng tên chuẩn `models/ufld_tusimple.onnx` trước khi chạy ứng dụng hoặc benchmark.

---

## 1. Thông tin mô hình & Nguồn gốc (Provenance)

- **Kiến trúc:** Ultra-Fast-Lane-Detection (UFLD) — *Ultra Fast Structure-aware Deep Lane Detection* (ECCV 2020)
- **Backbone:** ResNet-18 (`tusimple_18`)
- **Dataset huấn luyện gốc:** TuSimple Highway Lane Detection
- **Định dạng suy luận:** ONNX (`.onnx`), chạy trên `onnxruntime` (`CPUExecutionProvider`)
- **Đường dẫn runtime bắt buộc:** `models/ufld_tusimple.onnx`
- **Kích thước file (đã xác minh trên máy tích hợp):** `244,894,647` bytes (~`233.55 MB`)
- **SHA-256 (đã xác minh trên máy tích hợp):** `9d81165035d489ee4fc087933bf736645f7fca3666c388fc87bd9db24771e4f7`

> **Lưu ý về checksum (cập nhật khi tích hợp):** checksum/size ở trên là giá trị **đo thực tế** từ file ONNX được xuất bằng đúng script bên dưới trên máy tích hợp. Việc xuất ONNX **không đảm bảo byte-for-byte giữa các phiên bản `torch`/`onnx`** (metadata graph, producer/version string khác nhau), nên SHA-256 có thể lệch dù kiến trúc và trọng số giống nhau. Tiêu chí nghiệm thu thực sự là **kiểm tra runtime**: mở file bằng `onnxruntime` và xác nhận đúng `input [1,3,288,800]` → `output [1,101,56,4]` (mục 2) và suy luận được trên CPU. Giá trị SHA-256 do Quang ghi trước đó là `fb28bbecb372873ea16f92594bb2307d71ea1aaeb770138c9d425166650bdf16` (dung lượng `244,894,648` bytes); sai khác 1 byte/checksum là do quá trình serialize, không phải do sai model — xem báo cáo tích hợp.

### Nguồn tham chiếu chính thức
1. **Repo tác giả gốc (PyTorch weights):** [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
   - Google Drive (`tusimple_18.pth`): `https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing`
2. **ONNX Inference Reference:** [ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference)
3. **PINTO Model Zoo (140_Ultra-Fast-Lane-Detection):** [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection)
4. **ONNX Export Script Reference:** [cfzd/Ultra-Fast-Lane-Detection#218](https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/218)

---

## 2. Cấu hình Input / Output thực tế (đã xác minh qua `onnxruntime`)

Kết quả truy vấn trực tiếp từ `ort.InferenceSession("models/ufld_tusimple.onnx", providers=["CPUExecutionProvider"])`:

### Input Tensor
| Thuộc tính | Giá trị | Ghi chú |
|---|---|---|
| `input.name` | `'input'` | Tên node đầu vào |
| `input.shape` | `[1, 3, 288, 800]` | `NCHW` (`batch=1, channels=3, height=288, width=800`) |
| `input.type` | `tensor(float)` (`float32`) | Ảnh RGB đã chuẩn hóa theo ImageNet |

- **Preprocessing chuẩn:**
  1. Đổi hệ màu `BGR` (OpenCV) sang `RGB` và resize về `(800, 288)`.
  2. Chuẩn hóa pixel: `((img / 255.0) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`.
  3. Chuyển chiều `HWC -> NCHW` (`[1, 3, 288, 800]`, kiểu `np.float32`).

### Output Tensor
| Thuộc tính | Giá trị | Ghi chú |
|---|---|---|
| `output.name` | `'output'` | Tên node đầu ra |
| `output.shape` | `[1, 101, 56, 4]` | `[batch, griding_num + 1, cls_num_per_lane, num_lanes]` |
| `output.type` | `tensor(float)` (`float32`) | Logits phân loại vị trí lưới theo từng hàng neo |

- **Decoding & Postprocessing chuẩn:**
  - `griding_num = 100` (100 ô lưới ngang + 1 lớp vắng mặt ở index `100`).
  - `cls_num_per_lane = 56` tương ứng 56 `tusimple_row_anchor` từ `y = 64` đến `y = 284` (trong không gian cao `288`).
  - `num_lanes = 4` (tối đa 4 làn đường).
  - Áp dụng đảo thứ tự hàng, `softmax` trên 100 ô đầu tiên để tính kỳ vọng vị trí ngang `loc`, gán `0` tại các hàng có `argmax == 100`.
  - Quy đổi tọa độ từ không gian model `(800, 288)` về kích thước frame gốc `(W, H)`, lọc điểm ngoài biên, sắp xếp `y` tăng dần nghiêm ngặt và sắp thứ tự làn từ trái sang phải (`lane_index = 0, 1, ...`).

---

## 3. Hướng dẫn tải và chuẩn bị `models/ufld_tusimple.onnx`

### Cách 1 (Khuyên dùng — Nhanh, chỉ ~233 MB): Tải checkpoint chính thức `tusimple_18.pth` và xuất sang `models/ufld_tusimple.onnx`

Do gói `resources_tusimple.tar.gz` trên PINTO Model Zoo chứa kèm toàn bộ bản build TensorFlow/TFLite/OpenVINO/CoreML nên nặng tới **6.87 GB**, cách nhanh nhất (~1 phút) là tải trực tiếp trọng số pretrained `tusimple_18.pth` của tác giả `cfzd` từ Google Drive và xuất sang `models/ufld_tusimple.onnx` theo đúng script tại Issue #218:

```bash
python -c "
import os, urllib.request, torch, torchvision

os.makedirs('models', exist_ok=True)
pth_path = 'models/tusimple_18.pth'
onnx_path = 'models/ufld_tusimple.onnx'
url = 'https://drive.usercontent.google.com/download?id=1WCYyur5ZaWczH15ecmeDowrW30xcLrCn&export=download&confirm=t'

if not os.path.exists(pth_path):
    print('Downloading tusimple_18.pth (~233 MB)...')
    urllib.request.urlretrieve(url, pth_path)

class resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
    def forward(self, x):
        x = self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        x2 = self.layer2(x); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return x2, x3, x4

class parsingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet()
        self.pool = torch.nn.Conv2d(512, 8, 1)
        self.cls = torch.nn.Sequential(torch.nn.Linear(1800, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 101*56*4))
    def forward(self, x):
        _, _, fea = self.model(x)
        return self.cls(self.pool(fea).view(-1, 1800)).view(-1, 101, 56, 4)

state = torch.load(pth_path, map_location='cpu', weights_only=False)
sd = {k.replace('module.', ''): v for k, v in (state['model'] if 'model' in state else state).items()}
net = parsingNet()
net.load_state_dict(sd, strict=False)
net.eval()
torch.onnx.export(net, torch.zeros(1, 3, 288, 800), onnx_path, input_names=['input'], output_names=['output'], opset_version=11, dynamo=False)
os.remove(pth_path)
print('Saved ONNX model to:', onnx_path)
"
```

### Cách 2: Tải từ PINTO0309 Model Zoo (Archive đầy đủ ~6.87 GB)

1. Tải gói `resources_tusimple.tar.gz` từ script [`download_tusimple.sh`](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/download_tusimple.sh).
2. Giải nén và lấy file `saved_model_tusimple/ultra_falst_lane_detection_tusimple_288x800.onnx`.
3. Đổi tên và đặt vào đường dẫn:
   ```text
   models/ufld_tusimple.onnx
   ```

---

## 4. Kiểm tra nhanh sau khi đặt weights

Chạy smoke test của module Detector:

```bash
python tests/test_detector.py --model models/ufld_tusimple.onnx --image data/sample_frame.jpg
```

Hoặc chạy toàn bộ bộ kiểm thử `pytest`:

```bash
pytest tests/test_detector.py -v
```

---

## 5. Giấy phép (License)

Trọng số và kiến trúc mô hình UFLD gốc thuộc bản quyền của tác giả **Zequn Qin (`cfzd`)**, phát hành theo giấy phép **MIT License**:

- Copyright (c) 2020 cfzd ([Ultra-Fast-Lane-Detection LICENSE](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/LICENSE)).
- Chỉ sử dụng cho mục đích nghiên cứu/trình diễn học thuật trong khuôn khổ đồ án, không sử dụng cho hệ thống điều khiển phương tiện thực tế.

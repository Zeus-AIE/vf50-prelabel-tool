# VF-50 Pre-labeling Tool 🚀
### Bộ công cụ AI Pre-label Facial Landmark 50 Điểm (VF-50) & Chỉnh Nhãn Cho Cabin Xe Hơi (IR & RGB)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SCRFD-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20CUDA%2012-ee4c2c.svg)](https://pytorch.org/)
[![CVAT 1.1](https://img.shields.io/badge/CVAT-1.1%20XML%20Ready-orange.svg)](https://www.cvat.ai/)

---

## 📖 Giới Thiệu

**VF-50 Pre-labeling Tool** là bộ giải pháp toàn diện phục vụ việc gán nhãn tự động và tiền xử lý dữ liệu **Facial Landmark 50 điểm (VF-50)** theo đúng chuẩn của dự án VinFast / VinUni.

Công cụ được thiết kế đặc thù để giải quyết triệt để các thách thức trong môi trường cabin xe thực tế:
- **Ảnh hồng ngoại (IR):** Thiếu sáng, hạt nhiễu cao, lóa đèn IR, mắt và môi bị mất độ tương phản.
- **Tư thế người lái phức tạp:** Quay đầu góc lớn (Yaw > 35° đến > 55°), cúi/ngửa gắt (Pitch), nghiêng (Roll).
- **Góc camera lắp lệch:** Hỗ trợ tự động phát hiện hoặc ép xoay góc ảnh `90° CW`, `90° CCW`, `180°`.
- **Vật cản & Phụ kiện:** Người lái đeo kính râm, khẩu trang, bị vô lăng hoặc tay che khuất.
- **Tự động suy luận trạng thái:** Tự gán thuộc tính `visible`, `occluded`, `outside` theo chuẩn **Bảng 3.1** trong Guideline.
- **Xuất trực tiếp CVAT 1.1 XML:** Khớp 1-1 với 7 skeleton trên CVAT, import vào là làm việc được ngay.

---

## ⚡ Các Tính Năng Nổi Bật

1. **Bộ chọn 3 Mức độ Sức mạnh (Level Selector):**
   - **Mức 1 (Nhanh cơ bản):** YuNet + SPIGA WFLW (~80ms/ảnh, siêu nhẹ cho CPU hoặc máy yếu).
   - **Mức 2 (Mạnh nhất - Khuyên dùng):** SCRFD-10G + Ensemble SPIGA & TUFA (60/40) + 6DRepNet 3D Headpose + 2-Pass Refinement + Flip TTA + Đo độ bất định từng điểm (vòng tròn halo cam/đỏ cảnh báo điểm mốc nghi ngờ).
   - **Mức 3 (Chuyên sâu IR):** SCRFD-10G + SPIGA đã được Fine-tune chuyên biệt trên bộ Synthetic IR Augmentation (giảm NME trên ảnh IR cabin).
2. **Dán trực tiếp ảnh chụp màn hình (Clipboard Paste):**
   - Hỗ trợ nhấn `Ctrl + V` trực tiếp ở bất kỳ đâu trên giao diện web.
   - Nút **Dán từ Clipboard** tự động đọc ảnh chụp màn hình từ bộ nhớ tạm OS.
   - Hỗ trợ kéo thả ảnh (Drag & Drop) hoặc tải cả folder ảnh.
3. **Canvas Chỉnh Nhãn Trực Quan (Interactive Editor):**
   - **Kéo chuột trái:** Dời vị trí điểm landmark.
   - **Phím tắt nhanh khi hover:** Nhấn `V` (`Visible`), `O` (`Occluded`), `X` (`Outside`).
   - **Phím `B` (Vẽ khung mặt thủ công):** Kéo chuột khoanh vùng mặt cho các frame quá tối/mờ mà AI không tự detect được.
   - **Nút gán nhanh nhóm Outside:** Cho phép ẩn 1-click cả skeleton (`matphai`, `longmayphai`...) khi quay mặt góc khuất.
4. **Kiểm tra chất lượng & Xuất dữ liệu:**
   - Xuất file `annotations.xml` chuẩn **CVAT for images 1.1**.
   - Xuất file báo cáo `qc_report.csv` cảnh báo các frame có góc nghiêng lớn, điểm mờ bất định (`LOW_CONFIDENCE_POINTS`) hoặc mất cân đối topology mí mắt/môi.

---

## 🏗️ Cấu Trúc Thư Mục

```text
vf50-prelabel-tool/
├── setup.bat                  # Script tự động tạo môi trường và tải models (1-click)
├── run_ui.bat                 # Script khởi động nhanh Web UI (http://127.0.0.1:7860)
├── requirements.txt           # Danh sách thư viện Python chuẩn
├── cli.py                     # Công cụ chạy batch qua dòng lệnh
├── models/                    # Thư mục chứa model weights (tải tự động)
│   ├── scrfd_10g_bnkps.onnx
│   ├── face_detection_yunet_2023mar.onnx
│   ├── spiga_wflw.pt
│   ├── 6DRepNet_300W_LP_AFLW2000.pth
│   ├── tufa_face_prompt_base.pth
│   └── tufa_shape_98.npz
├── scripts/
│   ├── download_all_models.py # Tải tự động và verify toàn bộ weights
│   └── benchmark_level2.py    # Script benchmark đánh giá NME
├── src/
│   ├── vf50_schema.py         # Định nghĩa 7 skeleton (0–49) & ánh xạ WFLW-98
│   ├── detector.py            # YuNet face detector + CLAHE IR enhancement
│   ├── scrfd_detector.py      # SCRFD-10G ONNX detector đa tỉ lệ
│   ├── landmark_engine.py     # SPIGA + Ensemble TUFA + Flip TTA + Uncertainty
│   ├── tufa_engine.py         # TUFA DINO Transformer refiner
│   ├── headpose_6drepnet.py   # 6DRepNet 3D pose estimator (Yaw/Pitch/Roll)
│   ├── vf50_rules.py          # Topology nắn mí/môi & luật Bảng 3.1
│   ├── cvat_exporter.py       # Xuất CVAT XML 1.1, JSON, QC Report CSV
│   ├── visualizer.py          # Render ảnh preview kèm ID & vòng bất định
│   ├── pipeline.py            # Điều phối logic 3 Mức độ
│   └── web_app.py             # FastAPI backend phục vụ Web UI
├── static/
│   └── index.html             # Giao diện Web Canvas tương tác đầy đủ
└── vendor/                    # Mã nguồn inference sạch cho SPIGA, TUFA, 6DRepNet
```

---

## ⚙️ Cài Đặt Nhanh (1-Click Setup)

### Yêu Cầu Hệ Thống:
- **Hệ điều hành:** Windows 10/11 (hoặc Linux/macOS).
- **Python:** 3.10 trở lên (khuyên dùng Python 3.10 hoặc 3.11).
- **Phần cứng:**
  - *CPU mode:* Core i5 / Ryzen 5 trở lên, RAM 8GB (chạy tốt Mức 1 & Mức 2).
  - *GPU mode (Khuyên dùng):* NVIDIA RTX (GTX 1650, RTX 3050, RTX 3060 6GB, RTX 4060...), VRAM chỉ cần ~1.5–2.5 GB.

### Bước 1: Clone Repository
```powershell
git clone https://github.com/Zeus-AIE/vf50-prelabel-tool.git
cd vf50-prelabel-tool
```

### Bước 2: Chạy Tự Động Cài Đặt
Chỉ cần nhấp đúp file **`setup.bat`** (hoặc chạy từ terminal):
```powershell
.\setup.bat
```
*Script sẽ tự động:*
1. Tạo môi trường ảo `.venv`.
2. Kiểm tra GPU và cài PyTorch (CUDA 12.1 hoặc CPU tương ứng).
3. Cài các thư viện từ `requirements.txt`.
4. Tự động tải toàn bộ 6 model weights từ HuggingFace & Google Drive và verify tính toàn vẹn.

---

## 🖥️ Hướng Dẫn Sử Dụng Web UI

Nhấp đúp **`run_ui.bat`** hoặc chạy:
```powershell
.\run_ui.bat
```
Trình duyệt sẽ tự động mở tại địa chỉ: **`http://127.0.0.1:7860`**

### Quy trình làm việc trên giao diện:
1. **Nạp ảnh vào làm việc:**
   - **Cách 1:** Nhấn `Ctrl + V` để dán trực tiếp ảnh chụp màn hình.
   - **Cách 2:** Bấm nút **📋 Dán từ Clipboard**.
   - **Cách 3:** Dán đường dẫn thư mục ảnh vào ô *Đường dẫn thư mục* và bấm **Nạp thư mục**.
   - **Cách 4:** Kéo thả file ảnh trực tiếp vào danh sách ảnh bên trái.
2. **Chọn Mức Độ AI:**
   - Chọn **Mức 1**, **Mức 2 (Mạnh nhất)** hoặc **Mức 3 (IR)** trên thanh chọn mức độ.
3. **Cấu hình bổ sung (nếu cần):**
   - *Góc xoay:* Để `Auto` (tự động phát hiện ảnh xoay) hoặc chọn `90 CW` / `90 CCW` / `180`.
   - *Vị trí lái:* Mặc định `Lớn nhất` (khuôn mặt chính).
   - *Ngưỡng Yaw:* `35°` cho Occluded, `55°` cho Outside (theo đúng Guideline).
4. **Bấm Chạy Pre-label:**
   - Bấm **Chạy Ảnh Này** (đối với ảnh đang chọn) hoặc **Chạy Tất Cả** (đối với toàn bộ lô ảnh).
5. **Tinh chỉnh điểm mốc trên Canvas:**
   - Nhìn các điểm có **vòng tròn halo màu cam/đỏ** (đây là các điểm AI tự đánh giá độ bất định cao).
   - Nhấp giữ chuột trái và kéo để điều chỉnh điểm về đúng vị trí giải phẫu.
   - Nhấp chuột phải vào điểm (hoặc hover và bấm phím `V`, `O`, `X`) để chuyển trạng thái.
   - Nếu mặt quá mờ khiến AI không nhận diện được: Bấm phím **`B`** (hoặc nút *Vẽ khung mặt thủ công*), sau đó dùng chuột quét một hình chữ nhật quanh khuôn mặt.
   - Khi người lái quay mặt góc nghiêng lớn, dùng các nút ở cột bên phải để đặt cả nhóm (`Mắt phải = Outside`, `Lông mày phải = Outside`...) chỉ với 1 click.
6. **Xuất kết quả:**
   - Bấm **📥 Xuất CVAT XML** để tải file `annotations.xml`.

---

## 💻 Hướng Dẫn Chạy Batch Bằng CLI

Nếu bạn cần xử lý tự động hàng nghìn ảnh mà không cần mở giao diện web:

```powershell
.\.venv\Scripts\python.exe cli.py `
  --input "D:\Dataset\Cabin_IR_Batch1" `
  --output "D:\Dataset\CVAT_Export" `
  --level 2 `
  --rotate auto `
  --driver-side largest `
  --yaw-occluded 35 `
  --yaw-outside 55 `
  --save-previews
```

### Sử dụng file mẫu CVAT Template:
Nếu bạn đã tạo Task trên CVAT và có sẵn 7 skeleton cấu hình từ trước, hãy export file XML rỗng từ CVAT ra và truyền vào cờ `--template-xml`:

```powershell
.\.venv\Scripts\python.exe cli.py `
  --input "D:\Dataset\Cabin_IR_Batch1" `
  --output "D:\Dataset\CVAT_Export" `
  --template-xml "D:\Dataset\cvat_task_template.xml" `
  --level 2 `
  --save-previews
```

### Các file sinh ra trong thư mục `--output`:
- `annotations.xml`: File chú thích hoàn chỉnh định dạng CVAT 1.1 XML.
- `predictions.json`: Tọa độ chi tiết (x, y), độ tin cậy, độ bất định và góc xoay 3D (Yaw, Pitch, Roll).
- `qc_report.csv`: Danh sách kiểm soát chất lượng, liệt kê từng ảnh kèm cờ cảnh báo (`LOW_VISIBILITY`, `UNCERTAIN_BOUNDARY`, `LOW_CONFIDENCE_POINTS`).
- `previews/`: Thư mục ảnh đã vẽ các điểm mốc và ID để rà soát nhanh.

---

## 📥 Hướng Dẫn Import `annotations.xml` Vào CVAT

1. Truy cập vào Task dự án trên giao diện web **CVAT**.
2. Nhấp vào menu **Actions** (góc trên bên phải) $\rightarrow$ chọn **Upload annotations**.
3. Trong danh sách định dạng, chọn **`CVAT 1.1`** (hoặc `CVAT for images 1.1`).
4. Chọn file `annotations.xml` vừa xuất từ công cụ $\rightarrow$ Bấm **Submit**.
5. CVAT sẽ tự động nạp đủ 7 skeleton (`longmaytrai`, `longmayphai`, `songmui`, `mattrai`, `matphai`, `moingoai`, `moitrong`) với đúng 50 điểm ID `0–49` và đầy đủ thuộc tính `occluded` / `outside` vào từng frame ảnh!

---

## 📐 Danh Mục 50 Điểm Mốc VF-50 (7 Skeletons)

| Nhóm Skeleton | Số Điểm | ID Điểm (0–49) | Đặc điểm giải phẫu chính |
| :--- | :---: | :---: | :--- |
| **`longmaytrai`** | 5 | `0 – 4` | Cung mày trái: 0 (đầu mày trong) $\rightarrow$ 4 (đuôi mày ngoài) |
| **`longmayphai`** | 5 | `5 – 9` | Cung mày phải: 5 (đuôi mày ngoài) $\rightarrow$ 9 (đầu mày trong) |
| **`songmui`** | 9 | `10 – 18` | Sống mũi (10–13), đỉnh chóp mũi (14), chân cánh mũi (15–18) |
| **`mattrai`** | 8 | `19 – 26` | Mắt trái: 19 (khóe trong), 22 (đuôi ngoài), 20–21 (mí trên), 23–25 (mí dưới), 26 (tâm con ngươi) |
| **`matphai`** | 8 | `27 – 34` | Mắt phải: 27 (đuôi ngoài), 30 (khóe trong), 28–29 (mí trên), 31–33 (mí dưới), 34 (tâm con ngươi) |
| **`moingoai`** | 12 | `35 – 46` | Đường viền môi ngoài: khóe môi trái (35) $\rightarrow$ khóe môi phải (41) |
| **`moitrong`** | 3 | `47 – 49` | Khe mở môi trong: 47 (mép môi trên trong), 48 (khe mở trung tâm), 49 (mép môi dưới trong) |

---

## 🤝 Hỗ Trợ & Đóng Góp

- Mọi thắc mắc hoặc yêu cầu bổ sung tính năng, vui lòng tạo **Issue** trên GitHub repository hoặc liên hệ nhóm phát triển.
- Chúc các thành viên trong nhóm pre-label và gán nhãn đạt độ chính xác cao nhất và hoàn thành xuất sắc đồ án! 🎉

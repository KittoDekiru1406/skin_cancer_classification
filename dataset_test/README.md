# Dataset Test - Skin Cancer Classification

## Mô tả Dataset

Dataset này chứa hình ảnh test để phục vụ việc chẩn đoán bệnh ngoài da qua ảnh bằng mô hình mạng nơ-ron tích chập. Dataset được tổ chức theo 7 loại bệnh chính:

## Phân loại các loại bệnh

| Mã | Tên tiếng Anh | Tên tiếng Việt | Bản chất & Nguy cơ |
|---|---|---|---|
| **nv** | Melanocytic nevi | Nốt ruồi sắc tố | **Lành tính** - Chiếm 66.9% cases |
| **mel** | Melanoma | Ung thư hắc tố | **Ác tính** - Gây tử vong cao nếu muộn (11.1% cases) |
| **bkl** | Benign keratosis-like lesions | Keratosis lành tính | **Lành tính** - Khoảng 10.9% cases |
| **bcc** | Basal cell carcinoma | Ung thư tế bào đáy | **Ung thư** không phổ biến nhưng xâm lấn tại chỗ (5.1% cases) |
| **akiec** | Actinic keratosis & Bowen's disease | Keratosis do ánh sáng | **Tiền ung thư** (SCC in situ) - 3.3% cases |
| **vasc** | Vascular lesions | Tổn thương mạch máu | **Lành tính** hoặc hiếm khi ác - 1.4% cases |
| **df** | Dermatofibroma | U xơ da | **Lành tính** - 1.1% cases |

## Chi tiết từng loại bệnh

### 1. NV - Melanocytic nevi (Nốt ruồi sắc tố)
- **Bản chất**: Tổn thương lành tính từ tế bào melanocyte
- **Đặc điểm**: Đốm nâu/đen đồng đều, giới hạn rõ, kích thước < 6mm
- **Nguy cơ**: Lành tính, có thể bẩm sinh hoặc xuất hiện sau sinh

### 2. MEL - Melanoma (Ung thư hắc tố)
- **Bản chất**: Ung thư ác tính từ melanocyte
- **Đặc điểm**: Mảng không đều màu, bất xứng, đường viền không đều
- **Nguy cơ**: Rất nguy hiểm, cần sinh thiết đánh giá độ sâu (Breslow)

### 3. BKL - Benign keratosis-like lesions (Keratosis lành tính)
- **Bao gồm**: Seborrhoic keratosis, solar lentigo, lichen planus-like keratosis
- **Đặc điểm**: Tổn thương sắc tố, dạng nốt sần "dính trên da"
- **Nguy cơ**: Lành tính, thường gặp ở người lớn tuổi

### 4. BCC - Basal cell carcinoma (Ung thư tế bào đáy)
- **Bản chất**: Ung thư da phổ biến nhất
- **Đặc điểm**: Nốt đỏ-trắng, mạch máu nổi, dễ chảy máu
- **Nguy cơ**: Mọc chậm, ít di căn nhưng cần điều trị

### 5. AKIEC - Actinic keratosis & Bowen's disease
- **Bản chất**: Tổn thương tiền ung thư
- **Đặc điểm**: Da ngứa, đóng vảy, thô ráp ở vùng tiếp xúc nắng
- **Nguy cơ**: Có thể tiến triển thành ung thư biểu mô

### 6. VASC - Vascular lesions (Tổn thương mạch máu)
- **Bao gồm**: Angioma, angiokeratoma, pyogenic granuloma
- **Đặc điểm**: Tổn thương đỏ, tím, dễ chảy máu
- **Nguy cơ**: Chủ yếu lành tính

### 7. DF - Dermatofibroma (U xơ da)
- **Bản chất**: Nốt sợi lành tính
- **Đặc điểm**: Nốt nhỏ sẫm màu, có "dimple sign" khi ép
- **Nguy cơ**: Hoàn toàn lành tính

## Cấu trúc Dataset Test

```
dataset_test/
├── akiec/          # 10 ảnh
├── bcc/            # 10 ảnh  
├── bkl/            # 10 ảnh
├── df/             # 7 ảnh
├── mel/            # 10 ảnh
├── nv/             # 10 ảnh
└── vasc/           # 11 ảnh
```

**Tổng cộng**: 68 ảnh test được phân loại theo 7 categories

## Định dạng ảnh
- **Format**: JPG
- **Naming**: ISIC_XXXXXXX.jpg (tuân theo chuẩn ISIC Archive)
- **Resolution**: Đa dạng, sẽ được resize trong quá trình preprocessing

## Mục đích sử dụng
Dataset này được sử dụng để:
- Test hiệu năng của các mô hình đã train
- Demo hệ thống chẩn đoán tự động
- Đánh giá độ chính xác của từng model (MobileNet, ResNet50, Self-build CNN)

## Lưu ý quan trọng
⚠️ **Dataset này chỉ phục vụ mục đích học tập và nghiên cứu. Không được sử dụng để thay thế chẩn đoán y tế chuyên nghiệp.**

---
*Dataset được tổ chức từ HAM10000 dataset - Human Against Machine with 10000 training images*

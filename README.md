# 🏥 Hệ Thống Chẩn Đoán Bệnh Ngoài Da

**Xây dựng hệ thống chẩn đoán bệnh ngoài da qua ảnh bằng mô hình mạng nơ-ron tích chập**

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.104-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28-red.svg)

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Tính Năng](#tính-năng)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Models](#models)
- [Dataset](#dataset)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Contributors](#contributors)

## 🎯 Giới Thiệu

Hệ thống chẩn đoán bệnh ngoài da sử dụng Deep Learning để phân loại 7 loại bệnh da phổ biến:

| Mã | Tên Bệnh | Loại | Nguy Cơ |
|---|---|---|---|
| **NV** | Melanocytic nevi (Nốt ruồi sắc tố) | Lành tính | Thấp |
| **MEL** | Melanoma (Ung thư hắc tố) | Ác tính | Rất cao |
| **BKL** | Benign keratosis (Keratosis lành tính) | Lành tính | Thấp |
| **BCC** | Basal cell carcinoma (Ung thư tế bào đáy) | Ung thư | Trung bình |
| **AKIEC** | Actinic keratosis (Keratosis do ánh sáng) | Tiền ung thư | Cao |
| **VASC** | Vascular lesions (Tổn thương mạch máu) | Lành tính | Thấp |
| **DF** | Dermatofibroma (U xơ da) | Lành tính | Thấp |

## ✨ Tính Năng

### 🔍 Chẩn Đoán Chính
- **Upload ảnh**: Hỗ trợ PNG, JPG, JPEG
- **Multi-model prediction**: So sánh kết quả từ 3 models
- **Consensus prediction**: Kết quả tổng hợp từ tất cả models
- **Confidence scoring**: Độ tin cậy cho mỗi dự đoán
- **Real-time results**: Kết quả tức thời

### 📊 Visualization
- **Interactive charts**: Biểu đồ tương tác với Plotly
- **Probability distribution**: Phân bố xác suất tất cả classes
- **Model comparison**: So sánh hiệu năng các models
- **Disease information**: Thông tin chi tiết về từng bệnh

### 🎨 Giao Diện
- **Modern UI**: Giao diện hiện đại với Streamlit
- **Responsive design**: Tương thích đa thiết bị
- **Color-coded results**: Mã màu theo mức độ nguy hiểm
- **Multi-language**: Hỗ trợ tiếng Việt

## 🏗️ Kiến Trúc Hệ Thống

```
📦 Skin Cancer Classification System
├── 🔙 Backend (FastAPI)
│   ├── Model Loading & Management
│   ├── Image Preprocessing
│   ├── Prediction Engine
│   └── REST API Endpoints
│
├── 🖥️ Frontend (Streamlit)
│   ├── File Upload Interface
│   ├── Model Selection
│   ├── Results Visualization
│   └── Disease Information
│
├── 🤖 AI Models
│   ├── MobileNet (Transfer Learning)
│   ├── ResNet50 (Transfer Learning)
│   └── Custom CNN (Self-built)
│
└── 📊 Data
    ├── Training Dataset (HAM10000)
    ├── Test Dataset (68 images)
    └── Model Weights
```

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.9+
- 8GB RAM (khuyến nghị)
- GPU (tùy chọn, tăng tốc inference)

### Cài Đặt Nhanh

```bash
# Clone repository
git clone https://github.com/your-username/skin-cancer-classification.git
cd skin-cancer-classification

# Chạy script setup
chmod +x setup.sh
./setup.sh
```

### Cài Đặt Thủ Công

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build và chạy với Docker Compose
docker-compose up --build

# Hoặc chỉ backend
docker build -t skin-cancer-backend .
docker run -p 8000:8000 skin-cancer-backend
```

## 🎮 Sử Dụng

### Khởi Chạy Hệ Thống

#### Option 1: Khởi chạy riêng biệt
```bash
# Terminal 1 - Backend API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend UI  
streamlit run streamlit_app.py --server.port 8501
```

#### Option 2: Docker Compose
```bash
docker-compose up
```

### Truy Cập Ứng Dụng
- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Quy Trình Sử Dụng

1. **📤 Upload Ảnh**: Chọn ảnh da cần chẩn đoán
2. **🤖 Chọn Model**: Lựa chọn model hoặc "Tất cả models"
3. **🔍 Chẩn Đoán**: Click "Bắt Đầu Chẩn Đoán"
4. **📊 Xem Kết Quả**: Phân tích kết quả và đề xuất
5. **💾 Lưu Kết Quả**: Export hoặc screenshot kết quả

## 🤖 Models

### 1. MobileNet
- **Architecture**: MobileNetV1 + Custom Classifier
- **Input Size**: 224x224x3
- **Parameters**: ~4.2M
- **Speed**: Nhanh nhất
- **Use Case**: Mobile deployment

### 2. ResNet50
- **Architecture**: ResNet50 + Custom Classifier  
- **Input Size**: 224x224x3
- **Parameters**: ~25.6M
- **Accuracy**: Cao nhất
- **Use Case**: High accuracy applications

### 3. Self-Build CNN
- **Architecture**: Custom CNN từ scratch
- **Input Size**: 224x224x3
- **Parameters**: ~2.1M
- **Balance**: Cân bằng speed/accuracy
- **Use Case**: Lightweight deployment

### Model Performance

| Model | Accuracy | Speed | Size | Memory |
|-------|----------|-------|------|--------|
| MobileNet | 85.2% | ⭐⭐⭐ | 17MB | 512MB |
| ResNet50 | 91.7% | ⭐⭐ | 98MB | 1.2GB |
| Self-Build | 87.9% | ⭐⭐⭐ | 8MB | 256MB |

## 📊 Dataset

### Training Dataset: HAM10000
- **Total Images**: 10,015
- **Source**: Human Against Machine dataset
- **Classes**: 7 skin disease types
- **Format**: JPEG images
- **Resolution**: Varied (resized to 224x224)

### Test Dataset
- **Location**: `dataset_test/`
- **Total Images**: 68
- **Distribution**:
  - akiec: 10 images
  - bcc: 10 images  
  - bkl: 10 images
  - df: 7 images
  - mel: 10 images
  - nv: 10 images
  - vasc: 11 images

## 📚 API Documentation

### Endpoints

#### Health Check
```http
GET /
```

#### Get Available Models
```http
GET /models
```

#### Single Model Prediction
```http
POST /predict/{model_name}
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG/JPG/JPEG)
- model_name: mobilenet|resnet50|self_build
```

#### All Models Prediction
```http
POST /predict/all
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG/JPG/JPEG)
```

#### Disease Information
```http
GET /disease-info
```

### Response Format

```json
{
  "model_used": "mobilenet",
  "filename": "skin_lesion.jpg",
  "result": {
    "predicted_class": "mel",
    "confidence": 0.892,
    "all_predictions": {
      "mel": 0.892,
      "nv": 0.045,
      "bcc": 0.032,
      "...": "..."
    },
    "disease_info": {
      "name": "Melanoma (Ung thư hắc tố)",
      "type": "Ác tính",
      "description": "...",
      "risk": "Rất cao"
    }
  },
  "status": "success"
}
```

## 📸 Screenshots

### Main Interface
![Main Interface](docs/screenshots/main_interface.png)

### Prediction Results
![Prediction Results](docs/screenshots/prediction_results.png)

### Model Comparison
![Model Comparison](docs/screenshots/model_comparison.png)

### Disease Information
![Disease Info](docs/screenshots/disease_info.png)

## 🔧 Development

### Project Structure
```
skin_cancer_classification/
├── 📁 config/                 # Configuration & settings
│   ├── __init__.py
│   └── settings.py            # Constants, disease info, paths
├── 📁 models/                 # Data models & schemas
│   ├── __init__.py
│   └── schemas.py             # Pydantic models, enums, responses
├── 📁 services/               # Business logic
│   ├── __init__.py
│   └── prediction_service.py  # Prediction logic & consensus
├── 📁 utils/                  # Utilities
│   ├── __init__.py
│   ├── image_utils.py         # Image preprocessing
│   └── model_utils.py         # Model loading & management
├── 📄 app.py                  # FastAPI backend
├── 📄 streamlit_app.py        # Streamlit frontend
├── 📄 requirements.txt        # Dependencies
├── 📄 Dockerfile             # Docker configuration
├── 📄 docker-compose.yml     # Multi-container setup
├── 📄 setup.sh              # Setup script
├── 📄 start_demo.sh          # Quick start script
├── 📄 README.md              # This file
├── 📄 QUICKSTART.md          # Quick start guide
├── 📁 dataset_test/          # Test images
│   ├── README.md
│   ├── akiec/               # Actinic keratosis images
│   ├── bcc/                 # Basal cell carcinoma images
│   ├── bkl/                 # Benign keratosis images
│   ├── df/                  # Dermatofibroma images
│   ├── mel/                 # Melanoma images
│   ├── nv/                  # Melanocytic nevi images
│   └── vasc/                # Vascular lesions images
├── 📁 weights/              # Trained model weights
│   ├── mobilenet/           # MobileNet model files
│   ├── resnet50/            # ResNet50 model files
│   └── self_build/          # Custom CNN model files
├── 📁 notebook/             # Training notebooks
│   ├── mobilenet-skin-cancer.ipynb
│   ├── resnet50-skin-cancer-classification.ipynb
│   └── self-build-skin-cancer.ipynb
└── 📁 logs/                # Application logs
```

### Adding New Models

1. **Train Model**: Sử dụng notebooks trong `notebook/`
2. **Save Weights**: Lưu vào `weights/new_model/`
3. **Update Backend**: Thêm loading logic vào `app.py`
4. **Update Frontend**: Thêm option vào `streamlit_app.py`

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## ⚠️ Disclaimer

**QUAN TRỌNG**: Hệ thống này chỉ phục vụ mục đích nghiên cứu và học tập. 

**KHÔNG được sử dụng để thay thế chẩn đoán y tế chuyên nghiệp!**

Luôn tham khảo ý kiến bác sĩ da liễu cho việc chẩn đoán chính xác.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- **Tên của bạn** - *Initial work* - [GitHub](https://github.com/your-username)

## 🙏 Acknowledgments

- HAM10000 Dataset providers
- TensorFlow & Keras teams
- FastAPI & Streamlit communities
- Medical professionals for domain knowledge

---

⭐ **Star** this repository if you find it helpful!

📧 **Contact**: your-email@example.com

🔗 **Project Link**: https://github.com/your-username/skin-cancer-classification

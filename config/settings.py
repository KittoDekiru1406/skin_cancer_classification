"""
Configuration và constants cho skin cancer classification system
"""

from typing import Dict, Tuple
from models.schemas import DiseaseInfo, DiseaseCategory, RiskLevel


# Image processing constants
IMAGE_SIZE: Tuple[int, int] = (224, 224)  # Standard for MobileNet & ResNet50
SELFBUILD_IMAGE_SIZE: Tuple[int, int] = (75, 100)  # Specific for self-build model
SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = ('png', 'jpg', 'jpeg')
MAX_IMAGE_SIZE_MB: int = 10

# Model constants
CLASS_NAMES: Tuple[str, ...] = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
NUM_CLASSES: int = len(CLASS_NAMES)

# ImageNet normalization constants (for ResNet50)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Disease information mapping
DISEASE_INFO_MAP: Dict[str, DiseaseInfo] = {
    'nv': DiseaseInfo(
        name='Melanocytic nevi (Nốt ruồi sắc tố)',
        type=DiseaseCategory.BENIGN,
        description='Tổn thương lành tính từ tế bào melanocyte, tạo ra các đốm nâu/đen đồng đều',
        risk=RiskLevel.LOW,
        color='#28a745'
    ),
    'mel': DiseaseInfo(
        name='Melanoma (Ung thư hắc tố)',
        type=DiseaseCategory.MALIGNANT,
        description='Ung thư ác tính từ melanocyte, có thể gây tử vong cao nếu không điều trị kịp thời',
        risk=RiskLevel.VERY_HIGH,
        color='#dc3545'
    ),
    'bkl': DiseaseInfo(
        name='Benign keratosis (Keratosis lành tính)',
        type=DiseaseCategory.BENIGN,
        description='Các tổn thương sắc tố lành tính, thường gặp ở người lớn tuổi',
        risk=RiskLevel.LOW,
        color='#28a745'
    ),
    'bcc': DiseaseInfo(
        name='Basal cell carcinoma (Ung thư tế bào đáy)',
        type=DiseaseCategory.CANCER,
        description='Ung thư da phổ biến nhất, mọc chậm nhưng có thể xâm lấn tại chỗ',
        risk=RiskLevel.MEDIUM,
        color='#fd7e14'
    ),
    'akiec': DiseaseInfo(
        name='Actinic keratosis (Keratosis do ánh sáng)',
        type=DiseaseCategory.PRE_CANCER,
        description='Tổn thương tiền ung thư có thể tiến triển thành ung thư biểu mô',
        risk=RiskLevel.HIGH,
        color='#ffc107'
    ),
    'vasc': DiseaseInfo(
        name='Vascular lesions (Tổn thương mạch máu)',
        type=DiseaseCategory.BENIGN,
        description='Tổn thương mạch máu như angioma, thường lành tính',
        risk=RiskLevel.LOW,
        color='#17a2b8'
    ),
    'df': DiseaseInfo(
        name='Dermatofibroma (U xơ da)',
        type=DiseaseCategory.BENIGN,
        description='Nốt sợi lành tính, thường có dấu hiệu "dimple sign"',
        risk=RiskLevel.LOW,
        color='#6f42c1'
    )
}

# Model file paths
MODEL_PATHS: Dict[str, Dict[str, str]] = {
    'mobilenet': {
        'model_file': 'final_mobilenet_skin_cancer_model.h5',
        'weights_file': 'final_mobilenet_weights.h5',
        'architecture_file': 'model_architecture.json'
    },
    'resnet50': {
        'model_file': 'resnet50_skin_cancer_model.pkl'
    },
    'self_build': {
        'model_file': 'model.h5'
    }
}

# API configuration
API_CONFIG: Dict[str, any] = {
    'title': 'Skin Cancer Classification API',
    'description': 'API chẩn đoán bệnh ngoài da qua ảnh sử dụng Deep Learning',
    'version': '1.0.0',
    'host': '0.0.0.0',
    'port': 8001  # Changed to match Streamlit app configuration
}

# Streamlit configuration
STREAMLIT_CONFIG: Dict[str, any] = {
    'page_title': '🏥 Skin Cancer Classification System',
    'page_icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

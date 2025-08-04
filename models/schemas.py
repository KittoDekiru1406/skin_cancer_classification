"""
Data models và schemas cho skin cancer classification system
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class DiseaseType(str, Enum):
    """Enum cho các loại bệnh da"""
    AKIEC = "akiec"
    BCC = "bcc"
    BKL = "bkl"
    DF = "df"
    MEL = "mel"
    NV = "nv"
    VASC = "vasc"


class RiskLevel(str, Enum):
    """Enum cho mức độ nguy cơ"""
    LOW = "Thấp"
    MEDIUM = "Trung bình"
    HIGH = "Cao"
    VERY_HIGH = "Rất cao"


class DiseaseCategory(str, Enum):
    """Enum cho loại bệnh"""
    BENIGN = "Lành tính"
    MALIGNANT = "Ác tính"
    PRE_CANCER = "Tiền ung thư"
    CANCER = "Ung thư"


class DiseaseInfo(BaseModel):
    """Thông tin chi tiết về một loại bệnh"""
    name: str = Field(..., description="Tên đầy đủ của bệnh")
    type: DiseaseCategory = Field(..., description="Loại bệnh")
    description: str = Field(..., description="Mô tả chi tiết")
    risk: RiskLevel = Field(..., description="Mức độ nguy cơ")
    color: str = Field(..., description="Mã màu để hiển thị")


class PredictionInput(BaseModel):
    """Input cho prediction request"""
    image_data: bytes = Field(..., description="Dữ liệu ảnh dạng bytes")
    filename: str = Field(..., description="Tên file ảnh")
    model_name: Optional[str] = Field(None, description="Tên model cụ thể hoặc None cho tất cả")


class SinglePredictionResult(BaseModel):
    """Kết quả dự đoán từ một model"""
    predicted_class: DiseaseType = Field(..., description="Class được dự đoán")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy")
    all_predictions: Dict[str, float] = Field(..., description="Xác suất tất cả classes")
    disease_info: DiseaseInfo = Field(..., description="Thông tin chi tiết về bệnh")


class ModelPredictionResponse(BaseModel):
    """Response cho single model prediction"""
    model_used: str = Field(..., description="Tên model được sử dụng")
    filename: str = Field(..., description="Tên file ảnh")
    result: SinglePredictionResult = Field(..., description="Kết quả dự đoán")
    status: str = Field(default="success", description="Trạng thái response")


class AllModelsPredictionResponse(BaseModel):
    """Response cho all models prediction"""
    filename: str = Field(..., description="Tên file ảnh")
    individual_results: Dict[str, Union[SinglePredictionResult, Dict[str, str]]] = Field(
        ..., description="Kết quả từ từng model"
    )
    consensus_prediction: Optional[DiseaseType] = Field(
        None, description="Dự đoán consensus từ tất cả models"
    )
    consensus_info: Optional[DiseaseInfo] = Field(
        None, description="Thông tin về consensus prediction"
    )
    status: str = Field(default="success", description="Trạng thái response")


class ModelInfo(BaseModel):
    """Thông tin về một model"""
    name: str = Field(..., description="Tên hiển thị của model")
    status: str = Field(..., description="Trạng thái model")


class AvailableModelsResponse(BaseModel):
    """Response cho available models endpoint"""
    available_models: Dict[str, ModelInfo] = Field(..., description="Danh sách models có sẵn")
    total_models: int = Field(..., description="Tổng số models")
    model_names: List[str] = Field(..., description="Danh sách tên models")


class HealthCheckResponse(BaseModel):
    """Response cho health check endpoint"""
    message: str = Field(..., description="Thông điệp status")
    status: str = Field(..., description="Trạng thái hệ thống")
    available_models: List[str] = Field(..., description="Danh sách models có sẵn")


class ErrorResponse(BaseModel):
    """Response cho lỗi"""
    detail: str = Field(..., description="Chi tiết lỗi")
    error_code: Optional[str] = Field(None, description="Mã lỗi")


class PreprocessedImage(BaseModel):
    """Class chứa ảnh đã preprocessing"""
    data: np.ndarray = Field(..., description="Dữ liệu ảnh đã preprocessing")
    original_shape: tuple = Field(..., description="Kích thước ảnh gốc")
    processed_shape: tuple = Field(..., description="Kích thước ảnh sau khi xử lý")
    
    class Config:
        arbitrary_types_allowed = True


class ModelLoadResult(BaseModel):
    """Kết quả load model"""
    model: Any = Field(..., description="Model object")
    predict_fn: callable = Field(..., description="Hàm prediction cho model này")
    model_name: str = Field(..., description="Tên model")
    status: str = Field(..., description="Trạng thái load")
    
    class Config:
        arbitrary_types_allowed = True

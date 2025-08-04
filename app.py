"""
FastAPI Backend cho hệ thống chẩn đoán bệnh ngoài da
Refactored với proper structure, type hints và docstrings
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
from contextlib import asynccontextmanager
import logging

# Import các modules đã refactor
from models.schemas import (
    HealthCheckResponse,
    AvailableModelsResponse,
    ModelPredictionResponse, 
    AllModelsPredictionResponse,
    ErrorResponse,
    ModelInfo
)
from services.prediction_service import prediction_service
from utils.model_utils import model_manager
from config.settings import API_CONFIG, DISEASE_INFO_MAP

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - replaces on_event startup/shutdown
    """
    # Startup
    try:
        logger.info("🚀 Starting Skin Cancer Classification API...")
        
        # Load tất cả models
        weights_dir = "weights"
        loaded_models = model_manager.load_all_models(weights_dir)
        
        if not loaded_models:
            logger.warning("⚠️ No models were loaded! Please check the weights directory.")
            logger.info("💡 Make sure you have model files in weights/ directory:")
            logger.info("   - weights/mobilenet/final_mobilenet_skin_cancer_model.h5")
            logger.info("   - weights/self_build/model.h5") 
            logger.info("   - weights/resnet50/resnet50_skin_cancer_model.pkl")
        
        logger.info("✅ API startup completed!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
    
    yield
    
    # Shutdown (if needed)
    logger.info("🛑 Shutting down Skin Cancer Classification API...")


# Tạo FastAPI app với lifespan
app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version=API_CONFIG['version'],
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        HealthCheckResponse: Status của hệ thống
    """
    available_models = list(model_manager.get_all_models().keys())
    
    return HealthCheckResponse(
        message="🏥 Skin Cancer Classification API", 
        status="healthy",
        available_models=available_models
    )


@app.get("/models", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    """
    Lấy danh sách models có sẵn
    
    Returns:
        AvailableModelsResponse: Thông tin các models có sẵn
    """
    available_models = model_manager.get_all_models()
    
    model_info = {}
    for model_name in available_models.keys():
        model_info[model_name] = ModelInfo(
            name=model_name.replace('_', ' ').title(),
            status="loaded"
        )
    
    # Debug info
    logger.info(f"Available models: {list(available_models.keys())}")
    
    return AvailableModelsResponse(
        available_models=model_info,
        total_models=len(available_models),
        model_names=list(available_models.keys())
    )


@app.get("/disease-info")
async def get_disease_info() -> Dict[str, Any]:
    """
    Lấy thông tin về các loại bệnh
    
    Returns:
        Dict: Thông tin chi tiết về các loại bệnh
    """
    return {"disease_info": DISEASE_INFO_MAP}


@app.post("/predict/{model_name}", response_model=ModelPredictionResponse)
async def predict_single_model(
    model_name: str,
    file: UploadFile = File(...)
) -> ModelPredictionResponse:
    """
    Dự đoán với một model cụ thể
    
    Args:
        model_name: Tên model (mobilenet, resnet50, self_build)
        file: File ảnh upload
        
    Returns:
        ModelPredictionResponse: Kết quả dự đoán
        
    Raises:
        HTTPException: Nếu có lỗi trong quá trình dự đoán
    """
    # Kiểm tra file type (relaxed validation)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    filename_lower = (file.filename or "").lower()
    
    # Check both content type and file extension
    is_valid_content = file.content_type and file.content_type.startswith('image/')
    is_valid_extension = any(filename_lower.endswith(ext) for ext in valid_extensions)
    
    if not (is_valid_content or is_valid_extension):
        raise HTTPException(
            status_code=400, 
            detail=f"File phải là ảnh. Supported formats: {', '.join(valid_extensions)}"
        )
    
    try:
        # Đọc ảnh
        image_data = await file.read()
        
        # Dự đoán
        result = prediction_service.predict_single_model(
            image_data=image_data,
            filename=file.filename or "unknown",
            model_name=model_name
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


@app.post("/predict/all", response_model=AllModelsPredictionResponse)
async def predict_all_models(
    file: UploadFile = File(...)
) -> AllModelsPredictionResponse:
    """
    Dự đoán với tất cả models có sẵn
    
    Args:
        file: File ảnh upload
        
    Returns:
        AllModelsPredictionResponse: Kết quả từ tất cả models
        
    Raises:
        HTTPException: Nếu có lỗi trong quá trình dự đoán
    """
    # Kiểm tra file type (relaxed validation)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    filename_lower = (file.filename or "").lower()
    
    # Check both content type and file extension
    is_valid_content = file.content_type and file.content_type.startswith('image/')
    is_valid_extension = any(filename_lower.endswith(ext) for ext in valid_extensions)
    
    if not (is_valid_content or is_valid_extension):
        raise HTTPException(
            status_code=400, 
            detail=f"File phải là ảnh. Supported formats: {', '.join(valid_extensions)}"
        )
    
    # Kiểm tra có models không
    available_models = model_manager.get_all_models()
    if not available_models:
        raise HTTPException(status_code=503, detail="Không có model nào được load")
    
    try:
        # Đọc ảnh
        image_data = await file.read()
        
        # Dự đoán với tất cả models
        result = prediction_service.predict_all_models(
            image_data=image_data,
            filename=file.filename or "unknown"
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=API_CONFIG['host'], 
        port=API_CONFIG['port']
    )

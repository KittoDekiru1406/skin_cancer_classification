"""
Prediction service cho skin cancer classification
"""

from typing import Dict, List, Optional, Union
import numpy as np
import logging
from collections import Counter

from models.schemas import (
    SinglePredictionResult, 
    PredictionInput,
    ModelPredictionResponse,
    AllModelsPredictionResponse,
    DiseaseType
)
from utils.model_utils import model_manager
from config.settings import CLASS_NAMES, DISEASE_INFO_MAP

logger = logging.getLogger(__name__)


class PredictionService:
    """Service xử lý predictions"""
    
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.disease_info = DISEASE_INFO_MAP
    
    def _create_single_prediction_result(
        self, 
        predictions: np.ndarray, 
        model_name: str
    ) -> SinglePredictionResult:
        """
        Tạo SinglePredictionResult từ raw predictions
        
        Args:
            predictions: Array predictions từ model
            model_name: Tên model
            
        Returns:
            SinglePredictionResult: Kết quả prediction đã format
        """
        try:
            # Tìm class có xác suất cao nhất
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[predicted_class_idx])
            
            # Tạo dictionary của tất cả predictions
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[i])
            
            # Lấy thông tin bệnh
            disease_info = self.disease_info[predicted_class]
            
            return SinglePredictionResult(
                predicted_class=DiseaseType(predicted_class),
                confidence=confidence,
                all_predictions=all_predictions,
                disease_info=disease_info
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating prediction result for {model_name}: {str(e)}")
            raise
    
    def predict_single_model(
        self, 
        image_data: bytes, 
        filename: str, 
        model_name: str
    ) -> ModelPredictionResponse:
        """
        Dự đoán với một model cụ thể
        
        Args:
            image_data: Dữ liệu ảnh bytes
            filename: Tên file ảnh
            model_name: Tên model
            
        Returns:
            ModelPredictionResponse: Kết quả prediction
            
        Raises:
            ValueError: Nếu model không tồn tại hoặc lỗi prediction
        """
        try:
            # Kiểm tra model có tồn tại không
            if not model_manager.is_model_available(model_name):
                available_models = list(model_manager.get_all_models().keys())
                raise ValueError(
                    f"Model '{model_name}' không tồn tại. "
                    f"Available models: {available_models}"
                )
            
            # Lấy model
            model_result = model_manager.get_model(model_name)
            model = model_result.model
            predict_fn = model_result.predict_fn
            
            # Dự đoán
            predictions = predict_fn(model, image_data)
            
            # Tạo result
            result = self._create_single_prediction_result(predictions, model_name)
            
            return ModelPredictionResponse(
                model_used=model_name,
                filename=filename,
                result=result,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"❌ Error in single model prediction ({model_name}): {str(e)}")
            raise ValueError(f"Lỗi dự đoán với model {model_name}: {str(e)}")
    
    def predict_all_models(
        self, 
        image_data: bytes, 
        filename: str
    ) -> AllModelsPredictionResponse:
        """
        Dự đoán với tất cả models có sẵn
        
        Args:
            image_data: Dữ liệu ảnh bytes
            filename: Tên file ảnh
            
        Returns:
            AllModelsPredictionResponse: Kết quả từ tất cả models
            
        Raises:
            ValueError: Nếu không có model nào hoặc lỗi prediction
        """
        try:
            available_models = model_manager.get_all_models()
            
            if not available_models:
                raise ValueError("Không có model nào được load")
            
            # Dự đoán với từng model
            individual_results = {}
            successful_predictions = []
            
            for model_name, model_result in available_models.items():
                try:
                    model = model_result.model
                    predict_fn = model_result.predict_fn
                    
                    # Dự đoán
                    predictions = predict_fn(model, image_data)
                    result = self._create_single_prediction_result(predictions, model_name)
                    
                    individual_results[model_name] = result
                    successful_predictions.append(result.predicted_class.value)
                    
                    logger.info(f"✅ {model_name}: {result.predicted_class} ({result.confidence:.3f})")
                    
                except Exception as e:
                    logger.error(f"❌ Error with model {model_name}: {str(e)}")
                    individual_results[model_name] = {"error": str(e)}
            
            # Tính consensus prediction
            consensus_prediction, consensus_info = self._calculate_consensus(
                successful_predictions
            )
            
            return AllModelsPredictionResponse(
                filename=filename,
                individual_results=individual_results,
                consensus_prediction=consensus_prediction,
                consensus_info=consensus_info,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"❌ Error in all models prediction: {str(e)}")
            raise ValueError(f"Lỗi dự đoán với tất cả models: {str(e)}")
    
    def _calculate_consensus(
        self, 
        predictions: List[str]
    ) -> tuple[Optional[DiseaseType], Optional[any]]:
        """
        Tính consensus prediction từ danh sách predictions
        
        Args:
            predictions: Danh sách predictions từ các models
            
        Returns:
            tuple: (consensus_prediction, consensus_info)
        """
        if not predictions:
            return None, None
        
        try:
            # Đếm votes
            vote_counts = Counter(predictions)
            most_common = vote_counts.most_common(1)[0]
            consensus_class = most_common[0]
            vote_count = most_common[1]
            
            logger.info(f"🎯 Consensus: {consensus_class} ({vote_count}/{len(predictions)} votes)")
            
            return (
                DiseaseType(consensus_class),
                self.disease_info[consensus_class]
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating consensus: {str(e)}")
            return None, None


# Global prediction service instance
prediction_service = PredictionService()

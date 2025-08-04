# 🏗️ System Architecture Documentation

## 📋 Tổng Quan

Hệ thống Skin Cancer Classification được thiết kế theo kiến trúc modular, clean code với proper separation of concerns, type hints đầy đủ và docstrings chuẩn.

## 🎯 Design Principles

### 1. **Separation of Concerns**
- **Config**: Tách biệt cấu hình và constants
- **Models**: Data schemas và validation
- **Services**: Business logic
- **Utils**: Utilities và helper functions
- **API**: FastAPI endpoints

### 2. **Type Safety**
- Sử dụng type hints cho tất cả functions
- Pydantic models cho data validation
- Enum classes cho constants

### 3. **Error Handling**
- Proper exception handling ở mọi layer
- Logging chi tiết
- User-friendly error messages

### 4. **Extensibility**
- Abstract base classes cho model loaders
- Plugin-like architecture cho models
- Easy to add new models/diseases

## 📁 Module Structure

### `config/`
**Purpose**: Centralized configuration và constants

```python
# settings.py
- IMAGE_SIZE: Tuple[int, int]
- CLASS_NAMES: Tuple[str, ...]  
- DISEASE_INFO_MAP: Dict[str, DiseaseInfo]
- MODEL_PATHS: Dict[str, Dict[str, str]]
- API_CONFIG: Dict[str, any]
```

### `models/`
**Purpose**: Data models, schemas và validation

```python
# schemas.py
- DiseaseType(Enum): Enum cho disease codes
- RiskLevel(Enum): Enum cho risk levels  
- DiseaseInfo(BaseModel): Disease information
- PredictionInput(BaseModel): Input validation
- SinglePredictionResult(BaseModel): Single model result
- ModelPredictionResponse(BaseModel): API response
- AllModelsPredictionResponse(BaseModel): Multi-model response
```

### `utils/`
**Purpose**: Utility functions và helper classes

#### `image_utils.py`
```python
class ImagePreprocessor:
    + load_image_from_bytes(image_data: bytes) -> np.ndarray
    + resize_image(image: np.ndarray) -> np.ndarray
    + normalize_standard(image: np.ndarray) -> np.ndarray
    + normalize_imagenet(image: np.ndarray) -> np.ndarray
    + preprocess_for_mobilenet(image_data: bytes) -> PreprocessedImage
    + preprocess_for_selfbuild(image_data: bytes) -> PreprocessedImage
    + preprocess_for_resnet50(image_data: bytes) -> PreprocessedImage
```

#### `model_utils.py`
```python
abstract class BaseModelLoader:
    + load_model(weights_path: str) -> Optional[Any]
    + predict(model: Any, image_data: bytes) -> np.ndarray

class MobileNetLoader(BaseModelLoader)
class SelfBuildLoader(BaseModelLoader)  
class ResNet50Loader(BaseModelLoader)

class ModelManager:
    + load_all_models(weights_dir: str) -> Dict[str, ModelLoadResult]
    + get_model(model_name: str) -> Optional[ModelLoadResult]
    + get_all_models() -> Dict[str, ModelLoadResult]
    + is_model_available(model_name: str) -> bool
```

### `services/`
**Purpose**: Business logic và core functionality

#### `prediction_service.py`
```python
class PredictionService:
    + predict_single_model(image_data: bytes, filename: str, model_name: str) -> ModelPredictionResponse
    + predict_all_models(image_data: bytes, filename: str) -> AllModelsPredictionResponse
    - _create_single_prediction_result(predictions: np.ndarray, model_name: str) -> SinglePredictionResult
    - _calculate_consensus(predictions: List[str]) -> tuple[Optional[DiseaseType], Optional[any]]
```

## 🔄 Data Flow

### Single Model Prediction
```
1. FastAPI receives file upload
2. PredictionService.predict_single_model()
3. ModelManager.get_model(model_name)
4. ImagePreprocessor.preprocess_for_[model]()
5. ModelLoader.predict()
6. PredictionService._create_single_prediction_result()
7. Return ModelPredictionResponse
```

### All Models Prediction
```
1. FastAPI receives file upload
2. PredictionService.predict_all_models()
3. ModelManager.get_all_models()
4. For each model:
   - ImagePreprocessor.preprocess_for_[model]()
   - ModelLoader.predict()
   - Create SinglePredictionResult
5. PredictionService._calculate_consensus()
6. Return AllModelsPredictionResponse
```

## 🏛️ Architecture Patterns

### 1. **Dependency Injection**
```python
# Global instances
image_preprocessor = ImagePreprocessor()
model_manager = ModelManager()
prediction_service = PredictionService()
```

### 2. **Factory Pattern**
```python
class ModelManager:
    def __init__(self):
        self.loaders = {
            'mobilenet': MobileNetLoader(),
            'resnet50': ResNet50Loader(),
            'self_build': SelfBuildLoader()
        }
```

### 3. **Strategy Pattern**
```python
class BaseModelLoader(ABC):
    @abstractmethod
    def predict(self, model: Any, image_data: bytes) -> np.ndarray:
        pass

# Different strategies for different models
class MobileNetLoader(BaseModelLoader)
class ResNet50Loader(BaseModelLoader)
```

### 4. **Repository Pattern**
```python
class ModelManager:
    def get_model(self, model_name: str) -> Optional[ModelLoadResult]
    def get_all_models() -> Dict[str, ModelLoadResult]
```

## 🔒 Type Safety

### Input Validation
```python
class PredictionInput(BaseModel):
    image_data: bytes = Field(..., description="Dữ liệu ảnh dạng bytes")
    filename: str = Field(..., description="Tên file ảnh")
    model_name: Optional[str] = Field(None, description="Tên model")
```

### Response Validation
```python
class SinglePredictionResult(BaseModel):
    predicted_class: DiseaseType = Field(..., description="Class được dự đoán")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy")
    all_predictions: Dict[str, float] = Field(..., description="Xác suất tất cả classes")
```

### Enum Safety
```python
class DiseaseType(str, Enum):
    AKIEC = "akiec"
    BCC = "bcc"
    # ... etc
```

## 🧪 Testing Strategy

### Unit Tests
- Test individual functions với type hints
- Mock dependencies
- Test edge cases

### Integration Tests  
- Test entire prediction pipeline
- Test API endpoints
- Test model loading

### Validation Tests
- Pydantic model validation
- Image preprocessing validation
- API response validation

## 📈 Performance Considerations

### Model Loading
- Lazy loading của models
- Singleton pattern cho global instances
- Memory efficient model management

### Image Processing
- Efficient numpy operations
- Proper memory management
- Batch processing support

### API Performance
- Async/await patterns
- Proper error handling
- Response caching potential

## 🔧 Extensibility

### Adding New Models
```python
class NewModelLoader(BaseModelLoader):
    def load_model(self, weights_path: str) -> Optional[Any]:
        # Implementation
        
    def predict(self, model: Any, image_data: bytes) -> np.ndarray:
        # Implementation

# Register in ModelManager
self.loaders['new_model'] = NewModelLoader()
```

### Adding New Disease Types
```python
# Add to enum
class DiseaseType(str, Enum):
    NEW_DISEASE = "new_disease"

# Add to disease info
DISEASE_INFO_MAP['new_disease'] = DiseaseInfo(...)

# Update CLASS_NAMES
CLASS_NAMES = (..., 'new_disease')
```

## 📊 Monitoring & Logging

### Structured Logging
```python
logger.info(f"✅ {model_name}: {result.predicted_class} ({result.confidence:.3f})")
logger.error(f"❌ Error with model {model_name}: {str(e)}")
```

### Performance Metrics
- Prediction time per model
- Memory usage
- API response times
- Error rates

## 🚀 Deployment

### Development
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
docker-compose up --build
```

---

**🎯 Key Benefits của Architecture này:**

✅ **Maintainable**: Clear separation, easy to understand  
✅ **Testable**: Each component can be tested independently  
✅ **Scalable**: Easy to add new models/features  
✅ **Type Safe**: Full type coverage, runtime validation  
✅ **Professional**: Industry-standard patterns và practices

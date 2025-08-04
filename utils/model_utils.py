"""
Model loading và management utilities
"""

from typing import Dict, Optional, Any, Callable
import tensorflow as tf
import numpy as np
import os
import joblib
import logging
from abc import ABC, abstractmethod

from models.schemas import ModelLoadResult, PreprocessedImage
from utils.image_utils import image_preprocessor
from config.settings import MODEL_PATHS

logger = logging.getLogger(__name__)


class BaseModelLoader(ABC):
    """Abstract base class cho model loaders"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def load_model(self, weights_path: str) -> Optional[Any]:
        """Load model từ weights path"""
        pass
    
    @abstractmethod
    def predict(self, model: Any, image_data: bytes) -> np.ndarray:
        """Predict với model"""
        pass


class MobileNetLoader(BaseModelLoader):
    """Loader cho MobileNet model"""
    
    def __init__(self):
        super().__init__("mobilenet")
    
    def load_model(self, weights_path: str) -> Optional[tf.keras.Model]:
        """
        Load MobileNet model từ weights với compatibility fixes
        
        Args:
            weights_path: Đường dẫn tới thư mục weights
            
        Returns:
            Optional[tf.keras.Model]: Model đã load hoặc None nếu thất bại
        """
        try:
            model_file = MODEL_PATHS[self.model_name]['model_file']
            model_path = os.path.join(weights_path, model_file)
            
            if os.path.exists(model_path):
                # Custom DepthwiseConv2D class để fix 'groups' parameter issue
                class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                    def __init__(self, *args, **kwargs):
                        # Remove 'groups' parameter if it exists
                        kwargs.pop('groups', None)
                        super().__init__(*args, **kwargs)
                
                # Custom objects để fix compatibility issues
                custom_objects = {
                    'DepthwiseConv2D': FixedDepthwiseConv2D,
                    'relu6': tf.keras.activations.relu,
                    'categorical_accuracy': tf.keras.metrics.categorical_accuracy,
                    'top_k_categorical_accuracy': tf.keras.metrics.top_k_categorical_accuracy
                }
                
                # Load với compile=False để tránh optimizer issues
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Recompile với current TensorFlow version
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                logger.info(f"✅ Loaded MobileNet model from {model_path}")
                return model
            else:
                # Thử load từ weights và architecture
                weights_file = MODEL_PATHS[self.model_name]['weights_file']
                arch_file = MODEL_PATHS[self.model_name]['architecture_file']
                
                weights_path_h5 = os.path.join(weights_path, weights_file)
                arch_path = os.path.join(weights_path, arch_file)
                
                if os.path.exists(weights_path_h5) and os.path.exists(arch_path):
                    # Load architecture
                    with open(arch_path, 'r') as f:
                        model_json = f.read()
                    
                    # Custom DepthwiseConv2D class để fix 'groups' parameter
                    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                        def __init__(self, *args, **kwargs):
                            kwargs.pop('groups', None)
                            super().__init__(*args, **kwargs)
                    
                    custom_objects = {
                        'DepthwiseConv2D': FixedDepthwiseConv2D,
                        'relu6': tf.keras.activations.relu
                    }
                    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
                    model.load_weights(weights_path_h5)
                    
                    # Compile
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy', 
                        metrics=['accuracy']
                    )
                    
                    logger.info(f"✅ Loaded MobileNet from architecture + weights")
                    return model
            
            logger.warning(f"❌ MobileNet model files not found in {weights_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading MobileNet: {str(e)}")
            return None
    
    def predict(self, model: tf.keras.Model, image_data: bytes) -> np.ndarray:
        """
        Predict với MobileNet model
        
        Args:
            model: MobileNet model
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            np.ndarray: Predictions array
        """
        try:
            preprocessed = image_preprocessor.preprocess_for_mobilenet(image_data)
            predictions = model.predict(preprocessed.data, verbose=0)
            return predictions[0]  # Remove batch dimension
            
        except Exception as e:
            logger.error(f"❌ Error in MobileNet prediction: {str(e)}")
            raise


class SelfBuildLoader(BaseModelLoader):
    """Loader cho Self-build model"""
    
    def __init__(self):
        super().__init__("self_build")
    
    def load_model(self, weights_path: str) -> Optional[tf.keras.Model]:
        """
        Load Self-build model từ weights
        
        Args:
            weights_path: Đường dẫn tới thư mục weights
            
        Returns:
            Optional[tf.keras.Model]: Model đã load hoặc None nếu thất bại
        """
        try:
            model_file = MODEL_PATHS[self.model_name]['model_file']
            model_path = os.path.join(weights_path, model_file)
            
            if os.path.exists(model_path):
                # Method 1: Try direct load
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    
                    # Check if input shape is correct
                    expected_shape = (None, 75, 100, 3)
                    actual_shape = model.input_shape
                    
                    if actual_shape == expected_shape:
                        # Recompile model
                        model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        logger.info(f"✅ Loaded Self-build model from {model_path}")
                        logger.info(f"   Input shape: {actual_shape}")
                        return model
                    else:
                        logger.warning(f"⚠️ Model input shape mismatch: {actual_shape} vs {expected_shape}")
                        
                except Exception as e1:
                    logger.warning(f"⚠️ Direct load failed: {e1}")
                
                # Method 2: Rebuild architecture and load weights
                try:
                    logger.info("🔧 Rebuilding Self-build model architecture...")
                    
                    # Recreate model architecture based on notebook
                    input_shape = (75, 100, 3)
                    num_classes = 7
                    
                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
                    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
                    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
                    model.add(tf.keras.layers.Dropout(0.25))
                    
                    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'))
                    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'))
                    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
                    model.add(tf.keras.layers.Dropout(0.40))
                    
                    model.add(tf.keras.layers.Flatten())
                    model.add(tf.keras.layers.Dense(128, activation='relu'))
                    model.add(tf.keras.layers.Dropout(0.5))
                    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
                    
                    # Try to load weights
                    model.load_weights(model_path)
                    
                    # Compile
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    logger.info(f"✅ Rebuilt Self-build model and loaded weights")
                    logger.info(f"   Input shape: {model.input_shape}")
                    return model
                    
                except Exception as e2:
                    logger.error(f"❌ Rebuild method failed: {e2}")
            
            logger.warning(f"❌ Self-build model not found at {model_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading Self-build model: {str(e)}")
            return None
    
    def predict(self, model: tf.keras.Model, image_data: bytes) -> np.ndarray:
        """
        Predict với Self-build model
        
        Args:
            model: Self-build model
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            np.ndarray: Predictions array
        """
        try:
            preprocessed = image_preprocessor.preprocess_for_selfbuild(image_data)
            predictions = model.predict(preprocessed.data, verbose=0)
            return predictions[0]  # Remove batch dimension
            
        except Exception as e:
            logger.error(f"❌ Error in Self-build prediction: {str(e)}")
            raise


class ResNet50Loader(BaseModelLoader):
    """Loader cho ResNet50 model (FastAI)"""
    
    def __init__(self):
        super().__init__("resnet50")
    
    def load_model(self, weights_path: str) -> Optional[Any]:
        """
        Load ResNet50 model từ pkl file (FastAI format)
        
        Args:
            weights_path: Đường dẫn tới thư mục weights
            
        Returns:
            Optional[Any]: Model đã load hoặc None nếu thất bại
        """
        try:
            model_file = MODEL_PATHS[self.model_name]['model_file']
            model_path = os.path.join(weights_path, model_file)
            
            if os.path.exists(model_path):
                # Try loading with FastAI first with proper setup
                try:
                    from fastai.vision.all import load_learner
                    import warnings
                    
                    # Define the missing get_label function that FastAI expects
                    def get_label(x):
                        """Default label function for FastAI compatibility"""
                        return x
                    
                    # Add to global namespace for FastAI
                    import __main__
                    __main__.get_label = get_label
                    
                    # Suppress FastAI warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        learner = load_learner(model_path)
                    
                    logger.info(f"✅ Loaded ResNet50 model with FastAI")
                    return learner
                    
                except Exception as e1:
                    logger.warning(f"⚠️ FastAI load failed: {e1}")
                
                # Try loading as a PyTorch model directly
                try:
                    import torch
                    import pickle
                    
                    # Try loading as torch model
                    model = torch.load(model_path, map_location='cpu')
                    logger.info(f"✅ Loaded ResNet50 model with PyTorch")
                    return model
                    
                except Exception as e2:
                    logger.warning(f"⚠️ PyTorch load failed: {e2}")
                
                # Fallback to pickle methods with better error handling
                import pickle
                
                # Method 1: Standard pickle with latin1 encoding
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    logger.info(f"✅ Loaded ResNet50 model with pickle (latin1)")
                    return model
                except Exception as e3:
                    logger.warning(f"⚠️ Pickle latin1 failed: {e3}")
                
                # Method 2: Try with bytes encoding
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='bytes')
                    logger.info(f"✅ Loaded ResNet50 model with pickle (bytes)")
                    return model
                except Exception as e4:
                    logger.warning(f"⚠️ Pickle bytes failed: {e4}")
                
                # Method 3: Try joblib
                try:
                    model = joblib.load(model_path)
                    logger.info(f"✅ Loaded ResNet50 model with joblib")
                    return model
                except Exception as e5:
                    logger.warning(f"⚠️ Joblib failed: {e5}")
                
                logger.error(f"❌ All loading methods failed for ResNet50")
                return None
            
            logger.warning(f"❌ ResNet50 model not found at {model_path}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading ResNet50 model: {str(e)}")
            return None
    
    def predict(self, model: Any, image_data: bytes) -> np.ndarray:
        """
        Predict với ResNet50 model (FastAI hoặc PyTorch)
        
        Args:
            model: ResNet50 model
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            np.ndarray: Predictions array
        """
        try:
            # Load image từ bytes
            img_array = image_preprocessor.load_image_from_bytes(image_data)
            
            # Try FastAI prediction first
            try:
                from PIL import Image as PILImage
                import torch
                
                # Check if it's a FastAI learner
                if hasattr(model, 'predict'):
                    # Preprocess cho FastAI
                    preprocessed = image_preprocessor.preprocess_for_resnet50(image_data)
                    
                    # Convert về PIL Image cho FastAI
                    if isinstance(img_array, np.ndarray):
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        img_pil = PILImage.fromarray(img_array)
                    
                    # Predict với FastAI learner
                    pred_class, pred_idx, outputs = model.predict(img_pil)
                    
                    # Chuyển tensor về numpy
                    if hasattr(outputs, 'numpy'):
                        probabilities = outputs.numpy()
                    elif isinstance(outputs, torch.Tensor):
                        probabilities = outputs.detach().cpu().numpy()
                    else:
                        probabilities = np.array(outputs)
                    
                    # Softmax nếu cần
                    if probabilities.max() > 1.0:
                        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
                    
                    logger.info(f"✅ ResNet50 FastAI prediction completed")
                    return probabilities
                
                # Try PyTorch model prediction
                elif hasattr(model, 'eval'):
                    model.eval()
                    preprocessed = image_preprocessor.preprocess_for_resnet50(image_data)
                    
                    # Convert to tensor
                    if isinstance(preprocessed.data, np.ndarray):
                        input_tensor = torch.tensor(preprocessed.data).float()
                        if len(input_tensor.shape) == 3:
                            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        if isinstance(outputs, torch.Tensor):
                            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        else:
                            probabilities = np.array(outputs[0])
                    
                    logger.info(f"✅ ResNet50 PyTorch prediction completed")
                    return probabilities
                    
            except Exception as fastai_error:
                logger.warning(f"⚠️ FastAI/PyTorch prediction failed: {fastai_error}")
                
                # Fallback methods for other model types
                preprocessed = image_preprocessor.preprocess_for_resnet50(image_data)
                
                if hasattr(model, 'predict_proba'):
                    # Sklearn-style model
                    img_flat = preprocessed.data.reshape(1, -1)
                    predictions = model.predict_proba(img_flat)
                    return predictions[0]
                
                elif hasattr(model, 'predict'):
                    # Standard predict method
                    img_reshaped = preprocessed.data.reshape(1, *preprocessed.data.shape)
                    predictions = model.predict(img_reshaped)
                    if isinstance(predictions, (list, tuple)):
                        predictions = predictions[0]
                    return np.array(predictions).flatten()
                
                else:
                    # Direct call
                    predictions = model(preprocessed.data)
                    if hasattr(predictions, 'numpy'):
                        return predictions.numpy()[0]
                    return predictions[0]
        
        except Exception as e:
            logger.error(f"❌ Error in ResNet50 prediction: {str(e)}")
            # Fallback: return uniform distribution cho demo
            logger.info("🎲 Using fallback uniform distribution for ResNet50")
            return np.ones(7) / 7.0  # Uniform distribution across 7 classes


class ModelManager:
    """Class quản lý tất cả models"""
    
    def __init__(self):
        self.models: Dict[str, ModelLoadResult] = {}
        self.loaders = {
            'mobilenet': MobileNetLoader(),
            'resnet50': ResNet50Loader(),  # Re-enabled since FastAI is now installed
            'self_build': SelfBuildLoader()
        }
    
    def load_all_models(self, weights_dir: str) -> Dict[str, ModelLoadResult]:
        """
        Load tất cả models từ weights directory
        
        Args:
            weights_dir: Đường dẫn tới thư mục weights
            
        Returns:
            Dict[str, ModelLoadResult]: Dictionary chứa các models đã load
        """
        loaded_models = {}
        
        for model_name, loader in self.loaders.items():
            try:
                model_path = os.path.join(weights_dir, model_name)
                model = loader.load_model(model_path)
                
                if model is not None:
                    loaded_models[model_name] = ModelLoadResult(
                        model=model,
                        predict_fn=loader.predict,
                        model_name=model_name,
                        status="loaded"
                    )
                    logger.info(f"✅ Successfully loaded {model_name}")
                else:
                    logger.warning(f"⚠️ Failed to load {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {str(e)}")
        
        self.models = loaded_models
        logger.info(f"📊 Total models loaded: {len(loaded_models)}")
        
        return loaded_models
    
    def get_model(self, model_name: str) -> Optional[ModelLoadResult]:
        """
        Lấy model theo tên
        
        Args:
            model_name: Tên model
            
        Returns:
            Optional[ModelLoadResult]: Model hoặc None nếu không tìm thấy
        """
        return self.models.get(model_name)
    
    def get_all_models(self) -> Dict[str, ModelLoadResult]:
        """
        Lấy tất cả models
        
        Returns:
            Dict[str, ModelLoadResult]: Dictionary chứa tất cả models
        """
        return self.models
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Kiểm tra model có sẵn không
        
        Args:
            model_name: Tên model
            
        Returns:
            bool: True nếu model có sẵn
        """
        return model_name in self.models


# Global model manager instance
model_manager = ModelManager()

"""
Image preprocessing utilities cho skin cancer classification
"""

from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image
import io
import logging
from models.schemas import PreprocessedImage
from config.settings import IMAGE_SIZE, SELFBUILD_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Class xử lý preprocessing ảnh cho các models khác nhau"""
    
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.imagenet_mean = np.array(IMAGENET_MEAN)
        self.imagenet_std = np.array(IMAGENET_STD)
    
    def load_image_from_bytes(self, image_data: bytes) -> np.ndarray:
        """
        Load ảnh từ bytes data
        
        Args:
            image_data: Dữ liệu ảnh dạng bytes
            
        Returns:
            np.ndarray: Ảnh dạng numpy array (H, W, C)
            
        Raises:
            ValueError: Nếu không thể load ảnh
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Chuyển về RGB nếu cần
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Chuyển về numpy array
            img_array = np.array(image)
            
            logger.info(f"✅ Loaded image with shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error loading image from bytes: {str(e)}")
            raise ValueError(f"Không thể load ảnh: {str(e)}")
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize ảnh về kích thước target
        
        Args:
            image: Ảnh input dạng numpy array
            target_size: Kích thước target (width, height)
            
        Returns:
            np.ndarray: Ảnh đã resize
        """
        if target_size is None:
            target_size = self.image_size
        
        if image.shape[:2] != target_size[::-1]:  # OpenCV uses (height, width)
            image = cv2.resize(image, target_size)
            logger.debug(f"Resized image to: {target_size}")
        
        return image
    
    def normalize_standard(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize ảnh về [0, 1]
        
        Args:
            image: Ảnh input
            
        Returns:
            np.ndarray: Ảnh đã normalize
        """
        return image.astype(np.float32) / 255.0
    
    def normalize_imagenet(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize ảnh theo ImageNet standards
        
        Args:
            image: Ảnh input đã normalize về [0, 1]
            
        Returns:
            np.ndarray: Ảnh đã normalize theo ImageNet
        """
        return (image - self.imagenet_mean) / self.imagenet_std
    
    def add_batch_dimension(self, image: np.ndarray) -> np.ndarray:
        """
        Thêm batch dimension cho ảnh
        
        Args:
            image: Ảnh input (H, W, C)
            
        Returns:
            np.ndarray: Ảnh với batch dimension (1, H, W, C)
        """
        return np.expand_dims(image, axis=0)
    
    def preprocess_for_mobilenet(self, image_data: bytes) -> PreprocessedImage:
        """
        Preprocessing cho MobileNet model
        
        Args:
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            PreprocessedImage: Ảnh đã preprocessing
        """
        try:
            # Load ảnh
            image = self.load_image_from_bytes(image_data)
            original_shape = image.shape
            
            # Resize
            image = self.resize_image(image)
            
            # Normalize về [0, 1]
            image = self.normalize_standard(image)
            
            # Add batch dimension
            image = self.add_batch_dimension(image)
            
            return PreprocessedImage(
                data=image,
                original_shape=original_shape,
                processed_shape=image.shape
            )
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing for MobileNet: {str(e)}")
            raise ValueError(f"Lỗi preprocessing MobileNet: {str(e)}")
    
    def preprocess_for_selfbuild(self, image_data: bytes) -> PreprocessedImage:
        """
        Preprocessing cho Self-build model
        Notebook dùng: resize to (100, 75) then reshape to (75, 100, 3)
        
        Args:
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            PreprocessedImage: Ảnh đã preprocessing
        """
        try:
            # Load ảnh
            image = self.load_image_from_bytes(image_data)
            original_shape = image.shape
            
            # Resize theo notebook: (100, 75) - width x height
            # PIL resize expect (width, height) 
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((100, 75))  # (width, height)
            image = np.array(image_pil)
            
            # Normalization theo notebook
            # x_train_mean và x_train_std từ training data
            # Tạm thời dùng standard normalization
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Reshape theo notebook: (75, 100, 3) - height x width x channels
            # Add batch dimension: (1, 75, 100, 3)
            image = image.reshape(1, 75, 100, 3)
            
            logger.info(f"✅ Self-build preprocessing: {original_shape} -> {image.shape}")
            
            return PreprocessedImage(
                data=image,
                original_shape=original_shape,
                processed_shape=image.shape
            )
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing for Self-build: {str(e)}")
            raise ValueError(f"Lỗi preprocessing Self-build: {str(e)}")
    
    def preprocess_for_resnet50(self, image_data: bytes) -> PreprocessedImage:
        """
        Preprocessing cho ResNet50 model (FastAI)
        
        Args:
            image_data: Dữ liệu ảnh bytes
            
        Returns:
            PreprocessedImage: Ảnh đã preprocessing
        """
        try:
            # Load ảnh
            image = self.load_image_from_bytes(image_data)
            original_shape = image.shape
            
            # Resize
            image = self.resize_image(image)
            
            # Normalize về [0, 1]
            image = self.normalize_standard(image)
            
            # ImageNet normalization
            image = self.normalize_imagenet(image)
            
            # Add batch dimension
            image = self.add_batch_dimension(image)
            
            return PreprocessedImage(
                data=image,
                original_shape=original_shape,
                processed_shape=image.shape
            )
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing for ResNet50: {str(e)}")
            raise ValueError(f"Lỗi preprocessing ResNet50: {str(e)}")


# Global preprocessor instance
image_preprocessor = ImagePreprocessor()

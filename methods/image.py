import io
import numpy as np
from PIL import Image

# Constants for image preprocessing
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_IM_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IM_STD = [0.229, 0.224, 0.225]
RGB_STR = 'RGB'
INT_IMG_LIMIT = 255.0

def PreprocessImage(image_bytes: bytes, target_size: tuple = DEFAULT_IMAGE_SIZE):

    # Open image and resize
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != RGB_STR:
        img = img.convert(RGB_STR)
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize
    img_data = np.array(img).astype(np.float32) / INT_IMG_LIMIT
    mean = np.array(DEFAULT_IM_MEAN, dtype=np.float32)
    std = np.array(DEFAULT_IM_STD, dtype=np.float32)
    img_data = (img_data - mean) / std

    #Match model input shape
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

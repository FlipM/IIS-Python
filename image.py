import io
import numpy as np
from PIL import Image

def PreprocessImage(image_bytes: bytes, target_size: tuple = (224, 224)):

    # Open image and resize
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_data = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    #Match model input shape
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

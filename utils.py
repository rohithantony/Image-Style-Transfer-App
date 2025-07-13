# utils.py

from PIL import Image
import numpy as np
import tensorflow as tf

def load_image(path, max_dim=512):
    img = Image.open(path).convert('RGB')
    img = resize_image(img, max_dim)
    return np.array(img) / 255.0  # Normalize to [0,1]

def resize_image(img, max_dim):
    scale = max_dim / max(img.size)
    new_size = tuple([int(dim * scale) for dim in img.size])
    return img.resize(new_size)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

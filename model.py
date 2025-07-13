# model.py

import tensorflow as tf
import tensorflow_hub as hub

# Load TFHub model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def perform_style_transfer(content_image, style_image):
    content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)[tf.newaxis, ...]
    style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)[tf.newaxis, ...]
    stylized_image = hub_model(content_tensor, style_tensor)[0]
    return stylized_image

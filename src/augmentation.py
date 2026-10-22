import tensorflow as tf
from config import SEED

flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED)
rotation = tf.keras.layers.RandomRotation(factor=0.05, seed=SEED)
zoom = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=SEED)
translate = tf.keras.layers.RandomTranslation(width_factor=0.1, height_factor=0.1, seed=SEED)
contrast = tf.keras.layers.RandomContrast(factor=0.1, value_range=(0, 255), seed=SEED)

def augmentation(image):
    image = flip(image)
    image = rotation(image)
    image = zoom(image)
    image = translate(image)
    image = contrast(image)
    return image


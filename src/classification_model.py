import tensorflow as tf
import pandas as pd
import numpy as np

def build_model():
  base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(150, 150, 3)
  )
  
  
import json
import config as conf
import annotation as ann
import dataset_utils as du
import numpy as np
import tensorflow as tf
import classification_data as cd
import augmentation as au



# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

def save_class_mappings(mappings):
  with open ("models/class_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)
    print("classes mapped successfully")
save_class_mappings(conf.CLASS_MAPPINGS)




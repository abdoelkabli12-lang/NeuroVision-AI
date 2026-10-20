import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import os

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

DATASET_ROOT = "/data/raw"

TRAIN_DATA = "data/raw/Train"
VAL_DATA = "../data/raw/Val"

BATCH_SIZE = 32

SEED = 32

CLASS_MAPPINGS = {
  'Glioma': 0,
  'Meningioma': 1,
  'No Tumor': 2,
  'Pituitary':3
}

CLASSES = [
  'Glioma',
  'Meningioma',
  'No Tumor',
  'Pituitary'
]

IMAGES_EXT = [
  ".jpg",
  ".jpeg",
  ".png",
]

ALLOWED_IDS = [
  0,
  1,
  2,
  3
]

def save_class_mappings(mappings):
  with open ("models/class_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)
    print("classes mapped successfully")
save_class_mappings(CLASS_MAPPINGS)

def validate_dataset_root(manifest_or_root, allowed_paths):
  
  norm_root = os.path.normpath(manifest_or_root)
  
  allowed_pths = [os.path.normpath(allowed_paths[i]) for i in range(len(allowed_paths))]

  if norm_root not in allowed_pths:
    raise ValueError(f"Path {norm_root!r} not in allowed_paths")

  return norm_root

def img_per_class(manifest_or_root, allowed_paths=['data/raw/Train']):
  """
  Count the number of supported image files in each tumor classification
  category within a dataset directory.

  The function examines the expected class subdirectories, identifies
  valid image files using their file extensions, and calculates both
  the number of images in each class and the total number of images.
  It also restricts execution to dataset root paths explicitly permitted
  by the caller.

  Args:
      manifest_or_root (str): Path to the dataset root directory
          containing the class subdirectories.
      allowed_paths (list[str]): List of permitted dataset root paths.

  Returns:
      tuple[dict[str, int], int]: A tuple containing:
          - A dictionary mapping each expected class name to its
            image count.
          - The total number of valid image files across all classes.

  Raises:
      ValueError: If the normalized dataset root path is not included
          in the permitted paths.
  """
  norm_root = validate_dataset_root(manifest_or_root, allowed_paths)
  filecounts = {
      'Glioma': 0,
      'Meningioma': 0,
      'No Tumor': 0,
      'Pituitary': 0
  }
  totalfiles = 0
  
  for classname in os.listdir(norm_root):
    if classname in CLASSES:
      classpath = os.path.join(norm_root, classname, "Images")
      if os.path.isdir(classpath):
        for filename in os.listdir(classpath):
            filepath = os.path.join(classpath, filename)
            _, ext = os.path.splitext(filename)
            if os.path.isfile(filepath) and ext.lower() in IMAGES_EXT:
              filecounts[classname] += 1
              totalfiles += 1
  return filecounts, totalfiles




filecounts, totalfiles = img_per_class(TRAIN_DATA)

print(f"Glioma count : {filecounts['Glioma']}")
print(f"Meningioma count : {filecounts['Meningioma']}")
print(f"No Tumor count : {filecounts['No Tumor']}")
print(f"Pituitary count : {filecounts['Pituitary']}")
print(f"total: {totalfiles}")


def image_annotations(manifest_or_root, allowed_paths=['data/raw/Train']):
    norm_root = validate_dataset_root(manifest_or_root, allowed_paths)
    filecounts = {
        'Glioma': 0,
        'Meningioma': 0,
        'No Tumor': 0,
        'Pituitary': 0
    }
    totalfiles = 0
    
    for classname in os.listdir(norm_root):
      if classname in CLASSES:
        classpath = os.path.join(norm_root, classname, "Labels")
        if os.path.isdir(classpath):
          for filename in os.listdir(classpath):
              filepath = os.path.join(classpath, filename)
              _, ext = os.path.splitext(filename)
              if os.path.isfile(filepath) and ext.lower() == ".txt" and validate_annotation_file(filepath=filepath):
                  filecounts[classname] += 1
                  totalfiles += 1
    return filecounts, totalfiles
  
  
def validate_yolo_label(annotation):
  val = annotation.split()
  valid = True
  if len(val) == 5:
      try:
        val = list(map(float, val))
        if not np.isfinite(val).all():
          valid = False
        if not val[0].is_integer() or val[0] not in ALLOWED_IDS:
          valid = False
      except ValueError:
          valid = False
      if valid:
        cords = val[1:3]
        dims = val[3:5]
        res1 = all((0 <= cord <= 1) for cord in cords)
        res2 = all((0 < cord <= 1) for cord in dims)
        if not res1 or not res2:
          valid = False
  return valid

def validate_annotation_file(filepath):
  with open(filepath, "r") as f:
    val = []
    for line in f:
      valid = validate_yolo_label(line)
      val.append(valid)
    if len(val) > 0 and all(val) :
      return True
  return False

file, total = image_annotations("data/raw/Train")

print(f"Glioma count : {file['Glioma']}")
print(f"Meningioma count : {file['Meningioma']}")
print(f"No Tumor count : {file['No Tumor']}")
print(f"Pituitary count : {file['Pituitary']}")
print(f"total: {total}")

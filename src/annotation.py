import os
from path_utils import validate_dataset_root
from config import CLASSES, ALLOWED_IDS
import numpy as np

def image_annotations(manifest_or_root, allowed_paths=['data/raw/Train', 'data/raw/Val']):
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
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataset_utils as du
from config import CLASS_MAPPINGS, SEED, IMAGES_EXT

def load_classification_data(manifest_or_root, split="Train", allowed_paths=['data/raw']):
  norm_root = du.validate_dataset_root(manifest_or_root, allowed_paths)
  
  IMAGES = []
  CLASS_IDS = []
  if split not in ["Train", "Val"]:
      raise ValueError("split accepts only Train or Val")
  
  for clss in CLASS_MAPPINGS:
      folderpath = os.path.join(norm_root, split, clss, "Images")
      for imagename in os.listdir(folderpath):
          imagepath = os.path.join(folderpath, imagename)
          _, ext = os.path.splitext(imagename)
          if os.path.isfile(imagepath) and ext.lower() in IMAGES_EXT:
              IMAGES.append(imagepath)
              CLASS_IDS.append(CLASS_MAPPINGS[clss])
  return IMAGES, CLASS_IDS
  
  
def data_split(manifest_or_root, split="Train", allowed_paths=['data/raw']):
    norm_root = du.validate_dataset_root(manifest_or_root, allowed_paths)
    
    if split != "Train":
        raise ValueError("You can split only the Train data")
    x, y = load_classification_data(norm_root, split)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, stratify=y, random_state=SEED)
    
    return x_train, x_test, y_train, y_test

def image_process(image_path):
    image = tf.io.read_file(image_path)
    decoded_img = tf.image.decode_image(image, channels=3)
    resized_image = tf.image.resize(decoded_img, (150, 150))
    return resized_image
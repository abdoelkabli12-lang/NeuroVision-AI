import os
from sklearn.model_selection import train_test_split
import dataset_utils as du
from config import CLASS_MAPPINGS, SEED, IMAGES_EXT, DATASET_ROOT

train_manifest = du.build_detection_manifest(f"{DATASET_ROOT}/Train")

def data_split(manifest, seed=SEED):
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=seed)
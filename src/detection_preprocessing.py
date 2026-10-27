from dataset_utils import build_detection_manifest
from ultralytics import YOLO
import tensorflow as tf
from classification_data import data_loader, data_split, load_classification_data

x_val, y_val = load_classification_data("data/raw", split="Val")

x_train, x_test, y_train, y_test = data_split("data/raw")

train_ds = data_loader(
    x_train,
    y_train,
    batch_size=32,
    augment=True,
    shuffle=True
)

val_ds = data_loader(
  x_val,
  y_val,
  batch_size=32
)

def 
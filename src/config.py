from pathlib import Path

DATASET_ROOT = "data/raw"

TRAIN_DATA = "data/raw/Train"
VAL_DATA = "data/raw/Val"

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

DETECTION_DATA_ROOT = "data/processed/yolo"

MODELS = ("resnet50", "resnet101", "mobilenetv2")

REGULARISATIONS = ("L1", "L2")

MODELS_DIR = Path("models")

EPOCHS = 10

FINE_TUNE_LAYERS = 30
from dataset_utils import build_detection_manifest
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

history = model.train(
  data="data/processed/yolo/data.yaml",
  epochs=30,
  imgsz=640,
  batch=32,
  device=0,
  workers=0
)

from dataset_utils import build_detection_manifest
from ultralytics import YOLO
import csv


if __name__ == '__main__':
  model = YOLO("yolo11n.pt")

  history = model.train(
      data="data/processed/yolo/data.yaml",
      epochs=50,
      imgsz=640,
      batch=-1,
      device=0,
      workers=8,
      cache="ram",
      cos_lr=True,
  )
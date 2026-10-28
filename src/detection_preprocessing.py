from dataset_utils import build_detection_manifest
from ultralytics import YOLO
import csv


if __name__ == '__main__':
  # model = YOLO("yolo11n.pt")

  # history = model.train(
  #     data="data/processed/yolo/data.yaml",
  #     epochs=50,
  #     imgsz=640,
  #     batch=-1,
  #     device=0,
  #     workers=8,
  #     cache="ram",
  #     cos_lr=True,
  # )

  model = YOLO("runs/detect/train/weights/best.pt")

  metrics = model.val(
      data="data/processed/yolo/data.yaml",
      split="test"
  )

  print("mAP@0.5:", metrics.box.map50)
  print("mAP@0.5:0.95:", metrics.box.map)
  

  with open("test_metrics.csv", "w", newline="") as file:
      writer = csv.writer(file)

      writer.writerow(["Metric", "Value"])
      writer.writerow(["Precision", metrics.box.mp])
      writer.writerow(["Recall", metrics.box.mr])
      writer.writerow(["mAP@0.5", metrics.box.map50])
      writer.writerow(["mAP@0.5:0.95", metrics.box.map])
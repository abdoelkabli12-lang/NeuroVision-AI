from ultralytics import YOLO
import csv

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
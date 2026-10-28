import os
import csv
import shutil
from sklearn.model_selection import train_test_split
import dataset_utils as du
from config import CLASS_MAPPINGS, SEED, IMAGES_EXT, DATASET_ROOT, DETECTION_DATA_ROOT

T_manifest = du.build_detection_manifest(f"{DATASET_ROOT}/Train")
V_manifest = du.build_detection_manifest(f"{DATASET_ROOT}/Val")

def data_split(manifest, seed=SEED):
    train_manifest, test_manifest = train_test_split(manifest, test_size=0.15, shuffle=True, random_state=seed)
    return train_manifest, test_manifest

train_manifest, test_manifest = data_split(manifest=T_manifest)

for elem in test_manifest:
  elem['split'] = 'test'

output_path = "data/processed/yolo/detection_manifests.csv"

records = []

for split_name, manifest in [
    ("train", train_manifest),
    ("test", test_manifest),
    ("val", V_manifest),
]:
    for record in manifest:
        row = record.copy()
        row["split"] = split_name
        records.append(row)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    
    
def organize_yolo_dataset(manifest):
  for elem in manifest:
    destination = elem['split']
    image_dest = os.path.join(DETECTION_DATA_ROOT, "images", destination)
    label_dest = os.path.join(DETECTION_DATA_ROOT, "labels", destination)
    imagepath = elem['image_path']
    labelpath = elem['label_path']
    shutil.copy(imagepath, image_dest)
    shutil.copy(labelpath, label_dest)
  return "coppied successfuly"

organize_yolo_dataset(V_manifest)
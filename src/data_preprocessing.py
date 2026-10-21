import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from collections import defaultdict

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

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

def save_class_mappings(mappings):
  with open ("models/class_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)
    print("classes mapped successfully")
save_class_mappings(CLASS_MAPPINGS)

def validate_dataset_root(manifest_or_root, allowed_paths):
  
  norm_root = os.path.normpath(manifest_or_root)
  
  allowed_pths = [os.path.normpath(allowed_paths[i]) for i in range(len(allowed_paths))]

  if norm_root not in allowed_pths:
    raise ValueError(f"Path {norm_root!r} not in allowed_paths")

  return norm_root

def get_file_stems(filenames):
    """
    Extract filename stems by removing their file extensions.

    This utility processes a collection of filenames and returns
    their stems, allowing filenames with different extensions to
    be compared independently of their file types.

    Args:
        filenames (Iterable[str]): An iterable containing filenames
            with or without file extensions.

    Returns:
        list[str]: A list containing the filename stems in the
            same order as the input filenames.

    Example:
        Input:
            ["image_001.jpg", "image_002.png", "image_003.txt"]

        Output:
            ["image_001", "image_002", "image_003"]
    """
    stems = []
    for f in filenames:
        file, _ = os.path.splitext(f)
        stems.append(file)
    return stems
  
def find_mismatched_stems(filenames, reference_stems):
    """
    Identify filename stems that do not exist in a reference collection.

    This utility compares a collection of filename stems against a
    reference collection and returns the stems that have no matching
    entry. It can be used to identify images without corresponding
    annotation files or annotation files without corresponding images.

    Args:
        filenames (Iterable[str]): Filename stems to check.
        reference_stems (Collection[str]): Stems used as the reference
            for determining whether a match exists.

    Returns:
        list[str]: Filename stems that are absent from the reference
            collection, preserving their original order.

    Example:
        Input:
            filenames = ["image_001", "image_002", "image_003"]
            reference_stems = {"image_001", "image_003"}

        Output:
            ["image_002"]
    """
    orphan_file = []
    for i in filenames:
        if i not in reference_stems:
            orphan_file.append(i)
    return orphan_file
  
  

def img_per_class(manifest_or_root, allowed_paths=['data/raw/Train', 'data/raw/Val']):
  """
  Count the number of supported image files in each tumor classification
  category within a dataset directory.

  The function examines the expected class subdirectories, identifies
  valid image files using their file extensions, and calculates both
  the number of images in each class and the total number of images.
  It also restricts execution to dataset root paths explicitly permitted
  by the caller.

  Args:
      manifest_or_root (str): Path to the dataset root directory
          containing the class subdirectories.
      allowed_paths (list[str]): List of permitted dataset root paths.

  Returns:
      tuple[dict[str, int], int]: A tuple containing:
          - A dictionary mapping each expected class name to its
            image count.
          - The total number of valid image files across all classes.

  Raises:
      ValueError: If the normalized dataset root path is not included
          in the permitted paths.
  """
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
      classpath = os.path.join(norm_root, classname, "Images")
      if os.path.isdir(classpath):
        for filename in os.listdir(classpath):
            filepath = os.path.join(classpath, filename)
            _, ext = os.path.splitext(filename)
            if os.path.isfile(filepath) and ext.lower() in IMAGES_EXT:
              filecounts[classname] += 1
              totalfiles += 1
  return filecounts, totalfiles




filecounts, totalfiles = img_per_class(TRAIN_DATA)

print(f"Glioma count : {filecounts['Glioma']}")
print(f"Meningioma count : {filecounts['Meningioma']}")
print(f"No Tumor count : {filecounts['No Tumor']}")
print(f"Pituitary count : {filecounts['Pituitary']}")
print(f"total: {totalfiles}")


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

file, total = image_annotations("data/raw/Train")

print(f"Glioma count : {file['Glioma']}")
print(f"Meningioma count : {file['Meningioma']}")
print(f"No Tumor count : {file['No Tumor']}")
print(f"Pituitary count : {file['Pituitary']}")
print(f"total: {total}")

def audit_image_label_pairs(manifest_or_root, allowed_paths=['data/raw/Train','data/raw/Val']):
    norm_root = validate_dataset_root(manifest_or_root, allowed_paths)

    res_dict = {}

    for clss in CLASSES:
        image_dir = os.path.join(norm_root, clss, "Images")
        label_dir = os.path.join(norm_root, clss, "Labels")

        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"Missing Images or Labels directory for class: {clss}"
            )

        image_names = [
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGES_EXT
        ]

        label_names = [
            f for f in os.listdir(label_dir)
            if os.path.isfile(os.path.join(label_dir, f))
            and os.path.splitext(f)[1].lower() == ".txt"
        ]

        img_stems = get_file_stems(image_names)
        label_stems = get_file_stems(label_names)

        img_stems_set = set(img_stems)
        label_stems_set = set(label_stems)

        orphan_images = find_mismatched_stems(img_stems, label_stems_set)
        orphan_labels = find_mismatched_stems(label_stems, img_stems_set)

        res_dict[clss] = {
            "image_count": len(image_names),
            "label_count": len(label_names),
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels
        }

    return res_dict

results = audit_image_label_pairs("data/raw/Train")

for clss, data in results.items():
    print(f"\n{clss}")
    print(f"Images without labels: {len(data['orphan_images'])}")
    print(f"Labels without images: {len(data['orphan_labels'])}")
    print("Orphan images:", data["orphan_images"][:10])
    print("Orphan labels:", data["orphan_labels"][:10])






def visualize_dataset_grid(manifest_or_root=TRAIN_DATA, samples_per_class=6, seed=42):
    random.seed(seed)

    fig, axes = plt.subplots(
        len(CLASSES),
        samples_per_class,
        figsize=(18, 12),
        squeeze=False
    )

    for row, class_name in enumerate(CLASSES):
        image_dir = os.path.join(manifest_or_root, class_name, "Images")
        label_dir = os.path.join(manifest_or_root, class_name, "Labels")

        image_files = [
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGES_EXT
        ]

        selected = random.sample(
            image_files,
            min(samples_per_class, len(image_files))
        )

        for col, ax in enumerate(axes[row]):
            ax.axis("off")

            if col >= len(selected):
                continue

            image_name = selected[col]
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(
                label_dir,
                os.path.splitext(image_name)[0] + ".txt"
            )

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                ax.imshow(img)

            ax.set_title(
                f"{class_name}\n{image_name}",
                fontsize=8
            )

            if not os.path.isfile(label_path):
                ax.set_xlabel("Missing label", color="red", fontsize=8)
                continue

            with open(label_path, "r") as f:
                annotations = f.readlines()

            for line in annotations:
                values = line.split()

                if len(values) != 5:
                    continue

                class_id, x_center, y_center, box_width, box_height = map(
                    float, values
                )

                # Convert normalized YOLO coordinates to pixels
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2

                rectangle = patches.Rectangle(
                    (x_min, y_min),
                    box_width,
                    box_height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"
                )

                ax.add_patch(rectangle)

                ax.text(
                    x_min,
                    max(0, y_min - 3),
                    f"ID: {int(class_id)}",
                    color="white",
                    fontsize=7,
                    bbox=dict(facecolor="red", alpha=0.7, pad=2)
                )

    plt.tight_layout()
    plt.show()

visualize_dataset_grid()



def data_split(manifest_or_root, split="Train", allowed_paths=['data/raw']):
    norm_root = validate_dataset_root(manifest_or_root, allowed_paths)
    
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

flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED)
rotation = tf.keras.layers.RandomRotation(factor=0.05, seed=SEED)
zoom = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=SEED)
translate = tf.keras.layers.RandomTranslation(width_factor=0.1, height_factor=0.1, seed=SEED)
contrast = tf.keras.layers.RandomContrast(factor=0.1, value_range=(0, 255), seed=SEED)

def augmentation(image):
    image = flip(image)
    image = rotation(image)
    image = zoom(image)
    image = translate(image)
    image = contrast(image)
    return image

def load_classification_data(manifest_or_root, split="Train", allowed_paths=['data/raw']):
    norm_root = validate_dataset_root(manifest_or_root, allowed_paths)
    
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


image_path = r"data/raw/Train/Glioma/Images/gg (1).jpg"

image = image_process(image_path)

print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Min pixel:", tf.reduce_min(image).numpy())
print("Max pixel:", tf.reduce_max(image).numpy())



def visualize_augmentation(image_path):
    image = image_process(image_path)

    images = [
        image,
        flip(image),
        rotation(image),
        zoom(image),
        translate(image),
        contrast(image)
    ]

    titles = [
        "Original", "Horizontal Flip", "Rotation",
        "Zoom", "Translation", "Contrast"
    ]

    plt.figure(figsize=(12, 8))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8))
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
visualize_augmentation(image_path=image)


import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import os
from config import CLASSES, IMAGES_EXT, TRAIN_DATA
import augmentation as ag
from classification_data import image_process
import tensorflow as tf


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



def visualize_augmentation(image_path):
    image = image_process(image_path)

    images = [
        image,
        ag.flip(image),
        ag.rotation(image),
        ag.zoom(image),
        ag.translate(image),
        ag.contrast(image)
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
    
visualize_augmentation(image_path="data/raw/Train/Glioma/Images/gg (2).jpg")


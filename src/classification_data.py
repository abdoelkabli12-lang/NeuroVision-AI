import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataset_utils as du
from config import CLASS_MAPPINGS, SEED, IMAGES_EXT, DATASET_ROOT
import pandas as pd
from augmentation import augmentation

def load_classification_data(manifest_or_root, split="Train", allowed_paths=['data/raw']):
    norm_root = du.validate_dataset_root(manifest_or_root, allowed_paths)

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
def data_split(manifest_or_root, split="Train", allowed_paths=['data/raw']):
    norm_root = du.validate_dataset_root(manifest_or_root, allowed_paths)
    
    if split != "Train":
        raise ValueError("You can split only the Train data")
    x, y = load_classification_data(norm_root, split)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, stratify=y, random_state=SEED)
    
    return x_train, x_test, y_train, y_test

def image_process(image_path):
    image = tf.io.read_file(image_path)
    decoded_img = tf.image.decode_image(image, channels=3)
    decoded_img.set_shape([None, None, 3])
    resized_image = tf.image.resize(decoded_img, (150, 150))
    return resized_image

def split_manifest(x_train, x_test, y_train, y_test):
    manifest_table = []
    reversed_mappings = {v: k for k, v in CLASS_MAPPINGS.items()}
    
    zipped_train = zip(x_train, y_train)
    
    for pair in zipped_train:
        dic = {
            "image_path": pair[0],
            "class_id": pair[1],
            "class_name": reversed_mappings[pair[1]],
            "split": "train"
        }
        manifest_table.append(dic)
        
    zipped_test = zip(x_test, y_test)
    
    for pair in zipped_test:
        dic = {
            "image_path": pair[0],
            "class_id": pair[1],
            "class_name": reversed_mappings[pair[1]],
            "split": "test"
        }
        manifest_table.append(dic)
    df = pd.DataFrame(manifest_table)
    return df


def save_manifest(df, path):
    return df.to_csv(path, index=False)

def data_loader(image_paths, labels, batch_size=32, augment=False, shuffle=False, buffer_size=1000, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda x, y: (image_process(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, seed=seed)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
    
x_train, x_test, y_train, y_test = data_split("data/raw")

# df = split_manifest(x_train, x_test, y_train, y_test)

# save_manifest(df, "data/processed/classification_manifest.csv")


train_ds = data_loader(
    x_train,
    y_train,
    batch_size=32,
    augment=True,
    shuffle=True
)

# 3. Inspect one batch
for images, labels in train_ds.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Pixel minimum:", tf.reduce_min(images).numpy())
    print("Pixel maximum:", tf.reduce_max(images).numpy())
    print("Labels in batch:", labels.numpy())
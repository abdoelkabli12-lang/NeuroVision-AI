import os
from config import CLASSES, IMAGES_EXT, CLASS_MAPPINGS
from annotation import validate_annotation_file
from path_utils import validate_dataset_root
import Exceptions as e

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
  
  
def get_file_stem(filename):
    """
    Extract the stem of a single filename by removing its file extension.

    Args:
        filename (str): A filename with or without a file extension.

    Returns:
        str: The filename stem, excluding the final extension.

    Example:
        Input:
            "image_001.jpg"

        Output:
            "image_001"
    """
    
    file, _ = os.path.splitext(filename)
    return file

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
            # and validate_annotation_file(os.path.join(label_dir, f))
            and os.path.splitext(f)[1].lower() == ".txt"
        ]
        
        invalid_labels = []
        
        for ann in os.listdir(label_dir):
            path = os.path.join(label_dir, ann)
            if not validate_annotation_file(path):
                invalid_labels.append(ann)

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
            "orphan_labels": orphan_labels,
            "invalid annotations": invalid_labels
        }

    return res_dict

def build_detection_manifest(manifest_or_root, allowed_paths=None):
    res = []
    dic = {}
    norm_root = validate_dataset_root(manifest_or_root=manifest_or_root, allowed_paths=['data/raw/Train', 'data/raw/Val']) 
    for clss in os.listdir(norm_root):
        imagestems = []
        labelstems = []
        if clss in CLASSES:
            classpath = os.path.join(norm_root, clss)
            imagespath = os.path.join(classpath, "Images")
            labelspath = os.path.join(classpath, "Labels")
            for imagename in os.listdir(imagespath):
                imagestems.append(get_file_stem(imagename))
            for labelsname in os.listdir(labelspath):
                labelstems.append(get_file_stem(labelsname))
            for image in imagestems:
                if image not in labelstems:
                    raise e.MissingCredintialsError("Missing label for image")
            for label in labelstems:
                if label not in imagestems:
                    raise e.MissingCredintialsError("Missing image for label")
            for imagename in os.listdir(imagespath):
                imagestem = get_file_stem(imagename)
                dic = {
                    "image_path": os.path.join(imagespath, imagename),
                    "label_path": os.path.join(labelspath, imagestem + ".txt"),
                    "class_name": clss,
                    "class_id": CLASS_MAPPINGS[clss],
                    "split": os.path.basename(norm_root)
                }

                res.append(dic)

    return res
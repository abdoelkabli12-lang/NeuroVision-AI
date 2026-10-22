import os

def validate_dataset_root(manifest_or_root, allowed_paths):
  
  norm_root = os.path.normpath(manifest_or_root)
  
  allowed_pths = [os.path.normpath(allowed_paths[i]) for i in range(len(allowed_paths))]

  if norm_root not in allowed_pths:
    raise ValueError(f"Path {norm_root!r} not in allowed_paths")

  return norm_root

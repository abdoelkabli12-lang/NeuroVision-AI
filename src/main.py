import json
from config import CLASS_MAPPINGS

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

def save_class_mappings(mappings):
  with open ("models/class_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)
    print("classes mapped successfully")
save_class_mappings(CLASS_MAPPINGS)


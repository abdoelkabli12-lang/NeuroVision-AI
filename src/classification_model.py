import tensorflow as tf
import Exceptions as e
from classification_data import data_loader, data_split, load_classification_data

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_visible_devices(gpus[0], 'GPU')

x_val, y_val = load_classification_data("data/raw", split="Val")

x_train, x_test, y_train, y_test = data_split("data/raw")

train_ds = data_loader(
    x_train,
    y_train,
    batch_size=32,
    augment=True,
    shuffle=True
)

val_ds = data_loader(
  x_val,
  y_val,
  batch_size=32
)


image_batch, label_batch = next(iter(train_ds))



def build_model(model: str):
  model = model.lower()
  
  models = [
    "resnet50",
    "resnet101",
    "mobilenetv2"
  ]
  
  if model in models:
    match(model):
      case "resnet50":
        img_preprocessed = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)
        base_model = tf.keras.applications.ResNet50(
              include_top=False,
              weights="imagenet",
              input_shape=(150, 150, 3)
            )
      case "resnet101":
        img_preprocessed = tf.keras.layers.Lambda(tf.keras.applications.resnet101.preprocess_input)
        base_model = tf.keras.applications.ResNet101(
              include_top=False,
              weights="imagenet",
              input_shape=(150, 150, 3)
            )
      case "mobilenetv2":
        img_preprocessed = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)
        base_model = tf.keras.applications.MobileNetV2(
              include_top=False,
              weights="imagenet",
              input_shape=(150, 150, 3)
            )
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(
      4,
      activation='softmax',
      kernel_regularizer=tf.keras.regularizers.L2(1e-4),
      # bias_regularizer=tf.keras.regularizers.L2(1e-4),
      # activity_regularizer=tf.keras.regularizers.L2(1e-5)
    )
    base_model.trainable = False
    
    seq = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(150, 150, 3)),
      img_preprocessed,
      base_model,
      global_average_layer,
      tf.keras.layers.Dropout(0.3),
      prediction_layer
    ])
    
    adam = tf.keras.optimizers.Adam(
      learning_rate=0.0001
    )
    seq.compile(
      optimizer=adam,
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"]
    )
  else:
    raise e.ModelNotFoundError(f"Model {model} isn't in the list of accepted models")
  
  return seq

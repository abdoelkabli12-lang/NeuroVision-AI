import tensorflow as tf
from classification_model import build_model
from classification_data import data_loader, data_split, load_classification_data
import json

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


model = tf.keras.models.load_model(
  "models/best_classification_model.keras"
)

# model.summary()

resnet = model.get_layer(name="resnet50")

resnet.summary()

model.trainable = True

resnet.trainable = True

for layer in resnet.layers:
  layer.trainable = False


for layer in resnet.layers[-30:]:
  if not isinstance(layer, tf.keras.layers.BatchNormalization):
    layer.trainable = True

adam = tf.keras.optimizers.Adam(
    learning_rate=0.00001
  )


model.compile(
    optimizer=adam,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    
  )

early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  min_delta=0.002,
  patience=5,
  mode='auto',
  restore_best_weights=True
)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.1,
  patience=2,
  mode='auto',
  cooldown=0,
  min_delta=0.001
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath='models/best_classification_fine_tune_model.keras',
  mode='auto',
  save_best_only=True
)

callbacks = [early_stopping, reducelr, checkpoint]

history = model.fit(
  x=train_ds,
  validation_data=val_ds,
  epochs=10,
  callbacks=callbacks
)

with open("models/fine_tuning_classification_history.json", "w") as f:
  json.dump(history.history, f)


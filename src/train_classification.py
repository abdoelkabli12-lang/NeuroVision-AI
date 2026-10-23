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


model = build_model()

early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  min_delta=0.002,
  patience=3,
  mode='auto',
  restore_best_weights=False
)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.1,
  patience=3,
  mode='auto',
  cooldown=0,
  min_delta=0.001
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath='models/best_classification_model.keras',
  mode='auto',
  save_best_only=True
)

callbacks = [early_stopping, reducelr, checkpoint]

history = model.fit(
  x=train_ds,
  batch_size=32,
  epochs=10,
  callbacks=callbacks
)

with open("models/classification_model_history.json", "w") as f:
  json.dump(history.history, f)
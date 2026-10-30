import json
from pathlib import Path

import tensorflow as tf

from classification_data import data_loader, data_split, load_classification_data
from train_classification import BATCH_SIZE, EPOCHS, MODELS, MODELS_DIR, REGULARISATIONS, model_stem

FINE_TUNE_LAYERS = 30


def make_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.002, patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, min_delta=0.001),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True),
    ]


def unfreeze_backbone(model: tf.keras.Model, backbone_name: str) -> None:
    backbone = model.get_layer(backbone_name)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def main() -> None:
    x_val, y_val = load_classification_data("data/raw", split="Val")
    x_train, _, y_train, _ = data_split("data/raw")
    train_ds = data_loader(x_train, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_ds = data_loader(x_val, y_val, batch_size=BATCH_SIZE)

    for model_name in MODELS:
        for regularisation in REGULARISATIONS:
            stem = model_stem(model_name, regularisation)
            source_path = MODELS_DIR / f"best_{stem}.keras"
            target_path = MODELS_DIR / f"best_{stem}_fine_tune.keras"
            history_path = MODELS_DIR / f"{stem}_fine_tune_history.json"
            if not source_path.exists() or source_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Train this model first; a valid saved model is required: {source_path}"
                )

            print(f"\nFine-tuning {model_name} with {regularisation} regularisation")
            model = tf.keras.models.load_model(source_path)
            unfreeze_backbone(model, model_name)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=make_callbacks(target_path))
            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history.history, file, indent=2)
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()

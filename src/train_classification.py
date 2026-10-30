"""Train all classification architecture/regularisation configurations."""

import json
from pathlib import Path

import tensorflow as tf

from classification_data import data_loader, data_split, load_classification_data
from classification_model import build_model

from config import MODELS, MODELS_DIR, BATCH_SIZE, REGULARISATIONS, EPOCHS

def model_stem(model_name: str, regularisation: str) -> str:
    return f"{model_name.lower()}_model_{regularisation.upper()}"


def make_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.002, patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_delta=0.001),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True),
    ]


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    x_val, y_val = load_classification_data("data/raw", split="Val")
    x_train, _, y_train, _ = data_split("data/raw")
    train_ds = data_loader(x_train, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_ds = data_loader(x_val, y_val, batch_size=BATCH_SIZE)

    for model_name in MODELS:
        for regularisation in REGULARISATIONS:
            stem = model_stem(model_name, regularisation)
            model_path = MODELS_DIR / f"best_{stem}.keras"
            history_path = MODELS_DIR / f"{stem}_history.json"
            print(f"\nTraining {model_name} with {regularisation} regularisation")
            model = build_model(model_name, regularisation)
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=make_callbacks(model_path))
            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history.history, file, indent=2)
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()

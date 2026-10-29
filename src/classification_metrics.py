"""Evaluate every feature-extraction and fine-tuned classification model."""

import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from classification_data import data_loader, data_split
from train_classification import BATCH_SIZE, MODELS, MODELS_DIR, REGULARISATIONS, model_stem

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def evaluate_model(model_path: Path, test_ds, y_test) -> dict:
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=0)
    predictions = tf.argmax(probabilities, axis=1).numpy()
    return {
        "model_file": model_path.name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


def main() -> None:
    _, x_test, _, y_test = data_split("data/raw")
    test_ds = data_loader(x_test, y_test, batch_size=BATCH_SIZE)

    for model_name in MODELS:
        for regularisation in REGULARISATIONS:
            stem = model_stem(model_name, regularisation)
            model_stages = (
                ("feature_extraction", MODELS_DIR / f"best_{stem}.keras"),
                ("fine_tune", MODELS_DIR / f"best_{stem}_fine_tune.keras"),
            )
            for stage, model_path in model_stages:
                if not model_path.exists() or model_path.stat().st_size == 0:
                    print(f"Skipping missing or empty model: {model_path}")
                    continue
                metrics = evaluate_model(model_path, test_ds, y_test)
                metrics.update({"architecture": model_name, "regularisation": regularisation, "stage": stage})
                output_path = MODELS_DIR / f"{stem}_{stage}_metrics.json"
                with output_path.open("w", encoding="utf-8") as file:
                    json.dump(metrics, file, indent=2)
                print(f"Saved metrics: {output_path}")
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()

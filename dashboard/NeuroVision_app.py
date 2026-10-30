"""Streamlit interface for NeuroVision MRI classification and detection.

Run from the project root with: streamlit run dashboard/NeuroVision_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
DETECTION_METRICS_PATH = PROJECT_ROOT / "test_metrics.csv"


def available_classifiers() -> dict[str, Path]:
    """Return saved Keras models, prioritising the final project-level model."""
    preferred = [MODELS_DIR / "best_classification_fine_tune_model.keras", MODELS_DIR / "best_classification_model.keras"]
    paths: list[Path] = []
    for path in [*preferred, *sorted(MODELS_DIR.glob("*.keras"))]:
        if path.exists() and path not in paths:
            paths.append(path)
    return {path.stem.replace("_", " ").title(): path for path in paths}


@st.cache_resource(show_spinner="Loading the classification model…")
def load_classifier(path_as_text: str) -> tf.keras.Model:
    """Load a saved model once; training is never invoked by this app."""
    return tf.keras.models.load_model(path_as_text, compile=False)


@st.cache_resource(show_spinner="Loading the detection model…")
def load_detector(path_as_text: str) -> YOLO:
    return YOLO(path_as_text)


@st.cache_data
def load_class_names() -> list[str]:
    mapping_path = MODELS_DIR / "class_mappings.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def classify(image: Image.Image, model_path: Path) -> tuple[str, float, pd.DataFrame]:
    """Prepare an RGB 150×150 batch and calculate all class probabilities."""
    image_array = np.asarray(image.convert("RGB").resize((150, 150)), dtype=np.float32)
    probabilities = load_classifier(str(model_path)).predict(np.expand_dims(image_array, 0), verbose=0)[0]
    table = pd.DataFrame({"Category": load_class_names(), "Probability": probabilities})
    table["Probability (%)"] = table["Probability"] * 100
    table = table.sort_values("Probability", ascending=False).reset_index(drop=True)
    return str(table.loc[0, "Category"]), float(table.loc[0, "Probability"]), table


def detect(image: Image.Image, confidence: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Run YOLO inference and return its annotated image plus box-level results."""
    result = load_detector(str(DETECTION_MODEL_PATH)).predict(source=np.asarray(image.convert("RGB")), conf=confidence, verbose=False)[0]
    annotated_rgb = result.plot()[:, :, ::-1]
    rows = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = [round(value, 1) for value in box.xyxy[0].tolist()]
        rows.append({"Category": result.names[class_id], "Confidence (%)": round(float(box.conf[0].item()) * 100, 2), "Bounding box (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})"})
    return annotated_rgb, pd.DataFrame(rows)


def show_metrics() -> None:
    st.subheader("Detection evaluation")
    if DETECTION_METRICS_PATH.exists():
        metrics = pd.read_csv(DETECTION_METRICS_PATH)
        metrics["Value (%)"] = metrics["Value"] * 100
        st.dataframe(metrics[["Metric", "Value (%)"]], hide_index=True, use_container_width=True)
        st.bar_chart(metrics.set_index("Metric")["Value (%)"])
    else:
        st.info("Detection metrics have not been exported yet.")

    st.subheader("Classification training history")
    history_files = sorted(MODELS_DIR.glob("*history.json"))
    if not history_files:
        st.info("No classification training-history files were found.")
        return
    selected_history = st.selectbox("Training run", history_files, format_func=lambda path: path.stem.replace("_", " ").title())
    history = pd.DataFrame(json.loads(selected_history.read_text(encoding="utf-8")))
    history.index = history.index + 1
    history.index.name = "Epoch"
    metric_columns = [name for name in history if "accuracy" in name or "loss" in name]
    st.line_chart(history[metric_columns])


def main() -> None:
    st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")
    st.title("🧠 NeuroVision")
    st.caption("Brain MRI classification and tumor detection")

    classifiers = available_classifiers()
    if not classifiers:
        st.error(f"No saved classification models were found in {MODELS_DIR}.")
        st.stop()
    if not DETECTION_MODEL_PATH.exists():
        st.error(f"The saved detection model was not found at {DETECTION_MODEL_PATH}.")
        st.stop()

    with st.sidebar:
        st.header("Inference settings")
        selected_name = st.selectbox("Classification model", list(classifiers))
        confidence = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
        st.divider()
        st.caption("Saved models are cached for this session. This interface never starts training.")

    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    analyse = st.button("Analyze image", type="primary", disabled=uploaded_file is None)
    analysis_tab, metrics_tab = st.tabs(["MRI analysis", "Metrics and results"])

    with analysis_tab:
        if uploaded_file is None:
            st.info("Upload a JPG or PNG brain MRI image to begin.")
        elif not analyse:
            st.info("Image ready. Select **Analyze image** to run the saved models.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            input_column, output_column = st.columns(2)
            with input_column:
                st.subheader("Uploaded MRI")
                st.image(image, use_container_width=True)
            with st.spinner("Running classification and object detection…"):
                predicted_class, top_probability, probabilities = classify(image, classifiers[selected_name])
                annotated_image, detections = detect(image, confidence)
            with output_column:
                st.subheader("Tumor detection")
                st.image(annotated_image, use_container_width=True)

            st.subheader("Classification result")
            st.success(f"Predicted category: **{predicted_class}** — {top_probability:.1%} confidence")
            st.bar_chart(probabilities.set_index("Category")[["Probability (%)"]])
            st.dataframe(probabilities[["Category", "Probability (%)"]].style.format({"Probability (%)": "{:.2f}"}), hide_index=True, use_container_width=True)
            st.subheader("Detection results")
            if detections.empty:
                st.warning("No objects were detected at the chosen confidence threshold.")
            else:
                st.dataframe(detections, hide_index=True, use_container_width=True)

    with metrics_tab:
        show_metrics()


if __name__ == "__main__":
    main()

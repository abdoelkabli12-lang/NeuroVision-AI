import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from classification_data import data_loader, load_classification_data
import json
from classification_data import data_loader, data_split, load_classification_data


MODEL_PATH = "models/best_classification_fine_tune_model.keras"

_, x_test, _, y_test = data_split("data/raw")

test_ds = data_loader(
    x_test,
    y_test,
    batch_size=32
)

model = tf.keras.models.load_model(MODEL_PATH)


def evaluate_model(y_true, y_pred):
  accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
  precision = precision_score(y_true, y_pred, average="macro")
  recall = recall_score(y_true, y_pred, average="macro")
  f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
  conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
  classification_rep = classification_report(y_true=y_true, y_pred=y_pred, target_names=["Glioma", "Meningioma", "No Tumor", "Pituitary"], zero_division=0)
  return accuracy, precision, recall, f1, conf_matrix, classification_rep


def model_predict(model, val_ds):
  preditions = model.predict(val_ds)
  predicted_ids = tf.argmax(preditions, axis=1).numpy()
  return predicted_ids

prediction_ids = model_predict(model=model, val_ds=test_ds)
eval_model = evaluate_model(y_test, prediction_ids)

print(f"Accuracy: {eval_model[0]:.4f}")
print(f"Macro Precision: {eval_model[1]:.4f}")
print(f"Macro Recall: {eval_model[2]:.4f}")
print(f"Macro F1-Score: {eval_model[3]:.4f}")

print("\nClassification Report:")
print(eval_model[5])


plt.figure(figsize=(8, 6))

plt.imshow(eval_model[4], interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
plt.xticks(range(4), class_names, rotation=45, ha="right")
plt.yticks(range(4), class_names)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

for i in range(4):
    for j in range(4):
        plt.text(j, i, eval_model[4][i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

FEATURE_HISTORY_PATH = "models/classification_model_history.json"
FINE_TUNE_HISTORY_PATH = "models/fine_tuning_classification_history.json"


def load_history(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def plot_training_history(history, title, save_prefix, show_validation=True):
    epochs = range(1, len(history["accuracy"]) + 1)

    # Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["accuracy"], marker="o", label="Training Accuracy")

    if show_validation and "val_accuracy" in history:
        plt.plot(
            epochs,
            history["val_accuracy"],
            marker="o",
            label="Validation Accuracy"
        )

    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_accuracy.png", dpi=300)
    plt.show()

    # Loss
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["loss"], marker="o", label="Training Loss")

    if show_validation and "val_loss" in history:
        plt.plot(
            epochs,
            history["val_loss"],
            marker="o",
            label="Validation Loss"
        )

    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=300)
    plt.show()

    # Learning rate
    if "learning_rate" in history:
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, history["learning_rate"], marker="o")
        plt.title(f"{title} - Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_learning_rate.png", dpi=300)
        plt.show()


def main():
    feature_history = load_history(FEATURE_HISTORY_PATH)
    fine_tune_history = load_history(FINE_TUNE_HISTORY_PATH)

    plot_training_history(
        feature_history,
        title="Feature Extraction",
        save_prefix="feature_extraction",
        show_validation=False
    )

    plot_training_history(
        fine_tune_history,
        title="Fine-Tuning",
        save_prefix="fine_tuning",
        show_validation=True
    )


if __name__ == "__main__":
    main()
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from classification_data import data_loader, load_classification_data


MODEL_PATH = "models/best_classification_fine_tune_model.keras"

x_val, y_val = load_classification_data("data/raw", split="Val")


val_ds = data_loader(
  x_val,
  y_val,
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

prediction_ids = model_predict(model=model, val_ds=val_ds)
eval_model = evaluate_model(y_val, prediction_ids)

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
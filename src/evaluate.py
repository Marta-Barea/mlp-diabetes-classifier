from pathlib import Path
import os
import tensorflow as tf
import numpy as np

from .data_loader import load_data
from .utils.plot_confusion_matrix import plot_confusion_matrix
from .config import MODEL_OUTPUT_DIR


def get_latest_model(directory: str | Path, pattern: str = "best_mlp_*.h5") -> str | None:
    files = sorted(
        Path(directory).glob(pattern),
        key=os.path.getmtime,
        reverse=True
    )
    return str(files[0]) if files else None


def evaluator(model_path: str | None = None):
    X_train, X_test, y_train, y_test = load_data()

    model_path = model_path or get_latest_model(MODEL_OUTPUT_DIR)
    if model_path is None or not os.path.isfile(model_path):
        print("❌ Model file not found.")
        return

    print(f"🔄 Loading model: {Path(model_path).name}")
    model = tf.keras.models.load_model(model_path)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(
        f"\n📊 Train accuracy: {train_acc * 100:.2f}% (loss={train_loss:.4f})")
    print(
        f"📊 Test  accuracy: {test_acc * 100:.2f}% (loss={test_loss:.4f})")

    y_pred = np.argmax(model.predict(X_test[:10]), axis=1)
    print("\n🔎 First 10 predictions vs. actual values:")
    for i in range(10):
        print(f"     Predicted: {y_pred[i]}, Actual: {int(y_test[i])}")

    print("\n📊 Plotting confusion matrix...")
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    evaluator()

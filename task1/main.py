from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist

from mnist_classifier import MnistClassifier

LOGS_DIR = Path("logs")


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset.

    Returns:
        Tuple containing training and test data.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Ensure labels are integers
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, y_train, X_test, y_test


def evaluate_model(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray, str]:
    """
    Evaluate model performance.

    Args:
        y_pred: Predicted labels
        y_true: Ground truth labels

    Returns:
        Tuple of (accuracy, confusion matrix, classification report)
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    return accuracy, conf_matrix, class_report


def save_results(
    algorithm: str,
    accuracy: float,
    conf_matrix: np.ndarray,
    report: str,
) -> None:
    """Save evaluation results to logs folder."""
    LOGS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = LOGS_DIR / f"{algorithm}_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {algorithm}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")

        f.write("Classification Report:\n")
        f.write(report)

    print(f"Results saved to {file_path}")


def run_experiment(
    algorithm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train, predict, and evaluate a single model."""
    classifier = MnistClassifier(algorithm=algorithm)

    print(f"Train model: {algorithm}")
    classifier.train(X_train, y_train)

    print(f"\nPredicting with model: {algorithm}")
    predictions = classifier.predict(X_test)

    accuracy, conf_matrix, report = evaluate_model(predictions, y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    save_results(algorithm, accuracy, conf_matrix, report)


def main() -> None:
    """Run experiments for all algorithms."""
    X_train, y_train, X_test, y_test = load_mnist_data()

    algorithms = ["rf", "nn", "cnn"]

    for algo in algorithms:
        run_experiment(algo, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

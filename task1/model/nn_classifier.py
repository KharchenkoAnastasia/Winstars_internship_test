import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from interfaces.mnist_classifier_interface import MNISTClassifierInterface


class NeuralNetworkMnistClassifier(MNISTClassifierInterface):
    """Feed-forward neural network classifier for MNIST dataset."""

    def __init__(
        self,
        input_size: int = 28 * 28,
        num_classes: int = 10,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        self._model: tf.keras.Model | None = None

    def _build_model(self) -> tf.keras.Model:
        """Build and compile feed-forward neural network."""
        model = models.Sequential(
            [
                layers.Input(shape=(self.input_size,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape input data for feed-forward network.

        Expected input:
            (n_samples, 28, 28)

        Output:
            (n_samples, 784)
        """
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        print("X shape nn", X.shape)
        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the neural network."""
        X_train = self._preprocess(X_train)

        self._model = self._build_model()

        self._model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data."""
        if self._model is None:
            raise ValueError("Model is not trained yet. Call `train` first.")

        X = self._preprocess(X)

        predictions = self._model.predict(X, verbose=0)

        return np.argmax(predictions, axis=1)

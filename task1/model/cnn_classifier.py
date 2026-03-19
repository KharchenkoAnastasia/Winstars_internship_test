from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from interfaces.mnist_classifier_interface import MNISTClassifierInterface


class CNNMnistClassifier(MNISTClassifierInterface):
    """Convolutional Neural Network classifier for MNIST dataset."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        self._model: tf.keras.Model | None = None

    def _build_model(self) -> tf.keras.Model:
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
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
        Reshape input data for CNN.

        Expected input:
            (n_samples, 28, 28)

        Output:
            (n_samples, 28, 28, 1)
        """
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        print("X shape cnn", X.shape)

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
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
        if self._model is None:
            raise ValueError("Model is not trained yet. Call `train` first.")

        X = self._preprocess(X)

        predictions = self._model.predict(X, verbose=0)

        return np.argmax(predictions, axis=1)

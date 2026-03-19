from typing import Literal

import numpy as np

from interfaces.mnist_classifier_interface import MNISTClassifierInterface
from model.cnn_classifier import CNNMnistClassifier
from model.nn_classifier import NeuralNetworkMnistClassifier
from model.rf_classifier import RandomForestMnistClassifier

AlgorithmType = Literal["rf", "nn", "cnn"]


class MnistClassifier:
    """
    Wrapper class for MNIST classifiers.

    Provides a unified interface for different algorithms:
    - Random Forest (rf)
    - Feed-Forward Neural Network (nn)
    - Convolutional Neural Network (cnn)
    """

    def __init__(self, algorithm: AlgorithmType) -> None:
        self.algorithm = algorithm
        self._model: MNISTClassifierInterface = self._initialize_model(algorithm)

    def _initialize_model(self, algorithm: AlgorithmType) -> MNISTClassifierInterface:
        """Initialize classifier based on selected algorithm."""
        if algorithm == "rf":
            return RandomForestMnistClassifier()

        if algorithm == "nn":
            return NeuralNetworkMnistClassifier()

        if algorithm == "cnn":
            return CNNMnistClassifier()

        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. " "Available options: 'rf', 'nn', 'cnn'."
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the selected model."""
        self._model.train(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the selected model."""
        return self._model.predict(X)

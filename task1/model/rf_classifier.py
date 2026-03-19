import numpy as np
from sklearn.ensemble import RandomForestClassifier

from interfaces.mnist_classifier_interface import MNISTClassifierInterface


class RandomForestMnistClassifier(MNISTClassifierInterface):
    """Random Forest classifier for MNIST dataset."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model: RandomForestClassifier | None = None

    def _build_model(self) -> RandomForestClassifier:
        """Initialize Random Forest model."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten input data for Random Forest.

        Expected input:
            (n_samples, 28, 28)

        Output:
            (n_samples, 784)
        """
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Random Forest model."""
        X_train = self._preprocess(X_train)

        self._model = self._build_model()
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data."""
        if self._model is None:
            raise ValueError("Model is not trained yet. Call `train` first.")

        X = self._preprocess(X)

        return self._model.predict(X)

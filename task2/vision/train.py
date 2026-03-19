import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the image classification model.
    """
    parser = argparse.ArgumentParser(description="Train Image Classification Model")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/animals",
        help="Path to dataset directory (subfolders per class)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--target_size", type=int, default=224, help="Height and width to resize images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="vision/model",
        help="Directory path to save the trained model",
    )

    return parser.parse_args()


def preprocess_image(image_path: str, target_size: int) -> np.ndarray:
    """
    Load an image, resize to target size, convert BGR to RGB, normalize pixel values to [0,1]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize image to target size
    image = cv2.resize(image, (target_size, target_size))
    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    image = image / 255.0

    return image


def load_dataset(dataset_path: str, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all images and labels from dataset directory.
    Assumes dataset_path contains subfolders per class.
    """
    data: List[np.ndarray] = []
    labels: List[str] = []

    # Get sorted list of classes (subfolder names)
    animal_classes = sorted(os.listdir(dataset_path))

    for animal in animal_classes:
        animal_dir = os.path.join(dataset_path, animal)
        if not os.path.isdir(animal_dir):
            continue

        # Iterate over images in class folder
        for img_name in os.listdir(animal_dir):
            img_path = os.path.join(animal_dir, img_name)
            try:
                image = preprocess_image(img_path, target_size)
                data.append(image)
                labels.append(animal)
            except Exception as e:
                print(f"[WARNING] Skipping {img_path}: {e}")

    return np.array(data), np.array(labels)


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Build a simple CNN for image classification
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train(args: argparse.Namespace) -> None:
    """
    Load dataset, build model, train and save model + label classes
    """
    print("[INFO] Loading dataset...")
    X, y = load_dataset(args.data_path, args.target_size)
    print(f"[INFO] Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded: np.ndarray = label_encoder.fit_transform(y)

    # Split dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("[INFO] Building model...")
    model = build_model(
        input_shape=(args.target_size, args.target_size, 3), num_classes=len(label_encoder.classes_)
    )

    print("[INFO] Training started...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("[INFO] Saving model...")
    os.makedirs(args.model_path, exist_ok=True)
    model.save(args.model_path)

    # Save label classes for inference
    np.save(os.path.join(args.model_path, "label_classes.npy"), label_encoder.classes_)

    print("[INFO] Training completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    train(args)

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for image classification inference
    """
    parser = argparse.ArgumentParser(description="Image Classification Inference")

    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="vision/model/model.h5",
        help="Path to trained model (.h5)",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="vision/model/label_classes.npy",
        help="Path to label classes (.npy)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Image size used during training (height and width)",
    )

    return parser.parse_args()


def preprocess_image(image_path: str, target_size: int) -> npt.NDArray[np.float32]:
    """
    Load an image, resize, normalize, and add batch dimension
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to target size
    image = cv2.resize(image, (target_size, target_size))
    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Add batch dimension: (1, H, W, C)
    image = np.expand_dims(image, axis=0)

    return image


def predict(
    model: tf.keras.Model, image: npt.NDArray[np.float32], class_names: npt.NDArray[np.str_]
) -> Tuple[str, float]:
    """
    Predict the class label and confidence for an input image
    """
    preds: npt.NDArray[np.float32] = model.predict(image)  # type: ignore

    predicted_idx: int = int(np.argmax(preds, axis=1)[0])
    confidence: float = float(np.max(preds))
    label: str = str(class_names[predicted_idx])

    return label, confidence


def run_inference(args: argparse.Namespace) -> Tuple[str, float]:
    """
    Load model and labels, preprocess image, and predict
    """
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    print("[INFO] Loading labels...")
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    class_names: npt.NDArray[np.str_] = np.load(args.labels_path)

    print("[INFO] Preprocessing image...")
    image: npt.NDArray[np.float32] = preprocess_image(args.image_path, args.target_size)

    print("[INFO] Running prediction...")
    label, confidence = predict(model, image, class_names)

    return label, confidence


def main() -> None:
    """
    Main entry point
    """
    args = parse_args()
    label, confidence = run_inference(args)
    print(f"Predicted label: {label}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()

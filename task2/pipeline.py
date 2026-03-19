import argparse
from typing import Any, Dict

from ner.inference import predict as ner_predict
from vision.inference import run_inference as cv_inference


def main() -> bool:
    """
    Main function to run a full pipeline: NER (Named Entity Recognition) + CV (Computer Vision)
    """
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Full pipeline: NER + CV")

    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--text_input", type=str, required=True, help="Input text containing possible animal names"
    )
    parser.add_argument(
        "--ner_model_path", type=str, default="ner/ner_model", help="Path to the NER model"
    )
    parser.add_argument(
        "--cv_model_path", type=str, default="vision/model/model.h5", help="Path to the CV model"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="vision/model/label_classes.npy",
        help="Path to the labels file",
    )

    args = parser.parse_args()

    # Run NER model on input text
    ner_result: Dict[str, Any] = ner_predict(args.text_input, args.ner_model_path)
    animals = ner_result.get("animals", [])

    if len(animals) == 0:
        # No animal found in text
        print("No animal found in text.")
        print(False)
        return False

    text_animal: str = animals[0].lower()  # Take the first detected animal and convert to lowercase

    # Create a simple class to store CV arguments
    class CVArgs:
        image_path: str
        model_path: str
        labels_path: str
        target_size: int

    cv_args = CVArgs()
    cv_args.image_path = args.image_path
    cv_args.model_path = args.cv_model_path
    cv_args.labels_path = args.labels_path
    cv_args.target_size = 224  # Typical image size for CNN input

    # Run CV model on image
    image_label, confidence = cv_inference(cv_args)  # type: Tuple[str, float]
    image_label = image_label.lower()

    # Compare NER result with CV result
    is_match: bool = text_animal == image_label

    # Print results
    print(f"Text animal: {text_animal}")
    print(f"Image prediction: {image_label} ({confidence:.4f})")

    if is_match:
        print("The predictions from both models are the same.")
    else:
        print("The predictions from both models are different.")

    print(is_match)
    return is_match


if __name__ == "__main__":
    main()

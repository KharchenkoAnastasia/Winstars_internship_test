import argparse
from typing import Any, Dict, List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Define labels and mapping
LABEL_LIST: List[str] = ["O", "B-ANIMAL"]
ID2LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABEL_LIST)}


def extract_animal(tokens: List[str], labels: List[str]) -> List[str]:
    """
    Extract animal names from token-level labels
    """
    animals: List[str] = []
    current: List[str] = []

    for token, label in zip(tokens, labels):
        if label == "B-ANIMAL":
            if current:
                animals.append(" ".join(current))
                current = []
            current.append(token)
        elif label == "I-ANIMAL":
            current.append(token)
        elif current:
            animals.append(" ".join(current))
            current = []

    if current:
        animals.append(" ".join(current))

    return animals


def predict(sentence: str, model_path: str) -> Dict[str, Any]:
    """
    Tokenize input sentence and predict NER labels
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get token-level predictions
    logits = outputs.logits  # type: ignore
    predictions = torch.argmax(logits, dim=2)[0]

    # Convert token IDs back to string tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels: List[str] = [ID2LABEL[p.item()] for p in predictions]

    # Remove special tokens
    clean_tokens: List[str] = []
    clean_labels: List[str] = []
    for token, label in zip(tokens, labels):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            clean_tokens.append(token)
            clean_labels.append(label)

    # Extract animals from tokens and labels
    animals: List[str] = extract_animal(clean_tokens, clean_labels)

    return {
        "sentence": sentence,
        "tokens": clean_tokens,
        "labels": clean_labels,
        "animals": animals,
    }


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="NER Animal Extraction")
    parser.add_argument("--test_sentence", type=str, required=True, help="Sentence to run NER on")
    parser.add_argument("--model_path", type=str, default="ner/ner_model", help="Path to NER model")

    args = parser.parse_args()

    result: Dict[str, Any] = predict(args.test_sentence, args.model_path)
    return result


if __name__ == "__main__":
    res = main()
    print(res)  # Ensure CLI still outputs result

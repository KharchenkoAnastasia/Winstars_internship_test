import argparse
import json
import os
from typing import Any, Dict

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Define labels and mappings
LABEL_LIST = ["O", "B-ANIMAL"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}


def load_data(path: str) -> Dataset:
    """
    Load dataset from JSON file and convert to HuggingFace Dataset
    """
    with open(path) as f:
        data = json.load(f)

    tokens = [item["tokens"] for item in data]
    labels = [[LABEL2ID[l] for l in item["labels"]] for item in data]

    return Dataset.from_dict({"tokens": tokens, "labels": labels})


def tokenize_and_align_labels(
    examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int
) -> Dict[str, Any]:
    """
    Tokenize a batch of examples and align token-level labels with subword tokens
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    labels_batch = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels_batch.append(label_ids)

    tokenized["labels"] = labels_batch
    return tokenized


def compute_metrics(p: Any) -> Dict[str, float]:
    """
    Compute precision, recall, f1 and accuracy metrics for evaluation
    """
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    true_labels: list[int] = []
    true_preds: list[int] = []

    for pred, lab in zip(predictions, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                true_labels.append(int(l_))
                true_preds.append(int(p_))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="binary"
    )
    acc = accuracy_score(true_labels, true_preds)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER model for animal extraction")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--output_dir", type=str, default="ner/ner_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)

    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "val.json")

    # Load datasets
    train_dataset = load_data(train_path)
    val_dataset = load_data(val_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, args.max_length), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, args.max_length), batched=True
    )

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Data collator handles padding dynamically
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

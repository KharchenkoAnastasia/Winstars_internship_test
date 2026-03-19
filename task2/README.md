# Task 2: Named Entity Recognition + Image Classification Pipeline

A two-model ML pipeline that extracts animal names from text using a BERT-based NER model, classifies animals in images using a CNN, and cross-validates both results to produce a single boolean output.

---

## Project Structure

```
task2/
├── data/
│   ├── animals/               # Image dataset (one subfolder per class)
│   │   ├── cat/
│   │   ├── dog/
│   │   └── ...
│   ├── ner/                   # NER dataset
│   │   ├── train.json
│   │   └── val.json
│   ├── cat.jpg                # Example test images
│   └── panda.jpg
│
├── ner/
│   ├── ner_model/             # Saved fine-tuned NER model
│   ├── __init__.py
│   ├── inference.py           # NER inference script
│   └── train.py               # NER training script
│
├── vision/
│   ├── model/                 # Saved CNN model + label classes
│   │   ├── model.h5
│   │   └── label_classes.npy
│   ├── __init__.py
│   ├── inference.py           # Vision inference script
│   └── train.py               # Vision training script
│
├── notebooks/                 # EDA notebooks
├── pipeline.py                # Full pipeline entry point
├── pyproject.toml
├── poetry.lock
└── requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/task2.git
cd task2
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Requirements

- Python ^3.11
- PyTorch ^2.2.0
- TensorFlow ^2.15.0
- Transformers ^4.40.0
- Datasets ^2.18.0
- OpenCV ^4.9.0
- scikit-learn ^1.4.0
- NumPy ^1.26.0

---

## NER Model (BERT-based Animal Extraction)

The NER model fine-tunes `bert-base-cased` for token classification with two labels: `O` and `B-ANIMAL`.

### Training

```bash
python ner/train.py \
  --data_dir data/ner \
  --model_name bert-base-cased \
  --output_dir ner/ner_model \
  --epochs 3 \
  --batch_size 16 \
  --max_length 64
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to folder containing `train.json` and `val.json` |
| `--model_name` | `bert-base-cased` | Pretrained HuggingFace model name |
| `--output_dir` | `ner/ner_model` | Where to save the fine-tuned model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--max_length` | `64` | Max token length for truncation/padding |

**Expected `train.json` / `val.json` format:**

```json
[
  {
    "tokens": ["The", "rabbit", "hopped", "across", "the", "garden"],
    "labels": ["O", "B-ANIMAL", "O", "O", "O", "O"]
  }
]
```

### Inference

```bash
python ner/inference.py \
  --test_sentence "The rabbit hopped across the garden" \
  --model_path ner/ner_model
```

| Argument | Default | Description |
|---|---|---|
| `--test_sentence` | *(required)* | Input sentence to run NER on |
| `--model_path` | `ner/ner_model` | Path to the saved fine-tuned model |

**Example output:**

```python
{
  'sentence': 'The rabbit hopped across the garden',
  'tokens': ['The', 'rabbit', 'hopped', 'across', 'the', 'garden'],
  'labels': ['O', 'B-ANIMAL', 'O', 'O', 'O', 'B-ANIMAL'],
  'animals': ['rabbit','garden']
}
```

---

## Vision Model (CNN Animal Classifier)

The vision model is a simple CNN trained on a multi-class animal image dataset (≥10 classes). Images are resized to 224×224 and normalized to [0, 1].

### Dataset Layout

The image dataset must be organized into one subfolder per class:

```
data/animals/
├── cat/
│   ├── img001.jpg
│   └── ...
├── dog/
│   ├── img001.jpg
│   └── ...
└── tiger/
    └── ...
```

### Training

```bash
python vision/train.py \
  --data_path data/animals \
  --epochs 10 \
  --batch_size 32 \
  --target_size 224 \
  --model_path vision/model
```

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data/animals` | Path to the image dataset root |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--target_size` | `224` | Image resize dimension (height = width) |
| `--model_path` | `vision/model` | Directory to save model and label classes |

The script saves `model.h5` and `label_classes.npy` to `--model_path`.

### Inference

```bash
python vision/inference.py \
  --image_path data/panda.jpg \
  --model_path vision/model/model.h5 \
  --labels_path vision/model/label_classes.npy \
  --target_size 224
```

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to the input image |
| `--model_path` | `vision/model/model.h5` | Path to the saved `.h5` model |
| `--labels_path` | `vision/model/label_classes.npy` | Path to saved label classes |
| `--target_size` | `224` | Image resize dimension used during training |

**Example output:**

```
[INFO] Loading model...
[INFO] Loading labels...
[INFO] Preprocessing image...
[INFO] Running prediction...
Predicted label: panda, Confidence: 0.3280
```

---

## Full Pipeline

The pipeline combines both models end-to-end: it extracts the animal name from the input text with the NER model and compares it to the animal predicted in the input image by the CNN. It returns `True` if both agree, `False` otherwise.

```bash
python pipeline.py \
  --text_input "Wow! That cat is so cute!!" \
  --image_path data/cat.jpg \
  --ner_model_path ner/ner_model \
  --cv_model_path vision/model/model.h5 \
  --labels_path vision/model/label_classes.npy
```

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to the input image |
| `--text_input` | *(required)* | Free-form text mentioning an animal |
| `--ner_model_path` | `ner/ner_model` | Path to the NER model directory |
| `--cv_model_path` | `vision/model/model.h5` | Path to the CNN model file |
| `--labels_path` | `vision/model/label_classes.npy` | Path to the label classes file |

---

## Working Examples

### Example 1 — Not matching (text and image disagree)

```bash
python pipeline.py \
  --image_path data/cat.jpg \
  --text_input "Wow! That cat is so cute!!"
```

```
Text animal: cat
Image prediction: tiger (0.1869)
The predictions from both models are different.
False
```



---

## Edge Cases

### NER Model

**Multi-token animal names (e.g. "polar bear", "bald eagle")**

The current label set only contains `B-ANIMAL`. The `extract_animal` function does handle `I-ANIMAL` tokens, but the model was not trained with `I-ANIMAL` in `LABEL_LIST`, so it will never predict it. As a result, only the first token of a multi-word animal is captured and the rest is silently dropped.

```
Input:  "There is a polar bear outside."
Output: animals = ['polar']   ← 'bear' is lost
```

**Fix:** Add `"I-ANIMAL"` to `LABEL_LIST` and retrain with properly annotated multi-token spans.

---

**False positive animal tags (non-animal words tagged as B-ANIMAL)**

If the NER model is undertrained or the sentence is out-of-distribution, common nouns can be incorrectly tagged. This was observed in testing:

```
Input:  "The rabbit hopped across the garden"
Output: animals = ['rabbit', 'garden']   ← 'garden' is a false positive
```

The pipeline takes only the first detected animal (`animals[0]`), which mitigates but does not eliminate this issue. More training data and longer fine-tuning reduce false positives.

---

**Multiple animals in text**

When the text contains more than one animal name, only the first detected one is forwarded to the pipeline comparison. The rest are ignored silently.

```
Input:  "A cat and a dog are in the picture."
NER:    animals = ['cat', 'dog']
Pipeline uses: 'cat'   ← 'dog' is discarded
```

---



### Vision Model

**Low-confidence predictions**

The CNN outputs a softmax probability over all classes and always picks the argmax, even when confidence is very low. There is no rejection threshold, so an ambiguous or out-of-distribution image still produces a confident-looking label.

```
Image:  cat.jpg
Output: tiger (0.1869)   ← low confidence, wrong prediction
```

Consider adding a minimum confidence threshold (e.g., 0.5) and returning `"unknown"` when the threshold is not met, which would cause the pipeline to return `False` rather than a wrong match.

---

**Animal class not seen during training**

If the image contains an animal that was not in the training dataset, the model will silently predict whichever training class it most resembles. There is no out-of-distribution detection.

---





# MNIST Digit Classification

A modular MNIST handwritten digit classification system implementing three machine learning approaches: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network.


## Project Structure

```
task1/
├── interfaces/
│   ├── __init__.py
│   └── mnist_classifier_interface.py   # Abstract base class
├── model/
│   ├── __init__.py
│   ├── cnn_classifier.py               # CNN implementation
│   ├── nn_classifier.py                # Feed-forward NN implementation
│   └── rf_classifier.py                # Random Forest implementation
├── notebooks/
│   └── EDA_Handwritten_Digit_Recognition_(MNIST).ipynb
├── logs/                               # Auto-generated experiment results
├── main.py                             # Entry point
├── mnist_classifier.py                 # Unified classifier wrapper
├── pyproject.toml
└── requirements.txt
```

## Models

### Random Forest (`rf`)
Scikit-learn `RandomForestClassifier` with 100 estimators. Input images are flattened from `(28, 28)` to `(784,)` before training.

### Neural Network (`nn`)
Three-layer feed-forward network built with Keras:
- Input: 784 units
- Hidden: Dense(128, ReLU) → Dense(64, ReLU)
- Output: Dense(10, Softmax)

Trained for 5 epochs with batch size 32, Adam optimizer.

### CNN (`cnn`)
Convolutional network built with Keras:
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)

Input reshaped to `(28, 28, 1)`. Trained for 5 epochs with batch size 32, Adam optimizer.

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/task1.git
cd task1
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Running

**Basic run:**
```bash
python main.py
```

Results are saved to the `logs/` directory with timestamps, e.g. `logs/cnn_20260319_104605.txt`.


## Requirements

- Python ^3.11
- TensorFlow ^2.17
- scikit-learn ^1.3
- NumPy ^1.26

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest (RF) | 97.04% |
| Neural Network (NN) | 97.39% |
| CNN | **98.92%** |

### Random Forest — `logs/rf_20260319_104301.txt`

```
Model: rf
Accuracy: 0.9704

              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.96      0.97      0.97      1032
           3       0.96      0.96      0.96      1010
           4       0.97      0.97      0.97       982
           5       0.98      0.96      0.97       892
           6       0.98      0.98      0.98       958
           7       0.97      0.96      0.97      1028
           8       0.96      0.95      0.96       974
           9       0.96      0.95      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
```

### Neural Network — `logs/nn_20260319_104352.txt`

```
Model: nn
Accuracy: 0.9739

              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.97      1032
           3       0.94      0.98      0.96      1010
           4       0.98      0.97      0.98       982
           5       0.98      0.96      0.97       892
           6       0.99      0.97      0.98       958
           7       0.97      0.98      0.97      1028
           8       0.96      0.97      0.96       974
           9       0.97      0.96      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
```

### CNN — `logs/cnn_20260319_104605.txt`

```
Model: cnn
Accuracy: 0.9892

              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       0.99      1.00      0.99      1135
           2       1.00      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       0.99      0.99      0.99       958
           7       0.98      0.99      0.99      1028
           8       1.00      0.97      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

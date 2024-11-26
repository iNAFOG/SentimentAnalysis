# Sentiment Analysis Using LSTM

This repository contains an implementation of a **Long Short-Term Memory (LSTM)** model for sentiment analysis. The goal of this project is to classify text data as positive, negative, or neutral based on its sentiment.

## Features

- Preprocessing of text data, including tokenization and padding.
- Implementation of an LSTM model for sequential data.
- Ability to handle imbalanced datasets using techniques like oversampling or weighted loss.
- Training and evaluation on labeled sentiment datasets.
- Output metrics including accuracy, precision, recall, and F1 score.

---

## Setup and Requirements

### Prerequisites

Ensure you have Python 3.7 or later installed. The following libraries are required:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `nltk`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

1. **Dataset**  
   Prepare a labeled dataset containing text and sentiment labels (e.g., positive, negative, neutral).

2. **Tokenization and Padding**  
   Text data is tokenized into sequences of integers using the `Tokenizer` class from TensorFlow/Keras. Padding is applied to ensure sequences have uniform length.

3. **Train-Test Split**  
   The dataset is split into training and testing subsets for evaluation.

---

## Model Architecture

The LSTM model is designed to process sequential data. Its architecture includes:

- **Embedding Layer**: Converts integer sequences into dense vector representations.
- **LSTM Layer**: Captures long-term dependencies in the text.
- **Dense Layer**: Fully connected layer for classification.
- **Activation**: Softmax for multi-class sentiment classification.

### Example Architecture
```plaintext
Embedding -> LSTM -> Dense -> Softmax
```

---

## Training

The model is trained using:

- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score

To train the model, execute:

```bash
python train.py
```

---

## Evaluation

The trained model is evaluated on the test dataset, with results visualized using:

- Confusion Matrix
- Precision-Recall Curve

---

## Results

The model achieves the following metrics on the test dataset:

- **Accuracy**: `X%`
- **Precision**: `Y%`
- **Recall**: `Z%`
- **F1 Score**: `A%`

---

## Usage

### Predicting Sentiment

To use the trained model for sentiment prediction on new text data:

```python
from predict import predict_sentiment

text = "This product is amazing!"
result = predict_sentiment(text)
print(result)
```

### Output
The sentiment (e.g., Positive, Negative, Neutral) is displayed.

---

## Contributing

Feel free to contribute to this project! Open issues, suggest improvements, or submit pull requests.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

Special thanks to open-source libraries and datasets that made this project possible!

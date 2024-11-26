Here’s the updated README excluding the results section:

---

# Sentiment Analysis Using LSTM with Regularization and Dropout

This repository contains two LSTM-based models for binary sentiment analysis, each designed with different configurations to explore the impact of architectural choices, regularization techniques, and hyperparameter tuning.

---

## Features

- **Two Distinct Models**:
  - **Model 1**: Uses higher-dimensional embeddings and more LSTM units for richer feature extraction.
  - **Model 2**: Reduces embedding dimensions and LSTM units for a more lightweight architecture.
- **Regularization**: Dropout and L2 regularization prevent overfitting.
- **Early Stopping**: Stops training when validation loss does not improve.
- **Evaluation**: Metrics include accuracy and loss.

---

## Setup and Requirements

### Prerequisites

Ensure you have Python 3.7 or later installed. The following libraries are required:

- `tensorflow`
- `numpy`
- `pandas`
- `sklearn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

1. **Dataset Preparation**  
   The dataset should consist of text samples with binary sentiment labels (0 or 1).

2. **Tokenization and Padding**  
   Convert text into sequences using a tokenizer and apply padding to standardize input length.

3. **Train-Validation-Test Split**  
   Split the data into training, validation, and test sets for robust evaluation.

---

## Model Architectures

### **Model 1**: Richer Feature Extraction

1. **Embedding Layer**  
   - Embedding dimension: 128

2. **LSTM Layers**  
   - First LSTM: 128 units (`return_sequences=True`)
   - Second LSTM: 64 units

3. **Dropout**  
   - Dropout rate: 50%

4. **Dense Layer**  
   - Fully connected with 64 units, ReLU activation, and L2 regularization (0.01).

5. **Output Layer**  
   - Single neuron with sigmoid activation for binary classification.

**Compilation**:  
- Optimizer: Adam (learning rate: 0.0001)  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy  

**Training**:
```python
model.fit(train_padded, y_train, epochs=10, batch_size=32, validation_data=(validation_padded, y_validation), callbacks=[early_stopping])
```

---

### **Model 2**: Lightweight and Regularized

1. **Embedding Layer**  
   - Embedding dimension: 100

2. **LSTM Layers**  
   - First LSTM: 64 units (`return_sequences=True`)
   - Second LSTM: 32 units

3. **Dropout**  
   - Dropout rate: 40%

4. **Dense Layer**  
   - Fully connected with 64 units, ReLU activation, and L2 regularization (0.005).

5. **Output Layer**  
   - Single neuron with sigmoid activation for binary classification.

**Compilation**:  
- Optimizer: Adam (learning rate: 0.00005)  
- Loss: Binary Cross-Entropy  
- Metrics: Accuracy  

**Training**:
```python
model.fit(train_padded, y_train, epochs=10, batch_size=32, validation_data=(validation_padded, y_validation), callbacks=[early_stopping])
```

---

## Usage

### Predict Sentiment

To use the trained models for predictions:

```python
prediction = model.predict(new_text_padded)
print(f"Predicted Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
```

The output will be either **Positive** or **Negative**, depending on the prediction.

---

## Contributing

Contributions are welcome! Open issues, suggest features, or submit pull requests.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

Special thanks to TensorFlow/Keras for making deep learning accessible and to the open-source community for support and inspiration!

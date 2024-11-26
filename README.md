Here’s the updated README to include the prediction functionality:

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
- **Text Preprocessing**: Cleans and lemmatizes input text for model compatibility.
- **Prediction Functionality**: Allows real-time sentiment prediction for custom input text.

---

## Setup and Requirements

### Prerequisites

Ensure you have Python 3.7 or later installed. The following libraries are required:

- `tensorflow`
- `numpy`
- `pandas`
- `sklearn`
- `nltk`
- `bs4`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Predicting Sentiment for Custom Input

The repository includes a function to predict the sentiment of custom text input using the trained LSTM model.

### Function Overview

The `predict_sentiment` function performs the following steps:

1. **Text Preprocessing**:
   - Converts text to lowercase.
   - Removes non-alphanumeric characters.
   - Removes stopwords using NLTK.
   - Cleans HTML content using `BeautifulSoup`.
   - Lemmatizes the text for better tokenization.

2. **Tokenization and Padding**:
   - Converts preprocessed text into sequences using the tokenizer.
   - Pads sequences to the required input length of the model.

3. **Prediction**:
   - Uses the trained LSTM model to predict sentiment.
   - Outputs the sentiment score and classification as **Positive**, **Neutral**, or **Negative** based on the prediction value.

### Example Usage

```python
def predict_sentiment(text):
    """Predicts the sentiment of a given text using the trained model."""
    # Preprocess the text
    text = text.lower()
    text = re.sub('[^a-z A-z 0-9-]+', '', text)
    text = " ".join([y for y in text.split() if y not in stopwords.words('english')])
    text = re.sub(r'(http|https|ftp|ssh)://[\w_-]+(?:\.[\w_-]+)+[\w.,@?^=%&:/~+#-]*', '', text)

    text = BeautifulSoup(text, 'html.parser').get_text()
    text = " ".join(text.split())
    text = lemmatize_words(text)

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='pre')

    # Make a prediction
    prediction = model.predict(padded_sequence)

    # Return the sentiment score
    return prediction[0][0]

# Example input
custom_input = input("Enter your text: ")

# Predict sentiment
sentiment = predict_sentiment(custom_input)

# Classify sentiment
if sentiment >= 0.772:
    print(sentiment)
    print("Positive sentiment")
elif sentiment < 0.772 and sentiment > 0.5:
    print(sentiment)
    print("Neutral sentiment")
else:
    print(sentiment)
    print("Negative sentiment")
```

---

### **Output Classification**

- **Positive Sentiment**: Sentiment score ≥ 0.772
- **Neutral Sentiment**: 0.5 < Sentiment score < 0.772
- **Negative Sentiment**: Sentiment score ≤ 0.5

This functionality allows you to interact with the model in real-time and test it with any input text.

---

## Contributing

Contributions are welcome! Open issues, suggest features, or submit pull requests.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

Special thanks to TensorFlow/Keras for making deep learning accessible and to the open-source community for support and inspiration!

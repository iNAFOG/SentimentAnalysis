import streamlit as st
import numpy as np
import re
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved models
model1 = tf.keras.models.load_model('model1.h5')
model2 = tf.keras.models.load_model('model2.h5')

# Load the tokenizer from the file
with open(r'tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the max_length value used during training
max_length = 100

def lemmatize_words(text):
    """Lemmatizes the words in the input text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def preprocess_text(text):
    """Preprocesses the input text."""
    text = text.lower()
    text = re.sub('[^a-z A-z 0-9-]+', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', str(text))
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = " ".join(text.split())
    return lemmatize_words(text)

def predict_sentiment_ensemble(text):
    """Predicts sentiment using an ensemble of models."""
    # Preprocess the text
    text = preprocess_text(text)

    # Tokenize and pad the sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='pre')

    # Predictions from individual models
    prediction1 = model1.predict(padded_sequence)[0][0]
    prediction2 = model2.predict(padded_sequence)[0][0]

    # Ensemble method: Averaging
    ensemble_prediction = np.mean([prediction1, prediction2])

    return ensemble_prediction

# Streamlit app
st.title("Sentiment Analysis App")

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Text area for user input
st.text_area("Your Text", value=st.session_state.user_input, key="user_input")

# Buttons for prediction and reset
col1, col2 = st.columns(2)

# Predict Button
with col1:
    if st.button("Predict Sentiment"):
        user_input = st.session_state.user_input
        if user_input.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            sentiment = predict_sentiment_ensemble(user_input)
            if sentiment >= 0.772:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("Prediction: Positive Sentiment")
            elif 0.5 < sentiment < 0.772:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("Prediction: Neutral Sentiment")
            else:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("Prediction: Negative Sentiment")

# Reset Button
with col2:
    if st.button("Reset"):
        st.session_state.user_input = ""  # Safely clear the session state input

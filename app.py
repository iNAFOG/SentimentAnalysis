import streamlit as st
import numpy as np
import re
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
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

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        font-family: Consolas,monaco,monospace;
        color: #2c3e50;
    }
    .custom-heading {
        font-family: 'Roboto', sans-serif;
        font-size: 50px;
        color: #2ecc71;
        text-align: left;
    }
    .stTextArea textarea {
        border: 1px solid #bdc3c7;
        border-radius: 10px;
        padding: 15px;
        font-family: Consolas,monaco,monospace;
        font-size: 1rem;
        color: #34495e;
        background: #ecf0f1;
    }
    .stTextArea textarea:focus {
        border-color: #3498db;
        outline: none;
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.3);
    }
    .stButton>button {
        background-color: #fff;
        color: #fc8bcf;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF13F0;
        color: #fff;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(41, 128, 185, 0.3);
    }
    .footer {
        margin-top: 50px;
        padding: 20px;
        background: linear-gradient(90deg, #74b9ff, #81ecec);
        color: #2d3436;
        text-align: center;
        border-radius: 10px;
        font-size: 14px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown('<h1 class="custom-heading">Sentiment Analysis Model</h1>', unsafe_allow_html=True)
st.markdown('<h2>Enter text for sentiment analysis</h2>', unsafe_allow_html=True)

# Initialize session state for the text input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Text area for user input
user_input = st.text_area("Your Text", st.session_state.user_input, key="user_input")

# Buttons for prediction and reset
col1, col2 = st.columns(2)
with col1:
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            sentiment = predict_sentiment_ensemble(user_input)

            if sentiment >= 0.772:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("**Prediction:** Positive Sentiment")
            elif sentiment < 0.772 and sentiment > 0.5:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("**Prediction:** Neutral Sentiment")
            else:
                st.success(f"Sentiment Score: {sentiment:.2f}")
                st.write("**Prediction:** Negative Sentiment")

with col2:
    if st.button("Reset"):
        # Reset the session state variable
        st.session_state["user_input"] = ""

st.markdown("""
    <div class="footer">
        This application uses <strong>LSTM RNNs models with ensembling</strong> for sentiment analysis.<br>
        Developed by <strong>Prasoon</strong>, <strong>Abhinav</strong>, <strong>Rayan</strong>, and <strong>Shivam</strong>.
    </div>
""", unsafe_allow_html=True)

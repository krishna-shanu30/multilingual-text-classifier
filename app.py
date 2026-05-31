import streamlit as st
import joblib
import numpy as np

st.title("Multilingual Text Classifier")

# Load models
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

language_map = {
    "pt": "Portuguese",
    "bg": "Bulgarian",
    "zh": "Chinese",
    "th": "Thai",
    "ru": "Russian",
    "pl": "Polish",
    "ur": "Urdu",
    "sw": "Swahili",
    "tr": "Turkish",
    "es": "Spanish",
    "ar": "Arabic",
    "it": "Italian",
    "hi": "Hindi",
    "de": "German",
    "el": "Greek",
    "nl": "Dutch",
    "fr": "French",
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese"
}

def predict_lang(text):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    probabilities = model.predict_proba(text_vector)

    confidence = np.max(probabilities)

    language_code = encoder.inverse_transform(prediction)[0]

    language_name = language_map.get(language_code, language_code)

    return language_name, confidence

text = st.text_area("Enter Text")

if st.button("Predict"):
    lang, conf = predict_lang(text)

    st.success(f"Predicted Language: {lang}")
    st.info(f"Confidence: {round(conf * 100, 2)}%")
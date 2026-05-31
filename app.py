#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np


# In[11]:


model= joblib.load("language_model.pkl")
vectorizer= joblib.load("tfidf_vectorizer.pkl")
encode = joblib.load("label_encoder.pkl")
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


# In[19]:


def predict_lang(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    probabilities = model.predict_proba(text_vector)

    confidence = np.max(probabilities)

    language_code = encode.inverse_transform(prediction)[0]

    language_name = language_map.get(language_code, language_code)

    return language_name, confidence
text= input("enter text:")
lang,conf= predict_lang(text)
print("Predicted Language:",lang)
print("confidence:",round(conf*100,2),"%")


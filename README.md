Multilingual Text Classifier:

A multilingual NLP-powered language detection system developed using Character N-gram TF-IDF and Logistic Regression. 
The application is capable of automatically identifying 20+ languages from user-provided text while generating confidence-based predictions through an interactive Streamlit interface.

Key Features:

1. Automatic detection of 20+ languages
2. Character-level TF-IDF feature extraction
3. Machine Learning–based text classification
4. Confidence score prediction
5. Real-time multilingual text analysis
6. Interactive and user-friendly Streamlit web application
7. Lightweight and efficient NLP pipeline without deep learning models

Tech Stack:
1. Programming Language: Python
2. Machine Learning: Scikit-learn
3. Natural Language Processing: TF-IDF Vectorization
4. Model: Logistic Regression
5. Frontend/UI: Streamlit
6. Model Serialization: Joblib
7. Version Control: Git & GitHub

Project Workflow:
1. Text preprocessing and normalization
2. Character N-gram TF-IDF vectorization
3. Feature extraction from multilingual text
4. Logistic Regression model training
5. Language prediction with probability-based confidence scoring
6. Streamlit-based deployment and user interaction

Supported Languages:

English,
French,
Spanish,
German,
Hindi,
Portuguese,
Russian,
Chinese,
Japanese,
Arabic,
Bulgarian,
Thai,
Turkish,
Urdu,
Vietnamese,
Greek,
Dutch,
Polish,
Swahili,
Italian.

Local Installation & Execution:
1. pip install -r requirements.txt
2. streamlit run app.py

Future Enhancements:
1. Support for Hinglish and code-mixed language detection
2. Improved prediction confidence using advanced NLP techniques
3. Integration with translation APIs
4. Deployment using cloud platforms
5. Addition of deep learning and transformer-based models

Project Objective:

The primary objective of this project is to demonstrate how classical NLP techniques combined with Machine Learning can effectively solve multilingual language classification tasks without relying on computationally expensive transformer architectures.

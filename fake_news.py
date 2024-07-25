import time

import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('news_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


def predict_fake_news(news):
    transformed_news = vectorizer.transform([news])
    prediction = model.predict(transformed_news)
    confidence = model.predict_proba(transformed_news)
    return prediction[0], confidence


# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter the news article text below to check if it's real or fake.")

# Text input
user_input = st.text_area("News Text", height=200, placeholder="Type or paste the news article text here...")

# Predict button
if st.button("Predict"):
    if user_input:
        with st.spinner('Analyzing...'):
            prediction, confidence = predict_fake_news(user_input)

            # Simulate delay to show spinner
            time.sleep(2)  # Adjust time delay as needed

        # Clear spinner after prediction
        st.empty()

        # Display prediction result
        st.progress(100)

        if prediction == 1:
            st.success(f"The news article is Real.")
        else:
            st.error(f"The news article is Fake.")
    else:
        st.warning("Please enter some text to analyze.")

# To run the Streamlit app, save this file and use the command:
# streamlit run <filename>.py

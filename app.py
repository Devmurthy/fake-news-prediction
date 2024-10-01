import streamlit as st
import numpy as np
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to fetch news from NewsAPI
def fetch_news(query):
    api_key = 'c3711241efb546058787c9c934b4969a'  # Replace with your actual API Key
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles')
    
    if articles:
        return [article['title'] for article in articles[:3]]  # Limit to top 3 news titles
    else:
        return ["No news articles found for this query."]

# Function to make a prediction
def predict_news(news_content):
    # Transform the news content to the correct feature shape
    news_vector = vectorizer.transform([news_content])  # Use the vectorizer to transform the input
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        return "Real News"
    else:
        return "Fake News"

# Streamlit App Interface
def main():
    st.title("Fake News Predictor")

    st.subheader("Enter a news headline or content to check if it's real or fake:")
    user_input = st.text_area("Enter news content here")

    if st.button("Predict"):
        if user_input:
            result = predict_news(user_input)
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter news content to make a prediction.")

    st.subheader("Or check the latest news:")
    news_query = st.text_input("Enter a news topic to search")

    if st.button("Fetch News"):
        if news_query:
            articles = fetch_news(news_query)
            for idx, article in enumerate(articles, 1):
                st.write(f"{idx}. {article}")
                prediction = predict_news(article)
                st.write(f"Prediction: {prediction}")
        else:
            st.warning("Please enter a topic to search news.")

if __name__ == '__main__':
    main()

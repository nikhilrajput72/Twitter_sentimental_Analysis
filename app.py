import streamlit as st
import re
import pickle
import requests
import nltk
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# === CONFIG ===
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def clean_text(text, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def predict_sentiment(text, model, tokenizer, stop_words):
    cleaned = clean_text(text, stop_words)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

def get_tweets_from_nitter(username, count=5):
    url = f"https://nitter.net/{username}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    tweets = []

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return []

        soup = BeautifulSoup(res.text, 'html.parser')
        tweet_divs = soup.find_all('div', class_='tweet-content')
        for div in tweet_divs[:count]:
            tweet_text = div.get_text(strip=True)
            if tweet_text:
                tweets.append(tweet_text)
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    
    return tweets

def create_card(tweet_text, sentiment):
    color = "#4CAF50" if sentiment == "Positive" else "#F44336"
    return f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """

# === MAIN APP ===
def main():
    st.set_page_config(page_title="Nitter Sentiment Analyzer", layout="centered")
    st.title("🐦 Twitter Sentiment Analysis ")

    stop_words = load_stopwords()
    model, tokenizer = load_model_and_tokenizer()

    option = st.selectbox("Choose input method", ["Type your own text", "Fetch tweets from Twitter user"])

    if option == "Type your own text":
        text_input = st.text_area("Enter your text:")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, tokenizer, stop_words)
            st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)

    else:
        username = st.text_input("Enter Twitter username (e.g., elonmusk)")
        if st.button("Fetch Tweets"):
            tweets = get_tweets_from_nitter(username)
            if not tweets:
                st.warning("No tweets found or user may not exist on Nitter.")
            for tweet in tweets:
                sentiment = predict_sentiment(tweet, model, tokenizer, stop_words)
                st.markdown(create_card(tweet, sentiment), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

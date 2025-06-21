# Twitter Sentiment Analysis using Nitter

This project is a deep learning-powered sentiment analysis application built using **Streamlit**, allowing users to analyze sentiments of custom text inputs or live tweets from a public Twitter user via the **Nitter** frontend (a privacy-respecting Twitter proxy). The model classifies sentiments as **Positive** or **Negative** using a pre-trained Keras model.

---

## Features

- Real-time sentiment analysis of custom user input
- Fetch and analyze recent tweets from a public Twitter account using **Nitter**
- Sentiment prediction using a trained neural network model (Keras)
- Clean, interactive interface built with **Streamlit**
- Responsive sentiment cards styled dynamically based on prediction
- Text preprocessing includes tokenization, stopword removal, and padding

---

## Technology Stack

| Component       | Purpose                                     |
|----------------|----------------------------------------------|
| Python          | Core programming language                   |
| Streamlit       | Web application framework                   |
| TensorFlow/Keras| Deep learning model for sentiment analysis  |
| NLTK            | Natural language preprocessing              |
| Nitter          | Alternative Twitter frontend (web scraping) |
| BeautifulSoup   | HTML parsing for tweet extraction           |
| Pickle          | Tokenizer deserialization                   |




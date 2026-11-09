from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset.csv"

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="wide")
st.title("💬 Twitter Sentiment Analyzer")
st.caption("Explore sentiment in sample tweets or upload your own CSV with a 'tweet' column.")

@st.cache_resource
def get_analyzer():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

def classify(score):
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"

analyzer = get_analyzer()
mode = st.radio("Choose input", ["Analyze text", "Analyze CSV"], horizontal=True)

if mode == "Analyze text":
    text = st.text_area("Enter a tweet or sentence", placeholder="I love this product!")
    if st.button("Analyze sentiment", type="primary", disabled=not text.strip()):
        score = analyzer.polarity_scores(text)["compound"]
        sentiment = classify(score)
        st.metric("Sentiment", sentiment)
        st.metric("Compound score", f"{score:.3f}")
        st.progress((score + 1) / 2)
else:
    uploaded = st.file_uploader("Upload CSV (must contain a 'tweet' column)", type="csv")
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            if "tweet" not in data.columns:
                st.error("The CSV must contain a column named 'tweet'.")
            else:
                data["tweet"] = data["tweet"].fillna("").astype(str)
                data["score"] = data["tweet"].map(lambda x: analyzer.polarity_scores(x)["compound"])
                data["sentiment"] = data["score"].map(classify)
                st.dataframe(data, use_container_width=True)
                counts = data["sentiment"].value_counts()
                st.bar_chart(counts)
                st.download_button("Download analyzed CSV", data.to_csv(index=False), "sentiment_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
else:
    if DATASET.is_file():
        sample = pd.read_csv(DATASET)
        if "tweet" in sample.columns:
            st.subheader("Sample dataset")
            st.write(sample.head())

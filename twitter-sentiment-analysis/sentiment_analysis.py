from pathlib import Path

import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset.csv"


def label(score):
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def main():
    if not DATASET.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    data = pd.read_csv(DATASET)
    if "tweet" not in data.columns:
        raise ValueError("dataset.csv must contain a 'tweet' column")

    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    data["tweet"] = data["tweet"].fillna("").astype(str)
    data["sentiment_score"] = data["tweet"].map(
        lambda tweet: analyzer.polarity_scores(tweet)["compound"]
    )
    data["sentiment"] = data["sentiment_score"].map(label)

    print(data.head())
    print(data["sentiment"].value_counts())

    data["sentiment"].value_counts().plot(kind="bar", title="Twitter Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# download nltk data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# load dataset
data = pd.read_csv("dataset.csv")

print(data.head())

# initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

sentiments = []

for tweet in data['tweet']:
    score = sia.polarity_scores(tweet)
    sentiments.append(score['compound'])

data['sentiment_score'] = sentiments

def label(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

data['sentiment'] = data['sentiment_score'].apply(label)

print(data.head())

# count results
result = data['sentiment'].value_counts()

print(result)

# plot graph
result.plot(kind='bar')
plt.title("Twitter Sentiment Analysis")
plt.show()
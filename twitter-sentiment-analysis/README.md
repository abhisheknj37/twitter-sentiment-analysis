# Twitter Sentiment Analysis

<escape>🚀 Live Demo: https://twitter-sentiment-analysis-ovp6.onrender.com</escape>  
> Streamlit demo supports single-text sentiment checks and CSV uploads with a `tweet` column.

A small Python NLP project that scores tweets with NLTK VADER and labels each row as positive, negative or neutral.

## Requirements

- Python 3.9+
- pip

## Setup

Run these commands from the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r twitter-sentiment-analysis/requirements.txt
```

## Run

```bash
cd twitter-sentiment-analysis
python sentiment_analysis.py
```

The script reads `dataset.csv`, prints a preview and sentiment counts, and opens a bar chart. The first run downloads the NLTK VADER lexicon.

## Dataset format

The CSV must contain a `tweet` column. Rows with missing tweet text are treated as empty strings.

## Notes

This is an educational baseline using lexicon-based sentiment scoring, not a trained classifier. Results should not be treated as authoritative measurements of public opinion.

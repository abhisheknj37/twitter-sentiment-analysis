FROM python:3.11-slim
WORKDIR /app
COPY twitter-sentiment-analysis/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY twitter-sentiment-analysis/ ./twitter-sentiment-analysis/
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/share/nltk_data')"
ENV NLTK_DATA=/usr/local/share/nltk_data
EXPOSE 8501
CMD ["sh","-c","streamlit run twitter-sentiment-analysis/app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]

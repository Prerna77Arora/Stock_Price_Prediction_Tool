# sentiment.py
import os
import requests
from pytrends.request import TrendReq
from dotenv import load_dotenv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------- Initialize NLP models ----------------------
finbert_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
vader_analyzer = SentimentIntensityAnalyzer()

# ---------------------- News Sentiment ----------------------
def fetch_news_sentiment(ticker):
    if not NEWS_API_KEY:
        print("⚠️ Missing NEWS_API_KEY")
        return []

    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize=10'
        resp = requests.get(url).json()
        articles = resp.get('articles', [])
        results = []

        for article in articles:
            date = article.get('publishedAt', '')[:10]
            text = (article.get('title') or '') + ". " + (article.get('description') or '')
            if not text.strip():
                continue
            pred = finbert_pipeline(text)[0]
            score_map = {"positive": 1, "neutral": 0.5, "negative": 0}
            results.append({'date': datetime.strptime(date, '%Y-%m-%d').date(), 
                            'title': article.get('title',''), 
                            'score': score_map.get(pred['label'].lower(), 0.5)})
        return results
    except Exception as e:
        print(f"News sentiment error: {e}")
        return []

# ---------------------- Twitter Sentiment ----------------------
def fetch_twitter_sentiment(tweets):
    results = []
    try:
        for t in tweets:
            date, text = t.get('date'), t.get('text')
            vs = vader_analyzer.polarity_scores(text)
            results.append({'date': datetime.strptime(date,'%Y-%m-%d').date() if date else None,
                            'text': text, 
                            'score': (vs['compound'] + 1)/2})
        return results
    except Exception as e:
        print(f"Twitter sentiment error: {e}")
        return []

# ---------------------- Google Trends ----------------------
def fetch_google_trends(ticker, company_name=None):
    try:
        pytrends = TrendReq()
        kw = company_name if company_name else ticker
        pytrends.build_payload([kw], timeframe='now 7-d')
        data = pytrends.interest_over_time()
        if not data.empty:
            return [{'date': idx.date(), 'value': int(val)} for idx, val in data[kw].items()]
        return []
    except Exception as e:
        print(f"Google Trends error: {e}")
        return []

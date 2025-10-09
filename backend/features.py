# features.py
import pandas as pd

def create_features(hist_data, news_sentiment, twitter_sentiment, trends):
    """
    Inputs:
        hist_data - historical stock prices DataFrame with 'Date' & 'Close'
        news_sentiment - list of dicts: {'date': 'YYYY-MM-DD', 'score': float}
        twitter_sentiment - list of dicts: {'date': 'YYYY-MM-DD', 'score': float}
        trends - list of dicts: {'date': 'YYYY-MM-DD', 'value': float}
    Outputs:
        DataFrame with engineered features ready for LSTM
    """
    df = hist_data.copy()

    # Moving averages
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # Daily percentage change
    df['Pct_change'] = df['Close'].pct_change()

    # Rolling volatility
    df['Volatility'] = df['Close'].rolling(10).std()

    # Convert lists to dicts for fast lookup
    news_map = {x['date']: x['score'] for x in news_sentiment}
    twitter_map = {x['date']: x['score'] for x in twitter_sentiment}
    trend_map = {x['date']: x['value'] for x in trends}

    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['News_sentiment'] = df['Date'].apply(lambda x: news_map.get(x, 0.5))  # default neutral
    df['Twitter_sentiment'] = df['Date'].apply(lambda x: twitter_map.get(x, 0.5))
    df['Trend_score'] = df['Date'].apply(lambda x: trend_map.get(x, 0.5))

    df = df.fillna(0)
    return df

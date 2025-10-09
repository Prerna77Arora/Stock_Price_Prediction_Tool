# stock_data.py
import yfinance as yf
import pandas as pd
from functools import lru_cache

def get_stock_list(category=None):
    df = pd.read_csv('data/stock_list.csv')  # columns: Ticker, Name, Sector
    if category:
        df = df[df['Sector'].str.lower() == category.lower()]
    return df

@lru_cache(maxsize=50)
def fetch_historical_prices(ticker, period='6mo', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
        return hist.reset_index()
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])

def fetch_latest_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if data.empty:
            return None
        return round(data['Close'].iloc[-1],2)
    except Exception as e:
        print(f"Error fetching latest price for {ticker}: {e}")
        return None

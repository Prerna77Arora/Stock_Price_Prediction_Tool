import streamlit as st
from backend.stock_data import get_stock_list, fetch_historical_prices, fetch_latest_price
from backend.sentiment import fetch_news_sentiment, fetch_twitter_sentiment, fetch_google_trends
from backend.features import create_features
from backend.model import predict_next_day_price
import plotly.graph_objs as go

# Page config
st.set_page_config(page_title="Stock Prediction Tool", layout="wide")
st.title("📈 Stock Prediction Tool")

# Load stocks
stocks = get_stock_list()
sectors = stocks['Sector'].unique().tolist()

# Sidebar: Select sector
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)
filtered_stocks = stocks if selected_sector == "All" else stocks[stocks['Sector'] == selected_sector]

# Sidebar: Select stock
stock_options = filtered_stocks['Name'] + " (" + filtered_stocks['Ticker'] + ")"
selected_stock_name = st.sidebar.selectbox("Select Stock", stock_options)

# Get selected stock ticker
selected_stock = filtered_stocks[
    filtered_stocks['Name'] + " (" + filtered_stocks['Ticker'] + ")" == selected_stock_name
].iloc[0]
ticker = selected_stock['Ticker']

# Header
st.header(f"{selected_stock['Name']} ({ticker})")

# Fetch data
hist = fetch_historical_prices(ticker)
latest_price = fetch_latest_price(ticker)
news = fetch_news_sentiment(ticker)
twitter = fetch_twitter_sentiment(ticker)  # placeholder safe
trends = fetch_google_trends(ticker)

# Create features & predict
df_features = create_features(hist, news, twitter, trends)
predicted_price, model_used = predict_next_day_price(ticker, df_features)

# Buy/Sell/Hold suggestion
suggestion = "Hold"
if latest_price and predicted_price:
    if predicted_price > latest_price * 1.01:
        suggestion = "Buy"
    elif predicted_price < latest_price * 0.99:
        suggestion = "Sell"

# Display key info in columns
col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", latest_price if latest_price else "N/A")
col2.metric("Predicted Price", predicted_price if predicted_price else "N/A")
col3.metric("Suggestion", suggestion)

# Historical chart with predicted price
fig = go.Figure()
if not hist.empty:
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], name='Historical'))
if predicted_price and not hist.empty:
    fig.add_trace(go.Scatter(
        x=[hist['Date'].iloc[-1]], 
        y=[predicted_price],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Predicted'
    ))
st.plotly_chart(fig, use_container_width=True)

# News sentiment in expander
with st.expander("📢 News Sentiment"):
    if news:
        for item in news:
            st.write(f"{item['title']} (Score: {item['score']:.2f})")
    else:
        st.write("No news sentiment available.")

# Twitter sentiment in expander
with st.expander("🐦 Twitter Sentiment"):
    if twitter:
        for item in twitter:
            st.write(f"{item['text']} (Score: {item['score']:.2f})")
    else:
        st.write("Twitter sentiment not available.")

# Google Trends in expander
with st.expander("🔍 Google Trends"):
    if trends:
        for i, item in enumerate(trends):
            st.write(f"{item['date']}: {item['value']}")
    else:
        st.write("No Google Trends data available.")

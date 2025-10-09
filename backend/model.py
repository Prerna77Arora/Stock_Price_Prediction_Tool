# model.py
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
FEATURES = ['MA10','MA50','Pct_change','Volatility','News_sentiment','Twitter_sentiment','Trend_score']

TRAINED_STOCKS = [f.split('_')[0] for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]

def predict_next_day_price(ticker, df):
    """
    Predict next day price using trained LSTM.
    Returns predicted price and model used.
    """
    if ticker not in TRAINED_STOCKS:
        # fallback: last close
        return round(df['Close'].iloc[-1],2), None

    try:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.save")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        X_scaled = scaler.transform(df[FEATURES])
        if len(X_scaled) < 50:
            X_input = np.pad(X_scaled, ((50 - len(X_scaled),0),(0,0)), mode='edge').reshape(1,50,len(FEATURES))
        else:
            X_input = X_scaled[-50:].reshape(1,50,len(FEATURES))

        y_scaled = model.predict(X_input)
        y_full = np.zeros((1,len(FEATURES)))
        y_full[0,0] = y_scaled[0,0]
        predicted_price = scaler.inverse_transform(y_full)[0,0]
        return round(predicted_price,2), model
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return round(df['Close'].iloc[-1],2), None

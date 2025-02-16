import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load trained LSTM model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Sidebar Navigation
st.sidebar.title("Stocky AI")
st.sidebar.markdown("### Navigation")
pages = ["Home", "Charts", "Sectors", "Stocks", "Alerts", "Portfolio", "Settings", "Watchlist", "Help", "About Us"]
for page in pages:
    st.sidebar.button(page)
st.sidebar.button("Logout")

# Main Title
st.title("Stock Sector Predictions")

# Example Stock Data (Replace with real stock market API or database)
sectors = [
    "NIFTY FIN_SERVICE", "NIFTY AUTO", "NIFTY CONSUM", "NIFTY ENERGY", 
    "NIFTY FMCG", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL", 
    "NIFTY PSUBANK", "NIFTY REALTY", "NIFTY SERVICE", "NIFTY LDX", "NIFTY BANK"
]

# Generate random closing prices for demo (Replace with real prices)
np.random.seed(42)
closing_prices = np.random.uniform(1500, 2500, len(sectors))

# Predict Next Price using LSTM Model
def predict_price(price):
    scaled_input = scaler.transform(np.array([[price]]))
    prediction = model.predict(np.array([scaled_input]))
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

# Create DataFrame with Buy/Sell signals
data = []
for sector, price in zip(sectors, closing_prices):
    pred_price = predict_price(price)
    ema_20 = "Buy ↑" if pred_price > price else "Sell ↓"
    ema_50 = "Buy ↑" if pred_price > price * 1.02 else "Sell ↓"
    ema_100 = "Buy ↑" if pred_price > price * 1.05 else "Sell ↓"
    ema_200 = "Buy ↑" if pred_price > price * 1.10 else "Sell ↓"
    data.append([sector, ema_20, ema_50, ema_100, ema_200])

df = pd.DataFrame(data, columns=["Sector", "EMA 20", "EMA 50", "EMA 100", "EMA 200"])

# Apply Color Formatting
def color_ema(val):
    return "background-color: green; color: white" if "Buy" in val else "background-color: red; color: white"

styled_df = df.style.applymap(color_ema, subset=["EMA 20", "EMA 50", "EMA 100", "EMA 200"])

# Display Table
st.dataframe(styled_df, height=600, width=800)

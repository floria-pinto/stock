import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lstm_model import predict_stock  # Assuming you have an LSTM model function

# Sidebar - Select Stock
st.sidebar.title("Stocks")
# Dropdown for selecting stock
stock_options = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFC.NS", "ICICIBANK.NS"]
selected_stock = st.sidebar.selectbox("Select Stock", stock_options)

# Function to fetch stock data
def fetch_stock_data(stock_symbol, period="6mo"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period, interval='1d')
    return data

# Preprocess data for LSTM input (reshape, normalize if needed)
def preprocess_data(data):
    close_prices = data['Close'].values
    return close_prices.reshape(-1, 1)  # Adjust depending on your model's needs

# Display Stock Data
st.title(f"{selected_stock} Stock Analysis")
data = fetch_stock_data(selected_stock)
st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")

# Time Period Selection
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=2)
data = fetch_stock_data(selected_stock, period)

# Process data for prediction
processed_data = preprocess_data(data)

# Predict Future Prices using LSTM
try:
    predicted_prices = predict_stock(processed_data)  # Assuming this returns a list of predicted prices
    if len(predicted_prices) == 0:
        raise ValueError("No predictions returned.")
    data['Prediction'] = np.nan  # Placeholder for predictions
    # Assign predictions to the last n values
    last_n = len(predicted_prices)
    data.iloc[-last_n:, data.columns.get_loc('Prediction')] = predicted_prices
except Exception as e:
    st.error(f"Error predicting stock prices: {e}")
    data['Prediction'] = np.nan  # Fallback to NaN if prediction fails

# Plot Stock Chart with Closing and Predicted Prices
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Market Data'
))

# Plot Closing Price
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='black')))

# Plot Predicted Price (if available)
if not data['Prediction'].isna().all():
    fig.add_trace(go.Scatter(x=data.index, y=data['Prediction'], mode='lines', name='Predicted Price', line=dict(color='blue')))

fig.update_layout(title=f"{selected_stock} Price Chart", xaxis_title='Date', yaxis_title='Stock Price (₹)')
st.plotly_chart(fig)

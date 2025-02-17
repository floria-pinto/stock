import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lstm_model import predict_stock  # Assuming you have an LSTM model function

# Sidebar - Select Stock
st.sidebar.title("Stocks")
selected_stock = st.sidebar.text_input("Enter Stock Symbol (e.g., TCS.NS)", "TCS.NS")

# Fetch stock data function
def fetch_stock_data(stock_symbol, period="6mo"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period, interval='1d')
    return data

# Preprocess data for LSTM input
def preprocess_data(data):
    # Assume you're using the 'Close' price for predictions
    close_prices = data['Close'].values
    # Reshape or normalize data if needed for your model here
    # Example: Convert data into a 2D array (this depends on your model)
    return close_prices.reshape(-1, 1)  # For simplicity, assume this is sufficient

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
    # Assign predictions to last n values
    last_n = len(predicted_prices)
    data.iloc[-last_n:, data.columns.get_loc('Prediction')] = predicted_prices
except Exception as e:
    st.error(f"Error predicting stock prices: {e}")
    data['Prediction'] = np.nan  # Fallback to NaN if prediction fails

# Buy/Sell Signal Logic
short_window = 9
long_window = 21
data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
data['Buy_Signal'] = (data['Short_MA'] > data['Long_MA']) & (data['Short_MA'].shift(1) <= data['Long_MA'].shift(1))
data['Sell_Signal'] = (data['Short_MA'] < data['Long_MA']) & (data['Short_MA'].shift(1) >= data['Long_MA'].shift(1))

# Plot Stock Chart
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

# LSTM Predictions (if available)
if not data['Prediction'].isna().all():
    fig.add_trace(go.Scatter(x=data.index, y=data['Prediction'], mode='lines', name='LSTM Prediction', line=dict(color='blue')))

# Moving Averages
fig.add_trace(go.Scatter(x=data.index, y=data['Short_MA'], mode='lines', name='Short MA', line=dict(color='green')))
fig.add_trace(go.Scatter(x=data.index, y=data['Long_MA'], mode='lines', name='Long MA', line=dict(color='red')))

# Buy/Sell Markers
fig.add_trace(go.Scatter(x=data.index[data['Buy_Signal']], y=data['Close'][data['Buy_Signal']], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
fig.add_trace(go.Scatter(x=data.index[data['Sell_Signal']], y=data['Close'][data['Sell_Signal']], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

fig.update_layout(title=f"{selected_stock} Price Chart", xaxis_title='Date', yaxis_title='Stock Price (₹)')
st.plotly_chart(fig)

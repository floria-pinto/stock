import streamlit as st
import yfinance as yf
import pandas as pd

# Sidebar - Select Stock
st.sidebar.title("Stocks")
selected_stock = st.sidebar.text_input("Enter Stock Symbol (e.g., TCS.NS)", "TCS.NS")

# Function to fetch stock data
def fetch_stock_data(stock_symbol, period="6mo"):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period=period, interval='1d')
    return data

# Fetch stock data
data = fetch_stock_data(selected_stock, period="6mo")

# Check if data is empty
if data.empty:
    st.error("No data found for the selected stock symbol.")
else:
    # Display Stock Name & Current Price
    st.title(f"{selected_stock} Stock Analysis")
    st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")

import pandas as pd
import numpy as np
import yfinance as yf

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    ma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    data['Upper Band'] = ma + (std * 2)
    data['Lower Band'] = ma - (std * 2)
    data['MA20'] = ma

# Function to calculate MACD
def calculate_macd(data):
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Main stock analysis function
def analyze_stocks():
    stock_list = ['BANKNIFTY.NS', 'NIFTY50.NS', 'RELIANCE.NS']  # Add more stocks if needed
    analysis_results = {}
    
    for stock in stock_list:
        data = yf.download(stock, period='6mo', interval='1d')  # Fetch last 6 months of data
        calculate_bollinger_bands(data)
        data['RSI'] = calculate_rsi(data)
        calculate_macd(data)
        
        # Generate suggestions based on analysis
        data['Suggestion'] = np.where((data['RSI'] < 30) & (data['Close'] < data['Lower Band']), 'Buy', 
                            np.where((data['RSI'] > 70) & (data['Close'] > data['Upper Band']), 'Sell', 'Hold'))
        analysis_results[stock] = data[['Close', 'MA20', 'Upper Band', 'Lower Band', 'RSI', 'MACD', 'Signal Line', 'Suggestion']]
    return analysis_results

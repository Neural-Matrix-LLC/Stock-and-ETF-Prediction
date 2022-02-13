import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

# Normalize Data for LSTM
def normalize(data):
    logging.info(f'Normalize data')
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaler, scaled_data

# Generate returns for Volatiltiy models
def returns(data):
    logging.info(f'Generate returns')
    returns = 100 * df['Close'].pct_change().dropna()
    n = int(len(returns)*0.01)
    split_date = returns[-n:].index
    return returns, n, split_date
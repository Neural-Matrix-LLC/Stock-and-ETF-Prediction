import numpy as np 
import pandas as pd
import logging

# Generate returns for Volatiltiy models
def returns(close, split=0.01):
    logging.info(f'Generate returns')
    returns = 100 * close.pct_change().dropna()
    n = int(len(returns)*plit)
    split_date = returns[-n:].index
    return returns, n, split_date

# Compute realized volatility
def realized_vol(returns, rolling=5):
    realized_vol = returns.rolling(rolling).std()
    realized_vol = pd.DataFrame(realized_vol)
    realized_vol.reset_index(drop=True, inplace=True)
    returns_svm = returns ** 2
    returns_svm = returns_svm.reset_index()
    del returns_svm['index']

    X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
    X = X[4:].copy()
    X = X.reset_index()
    X.drop('index', axis=1, inplace=True)

    realized_vol = realized_vol.dropna().reset_index()
    realized_vol.drop('index', axis=1, inplace=True)
    return realized_vol, X
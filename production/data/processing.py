import pandas as pd
import logging

# Generate returns for Volatiltiy models
def returns(close, split=0.01):
    try:
        logging.info(f'Generate returns')
        returns = 100 * close.pct_change().dropna()
        n = int(len(returns)*split)
        split_date = returns[-n:].index
        return returns, n, split_date
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)

# Compute realized volatility
def realized_vol(returns, rolling_window):
    try:
        logging.info(f'Generate X and realized_vol with rolling window {rolling_window}')
        realized_vol = returns.rolling(rolling_window).std()
        returns_svm = returns ** 2
        X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
        X = X[4:].copy()
        X.reset_index(drop=True, inplace=True)
        realized_vol.dropna(inplace=True)
        return X, realized_vol
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)
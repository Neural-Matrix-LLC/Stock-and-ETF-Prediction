import pandas as pd
import logging
from os import path
from datetime import date
from data import data, processing
from model import garch, svr, mlp, lstm
from tensorflow import keras

logging.basicConfig(filename='logging/app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def garch_predict(symbol, returns):
    try:
        dpath = f"model/params/garch/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            params = pd.read_json(dpath)
        else:
            params = garch.tune(symbol, returns)
        garch_predict = garch.predict(returns, params)
        return garch_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def svr_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/svr/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            params = pd.read_json(dpath)
        else:
            params = svr.tune(X, realized_vol)
        svr_predict = svr.predict(symbol, X, realized_vol, params)
        return svr_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def mlp_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/mlp/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            params = pd.read_json(dpath)
        else:
            params = mlp.tune(symbol, X, realized_vol)
        mlp_predict = mlp.predict(symbol, X, realized_vol, params)
        return mlp_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def lstm_predict(symbol, close):
    try:
        dpath = f"params/lstm/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load LSTM model from {dpath}')
            model = keras.models.load_model(dpath)
        else:
            model = lstm.tune(symbol, close)
        lstm_predict = lstm.predict(model, close)
        return lstm_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def main():
    logging.info(f'Start main.py')
    try:
        symbol_list = data.load_symbols()
        logging.info(f'Loop through symbols')
        for symbol in symbol_list.Symbol:
            logging.info(f'Generate predictions for {symbol}')
            try:
                df = data.load_df(symbol)
                exchange = df.Exchange.iloc[0]

                # Data Processing
                logging.info(f'Data processing for {symbol}')
                close = df["Close"]
                returns = processing.returns(close)
                X, realized_vol = processing.realized_vol(returns, rolling_window=5)

                output_dict = {
                    "Date": date.now(),
                    "Symbol": symbol,
                    "Exchange": exchange,
                    "garch": garch_predict(symbol, returns),
                    "svr": svr_predict(symbol, X, realized_vol),
                    "mlp": mlp_predict(symbol, X, realized_vol),
                    "LSTM": lstm_predict(symbol, close)
                }
                output_df = pd.DataFrame(output_dict)
                output_df.to_csv(f'output/{symbol}_{date.now()}.csv')
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

"""
Table: DailyOutputs
Columns:
Date date PK 
Symbol varchar(45) PK 
Exchange varchar(45) PK 
predict_garch float 
predict_svr float 
predict_mlp float 
predict_LSTM float
"""

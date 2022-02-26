import pandas as pd
import logging
from os import path
from datetime import date
from data import data, processing
from model import garch, svr, mlp, lstm

logging.basicConfig(filename='logging/app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

host="143.244.188.157"
port="3306"
user="patrick-finProj"
password="Pat#21$rick"

def garch_predict(symbol, returns):
    try:
        dpath = f"model/params/garch/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            params = pd.read_json(dpath)
            garch_predict = garch.predict(returns, params)
            return garch_predict
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
            svr_predict = svr.predict(X, realized_vol, params)
            return svr_predict
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
            mlp_predict = mlp.predict(X, realized_vol, params)
            return mlp_predict
        else:
            params = mlp.tune(X, realized_vol)
            mlp_predict = mlp.predict(symbol, X, realized_vol, params)
            return mlp_predict
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
                df = data.load_df(symbol, host, port, user, password)
                exchange = df.Exchange.iloc[0]

                # Data Processing
                logging.info(f'Data processing for {symbol}')
                close = df["Close"]
                returns = processing.returns(close)
                realized_vol, X = processing.realized_vol(returns, rolling=5)
                
                # LSTM
                #lstm_predict = lstm.predict(symbol, close)
                lstm_predict = None

                output_dict = {
                    "Date": date.now(),
                    "Symbol": symbol,
                    "Exchange": exchange,
                    "garch": garch_predict(symbol, returns),
                    "svr": svr_predict(symbol, X, realized_vol),
                    "mlp": mlp_predict(symbol, X, realized_vol),
                    "LSTM": lstm_predict
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
DL_predict float 
LSTM float
"""

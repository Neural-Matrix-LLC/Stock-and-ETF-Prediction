import numpy as np
import pandas as pd
import logging
from datetime import date
from data import data, processing
from model import garch, svr, NN_vol, DL_vol, lstm

logging.basicConfig(filename='logging/app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

host="143.244.188.157"
port="3306"
user="patrick-finProj"
password="Pat#21$rick"

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

                # GJR
                dpath = f"model/params/garch/{stock_symbol}.csv"
                if path.isfile(dpath):
                    params = load_csv(dpath)
                    svr_predict = garch.predict(returns, params)
                else:
                    params = garch.tune(X, realized_vol)
                    svr_predict = garch.predict(returns, params)

                # SVR
                dpath = f"model/params/svr/{stock_symbol}.csv"
                if path.isfile(dpath):
                    params = load_csv(dpath)
                    svr_predict = svr.predict(X, realized_vol, params)
                else:
                    params = svr.tune(X, realized_vol)
                    svr_predict = svr.predict(X, realized_vol, params)
                
                # Predictions
                NN_vol_predict = NN_vol.predict(symbol)
                DL_vol_predict = DL_vol.predict(symbol)
                
                # LSTM
                lstm_predict = lstm.predict(symbol, close)

                ouptput_dict = {
                    "Date": date.now(),
                    "Symbol": symbol,
                    "Exchange": exchange,
                    "garch": garch_predict,
                    "svr": svr_predict,
                    "NN_vol": NN_vol_predict,
                    "DL_vol": DL_vol_predict,
                    "LSTM": lstm_predict
                }
                output_df = pd.DataFrame(dict)
                df.to_csv(f'output/{symbol}_{date.now()}.csv')
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
NN_predictions float 
DL_predict float 
LSTM float
"""


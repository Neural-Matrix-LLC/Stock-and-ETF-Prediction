import numpy as np
import pandas as pd
import logging
from datetime import date
from data import data, processing
from model import garch, gjrgarch, egarch, svr_linear, svr_rbf, NN_vol, DL_vol, lstm

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

                # Predictions
                garch_predict = garch.predict(data, p, q, o=0, vol='GARCH')
                gjrgarch_predict = gjrgarch.predict(symbol)
                egarch_predict = egarch.predict(symbol)

                svr_linear_predict = svr_linear.predict(symbol)
                svr_rbf_predict = svr_rbf.predict(symbol)
                NN_vol_predict = NN_vol.predict(symbol)
                DL_vol_predict = DL_vol.predict(symbol)
                lstm_predict = lstm.predict(symbol, close)

                ouptput_dict = {
                    "Date": date.now(),
                    "Symbol": symbol,
                    "Exchange": exchange,
                    "garch": garch_predict,
                    "gjrgarch": gjrgarch_predict,
                    "egarch": egarch_predict,
                    "svr_linear": svr_linear_predict,
                    "svr_rbf": svr_rbf_predict,
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
forecast_garch float 
forecast_gjrgarch float 
forecast_egarch float 
predict_svr_linear float 
predict_svr_rbf float 
NN_predictions float 
DL_predict float 
LSTM float
"""


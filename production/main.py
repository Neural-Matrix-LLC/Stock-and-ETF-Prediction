import pandas as pd
import logging
from os import path
from datetime import date
from data import data, processing
from model import garch, svr, mlp, lstm
from tensorflow import keras
import json

logging.basicConfig(filename='logging/app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def garch_predict(symbol, returns):
    try:
        dpath = f"model/params/garch/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = garch.tune(symbol, returns)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best garch parameters to {dpath}')
        garch_predict = garch.predict(returns, params)
        return garch_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def svr_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/svr/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = svr.tune(X, realized_vol)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best SVR parameters to {dpath}')
        svr_predict = svr.predict(X, realized_vol, params)
        return svr_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def mlp_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/mlp/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = mlp.tune(X, realized_vol)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best MLP parameters to {dpath}')           
        mlp_predict = mlp.predict(X, realized_vol, params)
        return mlp_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def lstm_predict(symbol, close):
    try:
        dpath = f"params/lstm/{symbol}.h5"
        if path.isfile(dpath):
            logging.info(f'Load LSTM model from {dpath}')
            model = keras.models.load_model(dpath)
        else:
            model = lstm.tune(symbol, close)
            model.save(dpath)
            logging.info(f'Export best LSTM model to {dpath}')
        lstm_predict = lstm.predict(model, close)
        return lstm_predict
    except Exception as e:
        logging.error("Exception occurred", exc_info=True) 

def main():
    logging.info(f'Start main.py')
    try:
        today = date.today()
        today = today.strftime("%Y-%m-%d")
        symbol_list = data.load_symbols()
        logging.info(f'Loop through symbols')

        rowlist = []
        for symbol in symbol_list.Symbol:
            logging.info(f'Generate predictions for {symbol}')
            try:
                df = data.load_df(symbol)
                exchange = df.Exchange.iloc[0]

                # Data Processing
                logging.info(f'Data processing for {symbol}')
                close = df["Close"]
                returns, n, split_date = processing.get_returns(close)
                X, realized_vol = processing.get_realized_vol(returns, rolling_window=5)

                output_dict = {
                    "Date": today,
                    "Symbol": symbol,
                    "Exchange": exchange,
                    "garch": garch_predict(symbol, returns),
                    "svr": svr_predict(symbol, X, realized_vol),
                    "mlp": mlp_predict(symbol, X, realized_vol),
                    "LSTM": lstm_predict(symbol, close)
                }
                rowlist.append(output_dict)
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)
        output_df = pd.DataFrame(rowlist)
        output_df.to_csv(f'daily_output/predict_{run_time}.csv')
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    main()

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

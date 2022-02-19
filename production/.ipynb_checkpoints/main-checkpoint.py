 import logging
from datetime import date
from data import data, processing
from model import garch, gjrgarch, egarch, svr_linear, svr_rbf, NN_vol, DL_vol, lstm

logging.basicConfig(filename='logging/app.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

host="143.244.188.157"
port="3306"
user="patrick-finProj"
password="Pat#21$rick"

def returns(close, ):
    logging.info(f'Generate returns')
    returns = 100 * close.pct_change().dropna()
    n = int(len(returns)*0.01)
    split_date = returns[-n:].index
    return returns, n, split_date

def main():
    logging.info(f'Start main.py')
    symbols = data.load_symbols()
    logging.info(f'Loop through symbols')
    for symbol in symbols:
        data = data.load_df(symbol, host, port, user, password)
        lstm_predict = lstm(symbol, df["Close"])
    
        ouptput_dict = {
            "Date": date.now(),
            "Symbol": symbol,
            "Exchange": None,
            "garch": garch_predict,
            "gjrgarch": gjr_predict,
            "egarch": egarch_predict,
            "svr_linear": svr_linear_predict,
            "svr_rbf": svr_rbf_predict,
            "NN": NN_predict,
            "DL": DL_predict,
            "LSTM": lstm_predict
        }
        output_df = pd.DataFrame(dict)
        df.to_csv(f'{symbol}_{date.now()}.csv') 

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

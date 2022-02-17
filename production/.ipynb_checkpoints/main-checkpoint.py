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
    symbols = data.fetch_symbols()
    logging.info(f'Loop through symbols')
    for symbol in symbols:
        data = data.fetch_df(symbol, host, port, user, password)
        df["Close"]
    

    
    #output_csv = output.to_csv(f"{date}
    
    return output_csv

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

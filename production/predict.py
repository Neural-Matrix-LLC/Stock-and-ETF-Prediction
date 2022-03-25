from os import path
import mysql.connector
import pandas as pd
import logging
from datetime import date, timedelta, datetime

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

logging.basicConfig(filename='logging/predict.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# GLOBAL VARIABLES
HOST="143.244.188.157"
PORT="3306"
USER="patrick-finProj"
PASSWORD="Pat#21$rick"

def load_daily_outputs():
    logging.info(f'Load data from DailyOutputs table in MySQL.')
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database="MarketPredict"
        )
        query = f"SELECT * FROM DailyOutputs;"
        dailyoutput = pd.read_sql(query, conn)
        conn.close()
        return dailyoutput
    except Exception as e:
        logging.error("Exception occurred at load_daily_outputs()", exc_info=True)

def load_prev_date(prev_date):
    logging.info(f'Load data on {prev_date} from histdailyprice3 table in MySQL.')
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database="GlobalMarketData"
        )
        query = f"SELECT Date, Symbol, Exchange, Close, Open, High, Low, Volume from histdailyprice3 WHERE Date='{prev_date}';"
        histdailyprice3 = pd.read_sql(query, conn)
        conn.close()
        return histdailyprice3
    except Exception as e:
        logging.error("Exception occurred at load_prev_date()", exc_info=True)

def prev_weekday(adate):
    try:
        _offsets = (3, 1, 1, 1, 1, 1, 2)
        prev_date = adate - timedelta(days=_offsets[adate.weekday()])
        logging.info(f'Previous market date is {prev_date}.')
        return prev_date
    except Exception as e:
        logging.error("Exception occurred at prev_weekday()", exc_info=True)

def get_price_movement(change):
    logging.info(f'Get price movement.')
    try:
        if change > 0:
            return 1
        elif change < 0:
            return -1
        else:
            return 0
    except Exception as e:
        logging.error("Exception occurred at get_price_movement()", exc_info=True)

def get_above_threshold(volatility, threshold):
    logging.info(f'Check whether volatility is above {threshold}%.')
    try:
        if volatility > threshold:
            return True
        else:
            return False
    except Exception as e:
        logging.error("Exception occurred at get_above_threshold()", exc_info=True)

def get_prediction(predict_df):
    logging.info(f'Get prediction from price movement and whether it is above threshold%.')
    try:
        if predict_df["price_movement"] == 1 and predict_df["above_threshold"]:
            return 1
        elif predict_df["price_movement"] == -1 and predict_df["above_threshold"]:
            return -1
        else:
            return 0
    except Exception as e:
        logging.error("Exception occurred at get_prediction()", exc_info=True)

def main():
    logging.info(f'Start predict.py')
    try:
        threshold = 2 # Percent
        today = date.today()
        today = today.strftime("%Y-%m-%d")
        
        dailyoutput_df = load_daily_outputs()
        predictDate = datetime.strptime(dailyoutput_df.loc[0, 'Date'], "%Y-%m-%d")
        
        prev_date = prev_weekday(predictDate).strftime("%Y-%m-%d")
        prev_date_df = load_prev_date(prev_date)
        predict_df = dailyoutput_df.merge(prev_date_df, on='Symbol')
        predict_df["close_change"] = predict_df.loc[:, 'LSTM'] - predict_df.loc[:, 'Close']
        predict_df["price_movement"] = predict_df["close_change"].apply(get_price_movement)
        predict_df["volatility"] = predict_df[['garch', 'svr', 'mlp']].mean(axis=1)
        predict_df["above_threshold"] = predict_df["volatility"].apply(lambda x: get_above_threshold(x, threshold))
        predict_df["prediction"] = predict_df.apply(get_prediction, axis=1)
        predict_df.to_csv(f'predict_final/predict_{today}.csv')
        logging.info(f'Exported predict_{today}.csv')
    except:
        logging.error("Exception occurred at main()", exc_info=True)
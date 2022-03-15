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
        logging.error("Exception occurred", exc_info=True)

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
        logging.error("Exception occurred", exc_info=True)

def get_price_movement(change):
    if change > 0:
        return 1
    elif change < 0:
        return -1
    else:
        return 0

def main():
    dailyoutput_df = load_daily_outputs()
    predictDate = datetime.strptime(dailyoutput_df.loc[0, 'Date'], "%Y-%m-%d")
    _offsets = (3, 1, 1, 1, 1, 1, 2)
    def prev_weekday(adate):
        return adate - timedelta(days=_offsets[adate.weekday()])
    prev_date = prev_weekday(predictDate).strftime("%Y-%m-%d")
    prev_date_df = load_prev_date(prev_date)
    predict_df = dailyoutput_df.merge(prev_date_df, on='Symbol')
    predict_df["close_change"] = predict_df.loc[:, 'LSTM'] - predict_df.loc[:, 'Close']
    predict_df["price_movement"] = predict_df["close_change"].apply(get_price_movement)
    pass
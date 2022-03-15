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

"""
Do not need to do parameter tuning every day.

Experiment: Run on Friday

DailyOutputs (Predicting for Monday)
predict_GARCH, preidct_SVR, predict_MLP - Volatility ()
predict_LSTM - Up and Down (no. 100 timesteps)

Friday 20 Predict 21.23 21.46

predict_GARCH, preidct_SVR, predict_MLP > threshold (0.02)
predict_LSTM > previous day close 

histdailyprice3(FRIDAY) 

PredictPrecent:
+ve:    Price going up over threshold
-ve:    Price going down over threshold
0:      Neither


PredictPercent: Monday predicted close vs Friday actual close
ActualPercent: Monday actual close vs Friday actual close
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

dailyoutput = load_daily_outputs()
predictDate = datetime.strptime(dailyoutput.loc[0, 'Date'], "%Y-%m-%d")
_offsets = (3, 1, 1, 1, 1, 1, 2)
def prev_weekday(adate):
    return adate - timedelta(days=_offsets[adate.weekday()])
prev_date = prev_weekday(predictDate).strftime("%Y-%m-%d")

def load_prev_date():
    logging.info(f'Load data from histdailyprice3 table in MySQL.')
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


"""
March 11 2022
100 days before =~ December 10 2022 for X_test
Model is complete.

(100, 1)
Feed in and get one output. Monday March 14 2022.

"""
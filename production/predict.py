from os import path
import mysql.connector
import pandas as pd
import logging

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
A: Price going up over threshold
B: Price going down over threshold
C: Neither
"""


def load_daily_outputs():
    HOST="143.244.188.157"
    PORT="3306"
    USER="patrick-finProj"
    PASSWORD="Pat#21$rick"
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database="MarketPredict"
        )
        query = f"SELECT * FROM DailyOutputs;"
        marketpredict = pd.read_sql(query, conn)
        conn.close()
        df = marketpredict.copy()
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
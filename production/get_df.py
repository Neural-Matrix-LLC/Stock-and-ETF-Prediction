import os
import sys
import mysql.connector
import numpy as np 
import pandas as pd
import logging

HOST="143.244.188.157"
PORT="3306"
USER="patrick-finProj"
PASSWORD="Pat#21$rick"

def get_df():
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database="GlobalMarketData"
        )
        query = f"SELECT Date, Close, Open, High, Low, Volume from histdailyprice3 WHERE Symbol='{stock_symbol}';"
        histdailyprice3 = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        conn.close()
        logging.error("Exception occurred", exc_info=True)
        
    return histdailyprice




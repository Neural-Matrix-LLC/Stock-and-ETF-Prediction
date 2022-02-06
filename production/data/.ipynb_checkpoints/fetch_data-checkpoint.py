import os
import sys
import mysql.connector
import numpy as np 
import pandas as pd
import logging

def get_df(stock_symbol, host, port, user, password):
    logging.info(f'Fetch data.')
    try: 
        conn = mysql.connector.connect(
            host,
            port,
            user,
            password,
            database="GlobalMarketData"
        )
        query = f"SELECT Date, Close, Open, High, Low, Volume from histdailyprice3 WHERE Symbol='{stock_symbol}';"
        histdailyprice3 = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        conn.close()
        logging.error("Exception occurred at get_df()", exc_info=True)
        
    return histdailyprice

import os
import sys
import mysql.connector
import numpy as np 
import pandas as pd
import logging

def fetch_symbols():
    """
    Return list of stock symbols.
    """
    
    try:
        logging.info(f'Fetch symbols.')
        stock_list = pd.read_csv("../stocks_and_etfs/stock_list.csv")
        #etf_list = pd.read_csv("../stocks_and_etfs/etf_list.csv")
        symbols = list(stock_list.iloc[:,0])
        return symbols       
    except Exception as e:
        conn.close()
        logging.error("Exception occurred", exc_info=True)

def fetch_df(stock_symbol, host, port, user, password):
    """
    Return dataframe from histdailyprice3
    """
    try: 
        logging.info(f'Fetch data.')
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
        return histdailyprice
    except Exception as e:
        conn.close()
        logging.error("Exception occurred at get_df()", exc_info=True)

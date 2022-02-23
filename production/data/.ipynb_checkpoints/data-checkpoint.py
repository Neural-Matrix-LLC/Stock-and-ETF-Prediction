import os
from os import path
import sys
import mysql.connector
import numpy as np 
import pandas as pd
import logging

def load_symbols():
    """
    Return list of stock symbols.
    """
    try:
        logging.info(f'Fetch symbols.')
        stock_list = pd.read_csv("../stocks_and_etfs/stock_list.csv")
        etf_list = pd.read_csv("../stocks_and_etfs/etf_list.csv")
        symbol_list = stock_list.append(etf_list).rename({"0": "Symbol"}, axis=1).reset_index(drop=True)
        return symbol_list
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def load_df(stock_symbol, host, port, user, password):
    """
    Return dataframe from histdailyprice3
    """
    dpath="../histdailyprice3/{}.csv".format(stock_symbol)
    if path.isfile(dpath):
        try:
            logging.info("Load data from {}".format(dpath))
            df = pd.read_csv(dpath)
        except Exception as e:
            logging.error("Exception occurred at load_df()", exc_info=True)
    else:
        logging.info(f'Load data from MySQL.')
        try: 
            conn = mysql.connector.connect(
                host,
                port,
                user,
                password,
                database="GlobalMarketData"
            )
            query = f"SELECT Date, Exchange, Close, Open, High, Low, Volume from histdailyprice3 WHERE Symbol='{stock_symbol}';"
            histdailyprice3 = pd.read_sql(query, conn)
            conn.close()
            df = histdailyprice3.copy()
            df.to_csv(dpath, index=False)
            return df
        except Exception as e:
            logging.error("Exception occurred at load_df()", exc_info=True)
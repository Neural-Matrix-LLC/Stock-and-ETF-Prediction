from os import path
import mysql.connector
import pandas as pd
import logging

def load_csv(dpath):
    try:
        logging.info(f"Load data from {dpath}")
        df = pd.read_csv(dpath)
        return df
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def load_symbols():
    """
    Return list of stock symbols.
    """
    try:
        logging.info(f'Fetch symbols in stocks_and_etfs/.')
        stock_list = pd.read_csv("data/stocks_and_etfs/stock_list.csv")
        etf_list = pd.read_csv("data/stocks_and_etfs/etf_list.csv")
        symbol_list = stock_list.append(etf_list).rename({"0": "Symbol"}, axis=1).reset_index(drop=True)
        return symbol_list
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def load_df(stock_symbol):
    """
    Return dataframe from histdailyprice3
    """
    HOST="143.244.188.157"
    PORT="3306"
    USER="patrick-finProj"
    PASSWORD="Pat#21$rick"

    dpath = f"histdailyprice3/{stock_symbol}.csv"
    if path.isfile(dpath):
        logging.info(f'Load data from {dpath}.')
        return load_csv(dpath)
    else:
        logging.info(f'Load data from histdailyprice3 table in MySQL.')
        try: 
            conn = mysql.connector.connect(
                host=HOST,
                port=PORT,
                user=USER,
                password=PASSWORD,
                database="GlobalMarketData"
            )
            query = f"SELECT Date, Exchange, Close, Open, High, Low, Volume from histdailyprice3 WHERE Symbol='{stock_symbol}';"
            histdailyprice3 = pd.read_sql(query, conn)
            conn.close()
            df = histdailyprice3.copy()
            df.to_csv(dpath, index=False)
            return df
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

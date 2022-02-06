import numpy as np 
import pandas as pd
import logging

stock_list = pd.read_csv("../stocks_and_etfs/stock_list.csv")
etf_list = pd.read_csv("../stocks_and_etfs/etf_list.csv")

def get_stock(stock_symbol, host, port, user, password):
    logging.info(f'Fetch symbol.')
    try: 
        
    except Exception as e:
        conn.close()
        logging.error("Exception occurred", exc_info=True)
        
    return stock_symbol

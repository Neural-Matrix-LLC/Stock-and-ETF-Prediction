import numpy as np 
import pandas as pd
import logging

stock_list = pd.read_csv("../stocks_and_etfs/stock_list.csv")
etf_list = pd.read_csv("../stocks_and_etfs/etf_list.csv")

def get_stock(stock_symbol, host, port, user, password):
    logging.info(f'Fetch symbol.')
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
        logging.error("Exception occurred", exc_info=True)
        
    return stock_symbol

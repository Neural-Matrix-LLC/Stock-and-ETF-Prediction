from os import path
from os import environ
import mysql.connector
import pandas as pd
import logging
from sqlalchemy import create_engine

DailySize = 300 # size of data to run daily predict

def get_Max_date(dbntable):
    try:
        HOST=environ.get("DBHOST")
        PORT=environ.get("DBPORT")
        USER=environ.get("DBUSER")
        PASSWORD=environ.get("DBPWD")
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD
        )
        query = f"SELECT max(Date) as maxdate from {dbntable} ;"
        logging.info(f'load_df query:{query}')
        df = pd.read_sql(query, conn)
        max_date = df.maxdate.iloc[0]
        return max_date
    except Exception as e:
        logging.error("Exception occurred at load_csv()", exc_info=True)

def load_csv(dpath):
    try:
        logging.info(f"Load data from {dpath}")
        df = pd.read_csv(dpath)
        return df
    except Exception as e:
        logging.error("Exception occurred at load_csv()", exc_info=True)

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
        logging.error("Exception occurred at load_symbols()", exc_info=True)

def load_df(stock_symbol, DailyMode=True, lastdt=None):
    """
    Return dataframe from histdailyprice3
    """
    HOST=environ.get("DBHOST")
    PORT=environ.get("DBPORT")
    USER=environ.get("DBUSER")
    PASSWORD=environ.get("DBPWD")
    DBNAME=environ.get("DBMKTDATA")
    
    dpath = f"histdailyprice3/{stock_symbol}.csv"
    if path.isfile(dpath) and (not DailyMode):
        logging.info(f'Load data from {dpath}.')
        return load_csv(dpath)
    else:
        logging.info(f'Load data from histdailyprice3 table in MySQL, Daily mode: {DailyMode}.')
        try: 
            conn = mysql.connector.connect(
                host=HOST,
                port=PORT,
                user=USER,
                password=PASSWORD,
                database=DBNAME
            )
            if DailyMode:
                nlimit = f" order by Date desc limit {DailySize}"
            else:
                nlimit = ""
            if lastdt is not None:
                dlimit = f" and Date<='{lastdt}'"
            else:
                dlimit =""
            query = f"SELECT Date, Exchange, Close, Open, High, Low, Volume from histdailyprice3 WHERE Symbol='{stock_symbol}'{dlimit} {nlimit};"
            logging.info(f'load_df query:{query}')
            histdailyprice3 = pd.read_sql(query, conn)
            conn.close()
            df = histdailyprice3.copy()
            df = df.sort_values(by=['Date'])
            if (not DailyMode):
                df.to_csv(dpath, index=False)

            return df
        except Exception as e:
            logging.error("Exception occurred at load_df()", exc_info=True)

def StoreDailyOutput(df):
    try:
        logging.info(f'StoreEOD size: {len(df)}')
        hostname=environ.get("DBHOST")
        PORT=environ.get("DBPORT")
        USER=environ.get("DBUSER")
        PASSWORD=environ.get("DBPWD")
        dbname=environ.get("DBPREDICT")
        table=environ.get("TBLDAILYOUTPUT")

        dbpath = "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=hostname, db=dbname, user=USER, pw=PASSWORD)
        logging.info(f'StoreEOD to {dbpath}')
        # Create SQLAlchemy engine to connect to MySQL Database
        engine = create_engine(dbpath)

        # Convert dataframe to sql table                                   
        df.to_sql(name=table, con=engine, if_exists='append', index=False)
    except Exception as e:
        logging.error("Exception occurred at StoreDailyOutput()", exc_info=True)


def StoreEOD(eoddata, DBn, TBLn):
    try:
        logging.info(f'StoreEOD size: {len(eoddata)} in table:{TBLn} on DB:{DBn}')
        hostname=environ.get("DBHOST")
        uname=environ.get("DBUSER")
        pwd=environ.get("DBPWD")

        dbpath = "mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=hostname, db=DBn, user=uname, pw=pwd)
        logging.info(f'StoreEOD to {dbpath}')
        # Create SQLAlchemy engine to connect to MySQL Database
        engine = create_engine(dbpath)

        # Convert dataframe to sql table                                   
        eoddata.to_sql(name=TBLn, con=engine, if_exists='append', index=False)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

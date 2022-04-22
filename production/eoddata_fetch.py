import os
from os import path
from os import environ
import datetime 
from eoddata_client import EodDataHttpClient
import pandas as pd
from sqlalchemy import create_engine
# import pymysql
import logging
from dateutil.rrule import rrule, DAILY
import mysql.connector
from data import data, processing
import predict

def fetch_eoddata(quotes, exch, startD, endD):
    try:
        
        rlist = []
        for i in range(len(quotes)):
            try:
                q = quotes[i]
                rec = {"Date":q.quote_datetime, "Symbol":q.symbol, "Exchange":exch, "Close":q.close, "Open":q.open, "High":q.high, "Low":q.low,'Volume':q.volume} 
                logging.info(f'fetch eod: {rec}')
                if q.quote_datetime >= startD and q.quote_datetime <= endD:
                    rlist.append(rec)
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)
        return rlist
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def fetch_by_symbols(Sdate, Edate):
    # read csv file
    try:
        logging.info(f'fetch_by_symbols from {Sdate} to {Edate}')
        symlist = pd.read_csv("stock_list.csv", names=['Symbol','Exchange'])
        client = EodDataHttpClient(username='thomaschoi', password='905916Tc')

        datap = f'eoddata_{Sdate.date()}-{Edate.date()}.csv'

        if path.isfile(datap):
            logging.info(f'Loading EODDATA from {datap}')
            eoddf = pd.read_csv(datap)
        else:
            rows = []
            for i, r in symlist.iterrows():
                try:
                    logging.info(f'fetch EODDATA {r.Symbol},{r.Exchange} from {Sdate} to{Edate}')
                    quotes = client.symbol_history_period_by_range(exchange_code=r.Exchange, symbol=r.Symbol, 
                        start_date=Sdate, end_date=Edate, period='D')
                    # quotes = client.quote_detail(exchange_code=exch, symbol=sym)

                    r = fetch_eoddata(quotes, r.Exchange, Sdate, Edate)
                    if (r is not None):
                        logging.debug(f"Type: {type(r)}, LEN:{len(r)}")
                        rows += r
                except Exception as e:
                    logging.error("Exception occurred", exc_info=True)                        

            eoddf = pd.DataFrame(rows)
            eoddf = eoddf.sort_values(by='Date')
            eoddf.to_csv(datap,index = False)

        eoddf['Date'] = pd.to_datetime(eoddf['Date'])
        # StoreEOD(eoddf)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def daily_output_columns():
    return ["Date", "Symbol", "Exchange", "garch", "svr", "mlp", "LSTM", "prev_Close", "prediction", "volatility"]
    # return ["Date", "Symbol", "Exchange", "garch", "svr","svrRP", "mlp","mlpRP", "LSTM", "prev_Close", "prediction", "volatility"]

def init_daily_output(Edate):
    logging.info(f"eoddata.init_daily_output {Edate}")
    DBHOST = environ.get("DBHOST")
    DBPORT = environ.get("DBPORT")
    DBUSER = environ.get("DBUSER")
    DBPWD = environ.get("DBPWD")
    DBMKTDATA = environ.get("DBMKTDATA")
    DBPREDICT = environ.get("DBPREDICT")
    TBLDAILYOUTPUT = environ.get("TBLDAILYOUTPUT")
    TBLDLYPRICE = environ.get("TBLDLYPRICE")
    TBLDAILYPERF = environ.get("TBLDAILYPERF")

    try:
        Sdate = Edate-datetime.timedelta(days=5)
        logging.info(f" Reset from {Sdate} to {Edate}")
        symbol_list = data.load_symbols()
        rowlist = pd.DataFrame()
        for symbol in symbol_list.Symbol:
            query = f"SELECT Date,Symbol, Exchange, Close as prev_Close from {DBMKTDATA}.{TBLDLYPRICE} WHERE Symbol='{symbol}' and Date>='{Sdate}' and Date<='{Edate}';"
            logging.info(f'init_daily_output query:{query}')
            conn = mysql.connector.connect(host=DBHOST,port=DBPORT,user=DBUSER,password=DBPWD)
            df = pd.read_sql(query, conn)
            rowlist = rowlist.append(df, ignore_index=True)
        logging.debug(rowlist)
        data.StoreEOD(rowlist, DBPREDICT, TBLDAILYOUTPUT)  
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def get_daily_performance(Sdate, Edate):
    logging.info(f"EODDATA daily performance from {Sdate} to {Edate}")
    DBHOST = environ.get("DBHOST")
    DBPORT = environ.get("DBPORT")
    DBUSER = environ.get("DBUSER")
    DBPWD = environ.get("DBPWD")
    DBMKTDATA = environ.get("DBMKTDATA")
    DBPREDICT = environ.get("DBPREDICT")
    TBLDAILYOUTPUT = environ.get("TBLDAILYOUTPUT")
    TBLDLYPRICE = environ.get("TBLDLYPRICE")
    TBLDAILYPERF = environ.get("TBLDAILYPERF")

    try:
        mythreshold = 2 # Percent
        conn = mysql.connector.connect(host=DBHOST,port=DBPORT,user=DBUSER,password=DBPWD)
        Hdate = Sdate - datetime.timedelta(days=5)
        query = f"SELECT Date,Symbol, Exchange, garch,svr,mlp,LSTM,prev_Close, prediction, volatility from {DBPREDICT}.{TBLDAILYOUTPUT} WHERE Date>='{Hdate}' and Date<='{Edate}';"
        logging.info(f'load_df query:{query}')
        df = pd.read_sql(query, conn)
        symlist = df.Symbol.unique()
        logging.debug(f'symbol-list:{symlist}')
        outputdf = pd.DataFrame(columns=df.columns.values.tolist())
        logging.debug(f'{outputdf.columns.values.tolist()}')
        for symbol in symlist:
            closedf = df[df["Symbol"]==symbol].reset_index(drop=True)
            symdf = closedf[closedf['Date'] >= Sdate].reset_index(drop=True)
            symdf['ActualDate'] = symdf['Date'].shift(periods=-1, axis=0)
            symdf['ActualClose'] = symdf['prev_Close'].shift(periods=-1, axis=0)
            symdf = symdf[:-1]
            symdf['ActualPercent'] = abs((symdf['ActualClose'] - symdf['prev_Close'])/symdf['prev_Close']*100)
            closedf['returns'] = 100*closedf.prev_Close.pct_change()
            closedf['vol'] = closedf.returns.rolling(5).std()
            logging.debug(closedf[closedf["Date"]>=Sdate].vol)
            symdf["ActualStd"] = closedf[closedf["Date"]>=Sdate].vol
            symdf["close_change"] = symdf.loc[:, 'ActualClose'] - symdf.loc[:, 'prev_Close']
            symdf["price_movement"] = symdf["close_change"].apply(predict.get_price_movement)
            symdf["above_threshold"] = symdf["ActualPercent"].apply(lambda x: predict.get_above_threshold(x, mythreshold))
            symdf["ActualTrend"] = symdf.apply(predict.get_prediction, axis=1)
            symdf = symdf[symdf["Date"]>=Sdate]
            outputdf = outputdf.append(symdf, ignore_index=True)
            logging.debug(f"\n{symdf.head(1)}")
            logging.debug(f"\n{outputdf.tail(2)}")
        outputdf = outputdf.sort_values(by=['Date', 'Symbol','Exchange'])

        # logging.debug("**   Start close_change   **")
        # logging.debug(outputdf["close_change"])
        # logging.debug("**   Start price_movement   **")
        # logging.debug(outputdf["price_movement"])
        # logging.debug("**   Start above_threshold   **")
        # logging.debug(outputdf["above_threshold"])
        # logging.debug(outputdf["ActualTrend"])

        outputdf.drop(columns=['close_change', 'price_movement','above_threshold'], inplace=True)
        lastdt = outputdf.iloc[-1, outputdf.columns.get_loc('ActualDate')]
        outputdf.to_csv(f'daily_output/dailyPerf_{Sdate}_{lastdt}.csv',index = False)
        data.StoreEOD(outputdf, DBPREDICT, TBLDAILYPERF)  
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def fetch_by_exchanges(Sdate, exchanges):
    try:
        logging.info(f'fetch_by_exchanges {exchanges} on {Sdate}')
        MKDB=environ.get("DBMKTDATA")
        MKTBL=environ.get("TBLDLYPRICE")
        client = EodDataHttpClient(username='thomaschoi', password='905916Tc')
        datap = f'daily_output/eoddata_{Sdate.date()}.csv'

        if path.isfile(datap):
            logging.info(f'Loading EODDATA from {datap}')
            eoddf = pd.read_csv(datap)
        else: 
            rows = []
            for exch in exchanges:
                try:
                    logging.info(f'fetch EODDAT {exch} on {Sdate}')
                    quotes = client.quote_list_by_date_period(exchange_code=exch, 
                                                                date=Sdate, period='D')

                    r = fetch_eoddata(quotes, exch, Sdate, Sdate)
                    if (r is not None):
                        logging.debug(f"Type: {type(r)}, LEN:{len(r)}")
                        rows += r
                except Exception as e:
                    logging.error("Exception occurred", exc_info=True)                        
            logging.info(f'Downloaded total {len(rows)} of records') 
            eoddf = pd.DataFrame(rows)
            eoddf = eoddf.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
            eoddf = eoddf.sort_values(by='Date')
            eoddf.to_csv(datap,index = False)
        eoddf['Date'] = pd.to_datetime(eoddf['Date'])
        data.StoreEOD(eoddf, MKDB, MKTBL)  
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(filename=f'logging/eoddatafetch_{datetime.date.today()}.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    projectstart = datetime.date(2022,3,24)
    Sdate = get_Max_date("MarketPredict.DailyOutputs")+ datetime.timedelta(days=1)
    Edate = datetime.date.today()
    exchanges = ['NYSE','AMEX','NASDAQ']
    logging.info(f"EODDATA Fetch from {Sdate} to {Edate}")
    for dt in rrule(DAILY, dtstart=Sdate, until=Edate):
        if Sdate.weekday() not in range(0, 5):
            logging.info(f'{Sdate} is not on weekday')
            break
        fetch_by_exchanges(dt, exchanges)
    Sdate = get_Max_date("MarketPredict.DailyPerformance")
    if (Sdate is None):
        Sdate = datetime.date(2022,3,24)
    get_daily_performance(Sdate, Edate)

import pandas as pd
import logging
from os import path
from datetime import datetime, timedelta
import predict
import eoddata_fetch
from data import data, processing
from model import garch, svr, mlp, lstm
from tensorflow import keras
import json
from dotenv import load_dotenv
from dateutil.rrule import rrule, DAILY

def garch_predict(symbol, returns):
    try:
        dpath = f"model/params/garch/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = garch.tune(symbol, returns)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best garch parameters to {dpath}')
        garch_predict = garch.predict(returns, params)
        return garch_predict
    except Exception as e:
        logging.error(f"Exception occurred at garch_predict({symbol})", exc_info=True)

def svr_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/svr/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = svr.tune(X, realized_vol)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best SVR parameters to {dpath}')
        svr_predict = svr.predict(X, realized_vol, params)
        return svr_predict
    except Exception as e:
        logging.error(f"Exception occurred at svrPredict{symbol}", exc_info=True)

def mlp_predict(symbol, X, realized_vol):
    try:
        dpath = f"model/params/mlp/{symbol}.csv"
        if path.isfile(dpath):
            logging.info(f'Load params from {dpath}')
            with open(dpath, "r") as ifile:
                params = json.load(ifile)
        else:
            params = mlp.tune(X, realized_vol)
            with open(dpath, "w") as outfile:
                json.dump(params, outfile)
            logging.info(f'Export best MLP parameters to {dpath}')           
        mlp_predict = mlp.predict(X, realized_vol, params)
        return mlp_predict
    except Exception as e:
        logging.error(f"Exception occurred at mlp_predict({symbol})", exc_info=True)

def lstm_predict(symbol, close):
    try:
        dpath = f"model/params/lstm/{symbol}.h5"
        if path.isfile(dpath):
            logging.info(f'Load LSTM model from {dpath}')
            model = keras.models.load_model(dpath, custom_objects={'custom_loss': lstm.custom_loss})
        else:
            model = lstm.tune(symbol, close)
            model.save(dpath)
            logging.info(f'Export best LSTM model to {dpath}')
        lstm_predict = lstm.predict(model, close)
        return lstm_predict
    except Exception as e:
        logging.error(f"Exception occurred at lstm_predict({symbol})", exc_info=True) 

def checkData(backDate, symlist):
    logging.info(f'Check data on {backDate}')
    try:
        for symbol in symlist:
            df = data.load_df(symbol, True, backDate)
            lastdt = df.iloc[-1, df.columns.get_loc('Date')]
            lastdt = lastdt.strftime("%Y-%m-%d")
            print(f'Last date of {symbol} data is {lastdt}')   
            # Data Processing
            logging.info(f'Data processing for {symbol}')
            close = df["Close"]
            returns, n, split_date = processing.get_returns(close)
            X, realized_vol = processing.get_realized_vol(returns, rolling_window=5)
            df['Return'] = returns
            df['RealizedVol'] = realized_vol
            df['X0'] = X[0]
            df['X1'] = X[1]
            df.to_csv(f'check_data{lastdt}_{symbol}.csv', index=False)
    except Exception as e:
        logging.error(f"Exception occurred at checkData({backDate})", exc_info=True)

def main(RunDaily, backDate):
    logging.info(f'Start main.py Rundaily:{RunDaily}, backDate:{backDate}')
    try:
        symbol_list = data.load_symbols()
        logging.info(f'Loop through symbols')
        dlypath = f'daily_output/dailyoutput_{backDate}.csv'

        if path.isfile(dlypath):
            logging.info(f'Loading dailyoutputs from {dlypath}')
            output_df = pd.read_csv(dlypath)
        else:
            rowlist = []
            for symbol in symbol_list.Symbol:
                logging.info(f'Generate predictions for {symbol}')
                try:
                    df = data.load_df(symbol, RunDaily, backDate)
                    exchange = df.Exchange.iloc[0]

                    lastClo = df.iloc[-1, df.columns.get_loc('Close')]
                    lastdt = df.iloc[-1, df.columns.get_loc('Date')]
                    lastdt = lastdt.strftime("%Y-%m-%d")
                    logging.info(f'Last date of {symbol} data is {lastdt} and Close {lastClo}')    

                    # Data Processing
                    logging.info(f'Data processing for {symbol}')
                    close = df["Close"]
                    returns, n, split_date = processing.get_returns(close)
                    X, realized_vol = processing.get_realized_vol(returns, rolling_window=5)

                    output_dict = {
                        "Date": lastdt, # use the last date from data set
                        "Symbol": symbol,
                        "Exchange": exchange,
                        "garch": garch_predict(symbol, returns),
                        "svr": svr_predict(symbol, X, realized_vol),
                        "mlp": mlp_predict(symbol, X, realized_vol),
                        "LSTM": lstm_predict(symbol, close),
                        "prev_Close": lastClo
                    }
                    rowlist.append(output_dict)
                except Exception as e:
                    logging.error("Exception occurred", exc_info=True)
            output_df = pd.DataFrame(rowlist)
            output_df = output_df.sort_values(by=['Date','Symbol'])
            output_df = predict.predictdf(output_df)
            output_df = output_df[["Date", "Symbol", "Exchange", "garch", "svr", "mlp", "LSTM", "prev_Close", "prediction", "volatility"]]
            output_df.to_csv(f'daily_output/dailyoutput_{backDate}.csv', index=False)
            logging.info(f'Exported dailyoutput_{backDate}.csv')
        data.StoreDailyOutput(output_df)
    except Exception as e:
        logging.error("Exception occurred at main()", exc_info=True)

if __name__ == '__main__':
    load_dotenv("mysql.env") #Check path for env variables
    logging.basicConfig(filename=f'logging/main_{datetime.today().date()}.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    # when train the models again, set rDaily = False and remove all model/params files
    mktdate = data.get_Max_date("GlobalMarketData.histdailyprice3")
    Sdate = mktdate+ timedelta(days=1)
    mToday = datetime.today().date()
    exchanges = ['NYSE','AMEX','NASDAQ']
    logging.info(f"EODDATA Fetch from {Sdate} to {mToday}")
    for dt in rrule(DAILY, dtstart=Sdate, until=mToday):
        if dt.weekday() not in range(0, 5):
            logging.info(f'{dt} is not on weekday')
            break
        eoddata_fetch.fetch_by_exchanges(dt, exchanges)
    logging.info(f"Finish EODDATA Fetch!!")

    mktdate = data.get_Max_date("GlobalMarketData.histdailyprice3")
    dlyoutput = data.get_Max_date("MarketPredict.DailyOutputs")
    rDaily = True
    if dlyoutput is None:
        dlyoutput = datetime.date(2022,3,23)
    Sdate = dlyoutput+ timedelta(days=1)
    logging.info(f"Process dailyoutput from {Sdate} to {mktdate}")
    if mktdate >= Sdate:
        for dt in rrule(DAILY, dtstart=Sdate, until=mktdate):
            dt = dt.date()
            if dt.weekday() < 5: # weekday only
                main(rDaily, dt)
    logging.info(f"Finish Process dailyoutput !!")

    Sdate = data.get_Max_date("MarketPredict.DailyPerformance")
    if (Sdate is None):
        Sdate = datetime(2022,3,24).date()
    Sdate = Sdate+ timedelta(days=1)
    eoddata_fetch.get_daily_performance(Sdate, mToday)
    logging.info(f"Finish Process DailyPerformance !!")
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

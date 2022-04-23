from os import path
from os import environ
import mysql.connector
import pandas as pd
import logging
import datetime as dt
from dotenv import load_dotenv
from data import data
from model import elasticnet
from dateutil.rrule import rrule, DAILY

HOST=environ.get("DBHOST")
PORT=environ.get("DBPORT")
USER=environ.get("DBUSER")
PASSWORD=environ.get("DBPWD")
DBNAME=environ.get("DBPREDICT")
 
def load_daily_outputs():
    logging.info(f'Load data from DailyOutputs table in MySQL.')
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DBNAME
        )
        query = f"SELECT * FROM DailyOutputs;"
        dailyoutput = pd.read_sql(query, conn)
        conn.close()
        return dailyoutput
    except Exception as e:
        logging.error("Exception occurred at load_daily_outputs()", exc_info=True)

def get_price_movement(change):
    logging.info(f'Get price movement.')
    try:
        if change > 0:
            return 1
        elif change < 0:
            return -1
        else:
            return 0
    except Exception as e:
        logging.error("Exception occurred at get_price_movement()", exc_info=True)

def get_above_threshold(volatility, threshold):
    logging.info(f'Check whether volatility is above {threshold}%.')
    try:
        if volatility > threshold:
            return True
        else:
            return False
    except Exception as e:
        logging.error("Exception occurred at get_above_threshold()", exc_info=True)

def get_prediction(df):
    logging.info(f'Get prediction from price movement and whether it is above threshold%.')
    try:
        if df["price_movement"] == 1 and df["above_threshold"]:
            return 1 
        elif df["price_movement"] == -1 and df["above_threshold"]:
            return -1 
        else:
            return 0
    except Exception as e:
        logging.error("Exception occurred at get_prediction()", exc_info=True)

def main(RunDaily, backDate):
    logging.info(f'Start predict.py Rundaily:{RunDaily}, backDate:{backDate}')
    try:
        threshold = 2 # Percent
       
        dailyoutput_df = load_daily_outputs()

        dailyoutput_df["close_change"] = dailyoutput_df.loc[:, 'LSTM'] - dailyoutput_df.loc[:, 'prev_Close']
        dailyoutput_df["price_movement"] = dailyoutput_df["close_change"].apply(get_price_movement)

        dailyoutput_df["volatility"] = dailyoutput_df[['svr', 'mlp']].mean(axis=1)
        dailyoutput_df["above_threshold"] = dailyoutput_df["volatility"].apply(lambda x: get_above_threshold(x, threshold))
        dailyoutput_df["prediction"] = dailyoutput_df.apply(get_prediction, axis=1)
        
        predict_df = dailyoutput_df[["Date", "Symbol", "Exchange", "prediction"]]
        predict_df = predict_df.sort_values(by=['Date','Symbol'])  
        predict_df.to_csv(f'predict_final/predict_{backDate}.csv', index=False)
        logging.info(f'Exported predict_{backDate}.csv')
    except:
        logging.error("Exception occurred at main()", exc_info=True)

def predictdf(dailyoutput_df):
    logging.info(f'Start predict.py')
    try:
        threshold = 2 # Percent

        dailyoutput_df["close_change"] = dailyoutput_df.loc[:, 'LSTM'] - dailyoutput_df.loc[:, 'prev_Close']
        dailyoutput_df["price_movement"] = dailyoutput_df["close_change"].apply(get_price_movement)

        # Volatility aggregation for each symbol update later
        dailyoutput_df["volatility"] = dailyoutput_df[['svr', 'mlp']].mean(axis=1)
        # dailyoutput_df["volatility"] = dailyoutput_df[['garch', 'svr', 'mlp']].mean(axis=1)
        dailyoutput_df["above_threshold"] = dailyoutput_df["volatility"].apply(lambda x: get_above_threshold(x, threshold))
        dailyoutput_df["prediction"] = dailyoutput_df.apply(get_prediction, axis=1)
        return dailyoutput_df
    except:
        logging.error("Exception occurred at main()", exc_info=True)

if __name__ == '__main__':
    # when train the models again, set rDaily = False and remove all model/params files
    load_dotenv("mysql.env") #Check path for env variables
    logging.basicConfig(filename=f'logging/predict_{dt.date.today()}.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    rDaily = True
    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    preddt = data.get_Max_date("MarketPredict.StockPredict")
    if preddt is None:
        preddt = dt.date(2022,3,23)
    doutput = data.get_Max_date("MarketPredict.DailyOutputs")
    if doutput is None:
        doutput = dt.date(2022,3,23)
    logging.info(f"Process dailyoutput from {preddt} to {doutput}")
    print(f"Process Predit from {preddt} to {doutput}")
    if doutput > preddt:
        for dt in rrule(DAILY, dtstart=preddt+ dt.timedelta(days=1), until=doutput):
            dt = dt.date()
            if dt.weekday() < 5:        # weekday only
                print(f"process main on {dt} {type(dt)}")
                main(rDaily, dt)
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
prev_close float
"""
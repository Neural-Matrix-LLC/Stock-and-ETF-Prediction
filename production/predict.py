from os import path
import mysql.connector
import pandas as pd
import logging
from datetime import date, timedelta, datetime

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

logging.basicConfig(filename='logging/predict.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# GLOBAL VARIABLES
HOST="143.244.188.157"
PORT="3306"
USER="patrick-finProj"
PASSWORD="Pat#21$rick"

def load_daily_outputs():
    logging.info(f'Load data from DailyOutputs table in MySQL.')
    try: 
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database="MarketPredict"
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

def main():
    logging.info(f'Start predict.py')
    try:
        threshold = 2 # Percent
        today = date.today()
        today = today.strftime("%Y-%m-%d")
        
        dailyoutput_df = load_daily_outputs()
        predictDate = datetime.strptime(dailyoutput_df.loc[0, 'Date'], "%Y-%m-%d")

        dailyoutput_df["close_change"] = dailyoutput_df.loc[:, 'LSTM'] - dailyoutput_df.loc[:, 'prev_Close']
        dailyoutput_df["price_movement"] = dailyoutput_df["close_change"].apply(get_price_movement)

        # Volatility aqgregation
        dailyoutput_df["volatility"] = dailyoutput_df[['garch', 'svr', 'mlp']].mean(axis=1)

        dailyoutput_df["above_threshold"] = dailyoutput_df["volatility"].apply(lambda x: get_above_threshold(x, threshold))
        dailyoutput_df["prediction"] = dailyoutput_df.apply(get_prediction, axis=1)
        dailyoutput_df.to_csv(f'predict_final/predict_{today}.csv')
        logging.info(f'Exported predict_{today}.csv')
    except:
        logging.error("Exception occurred at main()", exc_info=True)
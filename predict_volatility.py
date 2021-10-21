import os
import mysql.connector
from dotenv import load_dotenv
load_dotenv("mysql.env")
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from arch import arch_model
from arch.__future__ import reindexing
from sklearn.metrics import precision_score, f1_score

HOST=os.environ.get("HOST")
PORT=os.environ.get("PORT")
USER=os.environ.get("USER")
PASSWORD=os.environ.get("PASSWORD")

try: 
    conn = mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database="GlobalMarketData"
    )
    query = f"SELECT Date, Symbol, Close from histdailyprice3;"
    histdailyprice3 = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    #conn.close()
    print(str(e))
    
df = histdailyprice3.copy()
df.set_index("Date", drop=True, inplace=True)
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

# List of stocks and etfs
symbol_list = ['MSFT', 'INTC', 'EBAY', 'XOM', 'AAPL', 'UAL', 'JPM', 'DAL', 'DKNG',
       'USB', 'WFC', 'BMY', 'CSCO', 'PFE', 'T', 'BA', 'C', 'AMD', 'BAC',
       'WORK', 'VIAC', 'FB', 'NFLX', 'SNAP', 'TWTR', 'UBER', 'KO', 'PINS',
       'TSLA', 'AVTR', 'MU', 'NVDA', 'ORCL', 'CMCSA', 'CTLT', 'PTON',
       'KDP', 'MS', 'WDC', 'BSX', 'NKLA', 'FE', 'FCX', 'CCL', 'VZ',
       'PLUG', 'PLTR', 'MRNA', 'GM', 'COP', 'OXY', 'AMAT', 'CCIV', 'CZR',
       'MRK', 'HPQ', 'AAL', 'QS', 'RBLX', 'AMC', 'CLF', 'MTCH', 'BRO',
       'LCID', 'HYG', 'TLT', 'LQD', 'EFA', 'EWW', 'EWZ', 'GLD', 'SLV',
       'XME', 'XOP', 'XLE', 'DBA', 'IYR', 'VNQ', 'XHB', 'SPY', 'QQQ',
       'IWM']

# Connect to MySQL database
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
    conn.close()
    print(str(e))
    
df = histdailyprice3.copy()
df.set_index("Date", drop=True, inplace=True)

# Parameter tuning function
def gridsearch(data, p_rng, q_rng):
    top_score, top_results = float('inf'), None
    top_models = []
    for p in p_rng:
        for q in q_rng:
            try:
                model = arch_model(data, vol='GARCH', p=p, q=q, dist='normal')
                model_fit = model.fit(disp='off')
                resid = model_fit.resid
                st_resid = np.divide(resid, model_fit.conditional_volatility)
                results = evaluate_model(resid, st_resid)
                results['AIC'] = model_fit.aic
                results['params']['p'] = p
                results['params']['q'] = q
                if results['AIC'] < top_score: 
                    top_score = results['AIC']
                    top_results = results
                elif results['LM_pvalue'][1] is False:
                    top_models.append(results)
            except:
                continue
    top_models.append(top_results)
    return top_models

# Results dataframe
columns = ["Symbol", "Precision Macro", "Precision Micro", "F1 Macro", "F1 Micro"]
garch_performance = pd.DataFrame(columns=columns)
garch_performance

# Loop through stocks and etfs
p_rng = range(0,30)
q_rng = range(0,40)

for symbol in symbol_list.Symbol:
    try:
        symbol_df = df[df.Symbol == symbol]
        symbol_df['pct_change'] = 100 * symbol_df['Close'].pct_change()
        symbol_df.dropna(inplace=True)

        top_models = gridsearch(symbol_df['pct_change'], p_rng, q_rng)

        p = top_models[0]['params']['p']
        q = top_models[0]['params']['q']

        rolling_predictions = []
        test_size = 365
        for i in range(test_size):
            train = symbol_df['pct_change'][:-(test_size-i)]
            model = arch_model(symbol_df['pct_change'], p = p, q = q, mean = 'constant', vol = 'GARCH', dist = 'normal')
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

        rolling_predictions = pd.Series(rolling_predictions, index=symbol_df['pct_change'].index[-test_size:])

        y_pred = np.array(rolling_predictions >= 2)
        y_true = np.array(abs(symbol_df['pct_change'][-test_size:]) >= 2)

        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        garch_performance.loc[len(garch_performance.index)] = [stock_symbol, precision_macro, precision_micro, f1_macro, f1_micro]
    except:
        continue

# Export results
garch_performance.to_csv("reports/GARCH_performance.csv")

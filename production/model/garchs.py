import numpy as np 
import pandas as pd
from arch import arch_model
import logging
from multiprocessing import Pool
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch
import time
from sklearn.metrics import precision_score, f1_score


# number of processes for parallel processing
# None is default, os.cpu_count() is used.
num_p = 10
p_rng = range(1,30)
q_rng = range(1,40)

def evaluate_model(residuals, st_residuals, lags=50):
    results = {
        'LM_pvalue': None,
        'F_pvalue': None,
        'SW_pvalue': None,
        'BIC': None,
        'params': {'p': None, 'q': None}
    }
    arch_test = het_arch(residuals, nlags=lags)
    shap_test = shapiro(st_residuals)
    # We want falsey values for each of these hypothesis tests
    results['LM_pvalue'] = [arch_test[1], arch_test[1] < .05]
    results['F_pvalue'] = [arch_test[3], arch_test[3] < .05]
    results['SW_pvalue'] = [shap_test[1], shap_test[1] < .05]
    return results

def p_calc_model(data, p, q, o, vol):
    res = {}
    try:
        logging.info("calc_model({},{})".format(p, q, o, vol))
        model = arch_model(returns, mean='zero', vol=vol, p=p, o=o, q=q)
        logging.info("calc_model.model_fit")
        model_fit = model.fit(disp='off')
        resid = model_fit.resid
        logging.info("calc_model.divide")
        st_resid = np.divide(resid, model_fit.conditional_volatility)
        res = evaluate_model(resid, st_resid)
        res['BIC'] = model_fit.bic
        res['params']['p'] = p
        res['params']['q'] = q
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    return res


def multip_gridsearch(data, p_rng, q_rng, o, vol):
    n_sym = len(p_rng) * len(q_rng)
    logging.info("multi_gridsearch: {} trials.".format(n_sym))
    top_score, top_results = float('inf'), None
    top_models = []
    
    try:
        ll = []
        for p in p_rng:
            for q in q_rng:
                ll.append((data, p, q, o, vol))    
        logging.info("Starting {} threads".format(n_sym))
        logging.info(ll)
        with Pool(processes=num_p) as pool:
            all_results = pool.starmap(p_calc_model, ll)
        logging.info("All Grid threads Finish.")

        for i in range(len(all_results)):
            results = all_results[i]
            if results['BIC'] < top_score: 
                top_score = results['BIC']
                top_results = results
            elif results['LM_pvalue'][1] is False:
                top_models.append(results)
    except Exception as e:
        logging("multi_gridsearch:{}".format(e))
    top_models.append(top_results)
    return top_models

p_rng = range(1,30)
q_rng = range(1,40)

def predict():
    try:
        print("Start Process {}".format(symbol))
        start = time.time()
        symbol_df = df[df.Symbol == symbol]
        symbol_df['pct_change'] = 100 * symbol_df['Close'].pct_change()
        symbol_df.dropna(inplace=True)

        top_models = multip_gridsearch(symbol_df['pct_change'], p_rng, q_rng)
        print("{}'s top model={}".format(symbol, top_models))

        p = top_models[0]['params']['p']
        q = top_models[0]['params']['q']

        rolling_predictions = []
        test_size = round(len(symbol_df) * 0.2)
        print("{}'s test size: {}/{}".format(symbol, test_size, len(symbol_df)))
        ll=[]
        for i in range(test_size):
            ll.append((symbol_df['pct_change'], i, p, q))
        # print(ll)
        print("Starting {} apply_model threads".format(len(ll)))
        with Pool(processes=num_p) as pool:
            rolling_predictions = pool.starmap(apply_Model, ll)
        print("Apply_model threads Finish.")
        # print("* rolling_predictions : {}".format(rolling_predictions))

        rolling_predictions = pd.Series(rolling_predictions, index=symbol_df['pct_change'].index[-test_size:])

        y_pred = np.array(rolling_predictions >= 2)
        y_true = np.array(abs(symbol_df['pct_change'][-test_size:]) >= 2)

        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        end = time.time()
        total_time = end-start
        garch_performance.loc[len(garch_performance.index)] = [symbol, precision_macro, precision_micro, f1_macro, f1_micro, total_time]
        garch_performance.to_csv("../reports/GARCH_performance.csv")
    
    except Exception as e:
        print("calc_model:{}".format(e)) 
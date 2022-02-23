import numpy as np 
import pandas as pd
from arch import arch_model
import logging
from multiprocessing import Pool
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import precision_score, f1_score

def evaluate_model(residuals, st_residuals, lags=50):
    results = {
        'LM_pvalue': None,
        'F_pvalue': None,
        'SW_pvalue': None,
        'AIC': None,
        'BIC': None,
        'params': {
            'mean': None,
            'vol': None,
            'p': None,
            'o': None,
            'q': None,
            'dist': None
            }
    }
    arch_test = het_arch(residuals, nlags=lags)
    shap_test = shapiro(st_residuals)
    # We want falsey values for each of these hypothesis tests
    results['LM_pvalue'] = [arch_test[1], arch_test[1] < .05]
    results['F_pvalue'] = [arch_test[3], arch_test[3] < .05]
    results['SW_pvalue'] = [shap_test[1], shap_test[1] < .05]
    return results

def p_calc_model(data, mean, vol, p, q, o, dist):
    res = {}
    try:
        logging.info("calc_model({},{})".format(mean, vol, p, o, q, dist))
        model = arch_model(data, mean='zero', mean=mean, vol=vol, p=p, o=o, q=q, dist=dist)
        logging.info("calc_model.model_fit")
        model_fit = model.fit(disp='off')
        resid = model_fit.resid
        logging.info("calc_model.divide")
        st_resid = np.divide(resid, model_fit.conditional_volatility)
        res = evaluate_model(resid, st_resid)
        res['AIC'] = model_fit.aic
        res['BIC'] = model_fit.bic
        res['params']['mean'] = mean
        res['params']['vol'] = vol
        res['params']['p'] = p
        res['params']['o'] = o
        res['params']['q'] = q
        res['params']['dist'] = dist
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    return res


def multip_gridsearch(data, mean_list, vol_list, p_rng, q_rng, o_rng, dist_list, num_p=10):
    n_sym = len(p_rng) * len(q_rng)
    logging.info("multi_gridsearch: {} trials.".format(n_sym))
    top_score, top_results = float('inf'), None
    top_models = []
    
    try:
        ll = []
        for mean in mean_list:
            for vol in vol_list:
                for p in p_rng:
                    for o in o_rng:
                        for q in q_rng:
                            for dist in dist_list:
                                ll.append((data, mean, vol, p, o, q, dist))  
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

def tune(data):
    
    # Parameters
    num_p = 10
    mean_list = ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HARX']
    vol_list = ['GARCH', 'ARCH', 'EGARCH','FIARCH', 'HARCH'] 
    p_rng = range(0,20)
    o_rng = range(0,20)
    q_rng = range(0,20)
    
    dist_list = ['normal', 't', 'skewt', 'ged']

    try:
        logging.info("Start GARCH Process")
        top_models = multip_gridsearch(data, mean_list, vol_list, p_rng, q_rng, o_rng, dist_list, num_p)
        logging.info("Top model={}".format(top_models))

        # Best parameters
        mean = top_models[0]['params']['mean']
        vol = top_models[0]['params']['vol']
        p = top_models[0]['params']['p']
        o = top_models[0]['params']['o']
        q = top_models[0]['params']['q']
        dist = top_models[0]['params']['dist']

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

        garch_performance.loc[len(garch_performance.index)] = [symbol, precision_macro, precision_micro, f1_macro, f1_micro, total_time]
        garch_performance.to_csv("../reports/GARCH_performance.csv")
    
    except Exception as e:
        print("calc_model:{}".format(e)) 


# horizon=1
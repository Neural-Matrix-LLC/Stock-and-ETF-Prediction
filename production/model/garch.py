import numpy as np 
import pandas as pd
from arch import arch_model
import logging
from multiprocessing import Pool
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import precision_score, f1_score


# Generate returns for Volatiltiy models
def data(close, split=0.01):
    try:
        logging.info(f'Generate returns')
        returns = 100 * close.pct_change().dropna()
        return returns
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def evaluate_model(residuals, st_residuals, lags=50):
    try:
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
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def p_calc_model(data, mean, vol, p, q, o, dist):
    res = {}
    try:
        logging.info(f"calc_model({mean},{vol}, {p}, {o}, {q}, {dist})")
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
        return res
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def multip_gridsearch(data, mean_list, vol_list, p_rng, q_rng, o_rng, dist_list, num_p=10):
    n_sym = len(p_rng) * len(q_rng)
    logging.info(f"multi_gridsearch: {n_sym} trials.")
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
        logging.info(f"Starting {n_sym} threads")
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
        logging.error("Exception occurred", exc_info=True)
    top_models.append(top_results)
    return top_models

def tune(symbol, data):
    # Parameters
    num_p = 10
    mean_list = ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HARX']
    vol_list = ['GARCH', 'ARCH', 'EGARCH', 'HARCH'] 
    p_rng = range(1,20)
    o_rng = range(0,20)
    q_rng = range(0,20)
    dist_list = ['normal', 't', 'skewt', 'ged']
    try:
        logging.info("Start GARCH Process")
        top_models = multip_gridsearch(data, mean_list, vol_list, p_rng, q_rng, o_rng, dist_list, num_p)
        logging.info(f"Top GARCH model={top_models}")
        # Best parameters
        mean = top_models[0]['params']['mean']
        vol = top_models[0]['params']['vol']
        p = top_models[0]['params']['p']
        o = top_models[0]['params']['o']
        q = top_models[0]['params']['q']
        dist = top_models[0]['params']['dist']
        top_params = {"mean": mean, "vol": vol, "p": p, "o": o, "q": q, "dist": dist}
        logging.info(f'Best GARCH params={top_params}')

        with open(f"params/garch/{symbol}", "w") as outfile:
            json.dump(top_params, outfile)
        logging.info(f'Export best SVR parameters to JSON')

        return top_params
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

def predict(data, params):
    try:
        logging.info(f"Build GARCH({params}))")
        model = arch_model(data, params)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        return pred.variance.values[-1,:][0] / 100  # Return the variance
    except Exception as e:
        print("calc_model:{}".format(e)) 

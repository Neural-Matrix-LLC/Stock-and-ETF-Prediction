import numpy as np 
import pandas as pd
from arch import arch_model
import logging
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch

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

def p_calc_model(data, p, q, vol):
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


def multip_gridsearch(data, p_rng, q_rng):
    n_sym = len(p_rng) * len(q_rng)
    logging.info("multi_gridsearch: {} trials.".format(n_sym))
    top_score, top_results = float('inf'), None
    top_models = []
    
    try:
        ll = []
        for p in p_rng:
            for q in q_rng:
                ll.append((data, p, q))    
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
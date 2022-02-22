import numpy as np 
import pandas as pd
from arch import arch_model
import logging
from threading import Thread

# GARCH
def garch(returns, P=1, Q=1, tune=True):
    try:
        if tune:
            logging.info(f'Tune GARCH.')
            bic_garch = []
            for p in range(1, 5):
                for q in range(1, 5):
                    garch = arch_model(returns, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
                    bic_garch.append(garch.bic)
                    if garch.bic == np.min(bic_garch):
                        best_param = p, q
            garch = arch_model(returns, mean='zero', vol='GARCH', p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
            logging.info(f'Compile GARCH({best_param[0]}, {best_param[0]}).')
            return garch

        garch = arch_model(returns, mean='zero', vol='GARCH', p=P, o=0, q=Q).fit(disp='off')
        logging.info(f'Compile GARCH({P}, {Q}).')
        return garch
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

        
def p_calc_model(data, p, q):
    res = {}
    try:
#         print("calc_model({},{})".format(p, q))
        model = arch_model(data, vol='GARCH', p=p, q=q, dist='normal')
#         print("calc_model.model_fit")
        model_fit = model.fit(disp='off')
        resid = model_fit.resid
#         print("calc_model.divide")
        st_resid = np.divide(resid, model_fit.conditional_volatility)
        res = evaluate_model(resid, st_resid)
        res['AIC'] = model_fit.aic
        res['params']['p'] = p
        res['params']['q'] = q
    except Exception as e:
        print("calc_model:{}".format(e)) 
    return res

from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def multip_gridsearch(data, p_rng, q_rng):
    n_sym = len(p_rng)*len(q_rng)
    print("multi_gridsearch: {} trials.".format(n_sym))
    top_score, top_results = float('inf'), None
    top_models = []
    
    try:
        ll=[]
        for p in p_rng:
            for q in q_rng:
                ll.append((data, p, q))    
        print("Starting {} threads".format(n_sym))
#         print(ll)
        with Pool(processes=num_p) as pool:
            all_results = pool.starmap(p_calc_model, ll)
            
        print("All Grid threads Finish.")

        for i in range(len(all_results)):
            results = all_results[i]
            if results['AIC'] < top_score: 
                top_score = results['AIC']
                top_results = results
            elif results['LM_pvalue'][1] is False:
                top_models.append(results)
    except Exception as e:
        print("multi_gridsearch:{}".format(e))
    top_models.append(top_results)
    return top_models
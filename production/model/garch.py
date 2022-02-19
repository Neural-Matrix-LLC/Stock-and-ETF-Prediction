import numpy as np 
import pandas as pd
from arch import arch_model
import logging

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


def calc_model(data, p, q, all_results, i):
    try:
#         print("calc_model({},{},{},{})".format(p, q, all_results, i))
        model = arch_model(data, vol='GARCH', p=p, q=q, dist='normal')
#         print("calc_model.model_fit")
        model_fit = model.fit(disp='off')
        resid = model_fit.resid
#         print("calc_model.divide")
        st_resid = np.divide(resid, model_fit.conditional_volatility)
        results = evaluate_model(resid, st_resid)
        results['AIC'] = model_fit.aic
        results['params']['p'] = p
        results['params']['q'] = q
        all_results[i] = results
    except Exception as e:
        print("calc_model:{}".format(e)) 
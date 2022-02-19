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

        


def multi_gridsearch(data, p_rng, q_rng):
    n_sym = len(p_rng)*len(q_rng)
    logging.info("multi_gridsearch: {} trials.".format(n_sym))
    top_score, top_results = float('inf'), None
    top_models = []
    threads = [None] * n_sym
    all_results = [None] * n_sym
    try:
        i = 0
        for p in p_rng:
            for q in q_rng:
                print("Start {} thread.".format(i))
                threads[i] = Thread(target=calc_model, args=(data, p, q, all_results, i))
                threads[i].start()
                i += 1
        for i in range(len(threads)):
            threads[i].join()
            logging.info("Join {} thread.".format(i))
        logging.info("All Grid threads Finish.")
        for i in range(len(all_results)):
            results = all_results[i]
            if results['AIC'] < top_score: 
                top_score = results['AIC']
                top_results = results
            elif results['LM_pvalue'][1] is False:
                top_models.append(results)
    except Exception as e:
        logging.info("multi_gridsearch:{}".format(e))
    top_models.append(top_results)
    return top_models
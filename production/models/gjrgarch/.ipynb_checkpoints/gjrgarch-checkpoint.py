import numpy as np 
import pandas as pd
from arch import arch_model
import logging

# GJR GARCH
def gjrgarch(returns, P=1, Q=1, tune=True):
    if tune:
        logging.info(f'Tune GJR GARCH.')
        bic_gjr_garch = []
        for p in range(1, 5):
            for q in range(1, 5):
                gjrgarch = arch_model(returns, mean='zero', p=p, o=1, q=q).fit(disp='off')
                bic_gjr_garch.append(gjrgarch.bic)
                if gjrgarch.bic == np.min(bic_gjr_garch):
                     best_param = p, q
        gjrgarch = arch_model(returns, mean='zero', p=best_param[0], o=1, q=best_param[1]).fit(disp='off')
        logging.info(f'GJR GARCH({best_param[0]}, {best_param[0]}).')
        return gjrgarch
   
    gjrgarch = arch_model(returns, mean='zero', vol='GARCH', p=P, o=0, q=Q).fit(disp='off')
    logging.info(f'GJR GARCH({P}, {Q}).')
    return gjrgarch

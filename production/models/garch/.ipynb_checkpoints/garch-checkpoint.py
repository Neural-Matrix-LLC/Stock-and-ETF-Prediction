import numpy as np 
import pandas as pd
from arch import arch_model
import logging


# GARCH
def garch(P=1, Q=1, tune=True):
    if tune:
        logging.info(f'{name} Tune GARCH.')
        bic_garch = []
        for p in range(1, 5):
            for q in range(1, 5):
                garch = arch_model(returns, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
                bic_garch.append(garch.bic)
                if garch.bic == np.min(bic_garch):
                    best_param = p, q
        garch = arch_model(returns, mean='zero', vol='GARCH', p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
        logging.info(f'{name} GARCH({best_param[0]}, {best_param[0]}).')
        return garch
    
    garch = arch_model(returns, mean='zero', vol='GARCH', p=P, o=0, q=Q).fit(disp='off')
    logging.info(f'{name} GARCH({P}, {Q}).')
    return garch
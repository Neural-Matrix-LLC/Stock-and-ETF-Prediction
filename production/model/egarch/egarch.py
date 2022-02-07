import numpy as np 
import pandas as pd
from arch import arch_model
import logging

def egarch(returns, P=1, Q=1, tune=True):
    if tune:
        logging.info(f'Tune EGARCH.')
        bic_egarch = []
        for p in range(1, 5):
            for q in range(1, 5):
                egarch = arch_model(returns, mean='zero', vol='EGARCH', p=p, q=q).fit(disp='off')
                bic_egarch.append(egarch.bic)
                if egarch.bic == np.min(bic_egarch):
                    best_param = p, q
        egarch = arch_model(returns, mean='zero', vol='EGARCH', p=best_param[0], q=best_param[1]).fit(disp='off')
        logging.info(f'Compile EGARCH({best_param[0]}, {best_param[0]}).')
        return egarch
    
    egarch = arch_model(returns, mean='zero', vol='EGARCH', p=P, q=Q).fit(disp='off')
    logging.info(f'Compile EGARCH({P}, {Q}).')
    return egarch

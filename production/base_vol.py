import numpy as np 
import pandas as pd
from arch import arch_model

returns = 100 * df['Close'].pct_change().dropna()
n = int(len(returns)*0.4)
split_date = returns[-n:].index

# GARCH
def best_GARCH():
    bic_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            garch = arch_model(returns, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
            bic_garch.append(garch.bic)
            if garch.bic == np.min(bic_garch):
                best_param = p, q
    garch = arch_model(returns, mean='zero', vol='GARCH',
                    p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
    return garch
garch = best_GARCH()
forecast_garch = garch.forecast(start=split_date[0])
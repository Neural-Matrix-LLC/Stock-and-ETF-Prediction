import numpy as np 
import pandas as pd
from arch import arch_model

returns = 100 * df['Close'].pct_change().dropna()
n = int(len(returns)*0.4)
split_date = returns[-n:].index

# GARCH
garch = arch_model(returns, mean='zero', vol='GARCH', p=1, o=0, q=1).fit(disp='off')
bic_garch = []


def best_garch():
    for p in range(1, 5):
        for q in range(1, 5):
            garch = arch_model(returns, mean='zero', vol='GARCH', p=p, o=0, q=q).fit(disp='off')
            bic_garch.append(garch.bic)
            if garch.bic == np.min(bic_garch):
                best_param = p, q
    garch = arch_model(returns, mean='zero', vol='GARCH',
                    p = best_param[0], o = 0, q = best_param[1]).fit(disp='off')
    return garch
forecast = best_garch().forecast(start=split_date[0])
forecast_garch = forecast
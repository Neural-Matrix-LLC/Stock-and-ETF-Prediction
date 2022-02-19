import numpy as np 
import pandas as pd
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import logging

# Linear SVR
def svr_lin():
    logging.info(f'{name} Tune Linear SVR.')
    svr_lin = SVR(kernel='linear')
    para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
    clf = RandomizedSearchCV(svr_lin, para_grid, n_jobs=8)
    return clf
svr_lin = tune_svr_lin 
clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_lin = clf.predict(X.iloc[-n:])
predict_svr_lin = pd.DataFrame(predict_svr_lin)
predict_svr_lin.index = returns.iloc[-n:].index
from os import path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import pickle
from datetime import date
import logging

# Linear SVR
def tuned_svr_lin(stock_symbol):
    logging.info(f'Tune Linear SVR.')
    try:
        if path.isfile(dpath):
            logging.info("Load data from {}".format(dpath))
            with open(f'svr_lin/{stock_symbol}.pickle', 'rb') as handle:
                clf = pickle.load(handle)
            return clf
        else:
            svr_lin = SVR(kernel='linear')

            para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
            clf = RandomizedSearchCV(svr_lin, para_grid, n_jobs=8)
            with open(f'svr_lin/{stock_symbol}.pickle', 'wb') as handle:
                pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return clf
    except Exception as e:
    print("calc_model:{}".format(e))
    
def predict(stock_symbol):
    logging.info(f'{name} Tune Linear SVR.')
    svr_lin = tune_svr_lin(stock_symbol)
    clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
    predict_svr_lin = clf.predict(X.iloc[-n:])
    predict_svr_lin = pd.DataFrame(predict_svr_lin)
    predict_svr_lin.index = returns.iloc[-n:].index
    return predict_svr_lin

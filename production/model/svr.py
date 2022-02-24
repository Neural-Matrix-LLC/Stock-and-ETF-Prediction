from os import path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import logging

#if path.isfile(dpath):
#    logging.info("Load data from {}".format(dpath))
#    with open(f'svr_lin/{stock_symbol}.pickle', 'rb') as handle:
#        clf = pickle.load(handle)
#    return clf

# with open(f'svr_lin/{stock_symbol}.pickle', 'wb') as handle:
#    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    return clf

# Linear SVR
def tune(data, kernel):
    logging.info(f'Tune SVR {kernel}')
    try:
        svr = SVR(kernel=kernel)
        para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
        clf = RandomizedSearchCV(svr, para_grid, n_jobs=8)
        return clf
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    
def predict(X_train, y_train, X_test, kernel, gamma, C, epsilon):
    clf = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon)
    # X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction

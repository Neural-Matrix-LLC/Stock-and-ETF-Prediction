from os import path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import logging

# Linear SVR
def tune(X, y):
    logging.info(f'Tune SVR hyperparameters')
    try:
        para_grid = {'kernel': ('linear', 'rbf','poly'), 'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
        svr = SVR()
        clf = RandomizedSearchCV(svr, para_grid, n_jobs=8)
        clf.fit(X, y)
        clf.best_params
        return clf
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
    
def predict(X_train, y_train, X_test, kernel, gamma, C, epsilon):
    clf = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon)
    # X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction

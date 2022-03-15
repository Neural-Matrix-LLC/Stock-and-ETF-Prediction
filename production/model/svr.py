from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import logging

# Tune SVR
def tune(X, y):
    logging.info(f'Tune SVR hyperparameters')
    try:
        para_grid = {'kernel': ('linear', 'rbf'), 'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
        svr = SVR()
        clf = RandomizedSearchCV(svr, para_grid, n_jobs=-1)
        clf.fit(X, y)
        top_params = clf.best_params_
        logging.info(f'Best SVR parameters {top_params}')
        return top_params
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Predict with SVR using best parameters
def predict(X, y, params):
    try:
        clf = SVR(kernel=params["kernel"], gamma=params["gamma"], C=params["C"], epsilon=params["epsilon"])
        clf.fit(X, y)
        prediction = clf.predict(X.iloc[-1:])
        return prediction[0]
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

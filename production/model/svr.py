from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import logging

# Tune SVR
def tune(X, y):
    logging.info(f'Tune SVR hyperparameters')
    try:
        para_grid = {'kernel': ('linear', 'rbf','poly'), 'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
        svr = SVR()
        clf = RandomizedSearchCV(svr, para_grid, n_jobs=-1)
        clf.fit(X, y)
        top_params = clf.best_params
        logging.info(f'Best SVR parameters {top_params}')
        return top_params
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Predict with SVR using best parameters
def predict(X_train, y_train, X_test, params):
    clf = SVR(params)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction

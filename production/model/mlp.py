from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import logging

# Tune MLP
def tune(X, y):
    logging.info(f'Tune MLP hyperparameters')
    try:
        param_grid = {'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05, 0.0005, 0.00005],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}
        mlp = MLPRegressor()
        clf = GridSearchCV(mlp, para_grid, n_jobs=-1)
        clf.fit(X, y)
        top_params = clf.best_params
        logging.info(f'Best MLP parameters {top_params}')
        return top_params
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Predict with MLP using best parameters
def predict(X_train, y_train, X_test, params):
    clf = MLPRegressor(params)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction

import numpy as np 
import pandas as pd
from arch import arch_model
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error as mse
import logging
# Python Files
import data

df = data.get_df()

# Processing
returns = 100 * df['Close'].pct_change().dropna()
n = int(len(returns)*0.4)
split_date = returns[-n:].index

# GARCH
def tune_garch():
    logging.info(f'{name} Tune GARCH.')
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
garch = tune_garch()
forecast_garch = garch.forecast(start=split_date[0])

# GJR GARCH
def tune_gjr_garch():
    logging.info(f'{name} Tune GJR GARCH.')
    bic_gjr_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            gjrgarch = arch_model(returns, mean='zero', p=p, o=1, q=q).fit(disp='off')
            bic_gjr_garch.append(gjrgarch.bic)
            if gjrgarch.bic == np.min(bic_gjr_garch):
                 best_param = p, q
    gjrgarch = arch_model(returns, mean='zero', p=best_param[0], o=1, q=best_param[1]).fit(disp='off')
    return gjrgarch
gjrgarch = tune_gjr_garch()
forecast_gjrgarch = gjrgarch.forecast(start=split_date[0])

def tune_egarch():
    logging.info(f'{name} Tune EGARCH.')
    bic_egarch = []
    for p in range(1, 5):
        for q in range(1, 5):
            egarch = arch_model(returns, mean='zero', vol='EGARCH', p=p, q=q).fit(disp='off')
            bic_egarch.append(egarch.bic)
            if egarch.bic == np.min(bic_egarch):
                best_param = p, q
    egarch = arch_model(returns, mean='zero', vol='EGARCH', p=best_param[0], q=best_param[1]).fit(disp='off')
    return egarch
egarch = tune_egarch()
forecast_egarch = egarch.forecast(start=split_date[0])

# Compute realized volatility
realized_vol = returns.rolling(5).std()
realized_vol = pd.DataFrame(realized_vol)
realized_vol.reset_index(drop=True, inplace=True)

returns_svm = returns ** 2
returns_svm = returns_svm.reset_index()
del returns_svm['index']

X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
X = X[4:].copy()
X = X.reset_index()
X.drop('index', axis=1, inplace=True)

realized_vol = realized_vol.dropna().reset_index()
realized_vol.drop('index', axis=1, inplace=True)

# Linear SVR
svr_lin = SVR(kernel='linear')
para_grid = {'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
clf = RandomizedSearchCV(svr_lin, para_grid, n_jobs=8)
clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_lin = clf.predict(X.iloc[-n:])
predict_svr_lin = pd.DataFrame(predict_svr_lin)
predict_svr_lin.index = returns.iloc[-n:].index

# RBF SVR
svr_rbf = SVR(kernel='rbf')
para_grid ={'gamma': sp_rand(), 'C': sp_rand(), 'epsilon': sp_rand()}
clf = RandomizedSearchCV(svr_rbf, para_grid, n_jobs=8)
clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_rbf = clf.predict(X.iloc[-n:])
predict_svr_rbf = pd.DataFrame(predict_svr_rbf)
predict_svr_rbf.index = returns.iloc[-n:].index

## NN
NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
para_grid_NN = {'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
                'max_iter': [500, 1000],
                'alpha': [0.00005, 0.0005 ]}
clf = RandomizedSearchCV(NN_vol, para_grid_NN, n_jobs=8)
clf.fit(X.iloc[:-n].values, realized_vol.iloc[1:-(n-1)].values.reshape(-1, ))
NN_predictions = clf.predict(X.iloc[-n:])
NN_predictions = pd.DataFrame(NN_predictions)
NN_predictions.index = returns.iloc[-n:].index

# Deep Learning
model = keras.Sequential([layers.Dense(256, activation="relu"),
                          layers.Dense(128, activation="relu"),
                          layers.Dense(1, activation="linear"),])

model.compile(loss='mse', optimizer='rmsprop')
epochs_trial = np.arange(100, 400, 4)
batch_trial = np.arange(100, 400, 4)
DL_pred = []
DL_RMSE = []
for i, j, k in zip(range(4), epochs_trial, batch_trial):
    model.fit(X.iloc[:-n].values,
              realized_vol.iloc[1:-(n-1)].values.reshape(-1,),
              batch_size=k, epochs=j, verbose=False)
    DL_predict = model.predict(np.asarray(X.iloc[-n:]))
    DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                         DL_predict.flatten() / 100)))
    DL_pred.append(DL_predict)

DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
DL_predict.index = returns.iloc[-n:].index

# Ensemble
forecast_garch
forecast_gjrgarch
forecast_egarch
predict_svr_lin
predict_svr_rbf
NN_predictions
DL_predict
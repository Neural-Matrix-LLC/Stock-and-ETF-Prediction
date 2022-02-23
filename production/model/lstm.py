import numpy as np 
import pandas as pd
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# normalize data
def normalize(close):
    try:
        scaler = MinMaxScaler(feature_range = (0,1))
        scaled_data = scaler.fit_transform(close.values.reshape(-1, 1))
        return scaler, scaled_data
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)
        
# test-train split
def create_dataset(dataset, time_step):
    try:
        x_data, y_data = [], []
        for i in range(len(dataset)-time_step-1):
            x_data.append(dataset[i:(i+time_step), 0])
            y_data.append(dataset[i + time_step, 0])
        return np.array(x_data), np.array(y_data)
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)

# Preparing train and test data
def test_train_split(scaled_data, train_size=0.8, time_step=100):
    try:
        training_size = int(len(scaled_data) * train_size)
        test_size = len(scaled_data) - training_size
        train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)

# Compile Model
def build_model(hp):
    try:
        model = Sequential()
        model.add(layers.LSTM(units = hp.Choice('layer1_units', [10,20,30,40,50,60,70,80,90,100]), return_sequences=True,input_shape=(100,1)))

        for i in range(hp.Int('num_layers', 2, 15)):                        
            model.add(layers.LSTM(units =  hp.Int('units' + str(i), min_value=10, max_value=150, step=10), return_sequences=True))

        model.add(LSTM(units = hp.Choice('last_lstm_units', [50, 100, 150])))
        model.add(Dropout(rate = hp.Choice('rate', [0.3, 0.4, 0.5, 0.6, 0.7])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam' )
        return model
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)

# Tune Model Parameters
def keras_tuner():
    try:
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials= 5,
            executions_per_trial=3,
            directory='lstm_tuner', project_name = f'{stock_symbol}')
        tuner.search(X_train, y_train,
                     epochs= 5,
                     validation_data=(X_test, ytest))
        model = tuner.get_best_models(num_models=1)[0]
        return model
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)

# LSTM prediction    
def predict(stock_symbol, close):
    try:
        scaler, scaled_data = normalize(close)
        X_train, y_train, X_test, y_test = test_train_split(scaled_data, train_size=0.8, time_step=100)
        model = keras_tuner()
        model.save(f"saved_lstm/{stock_symbol}_lstm.h5")
        scaled_predict = model.predict(X_train)
        predict = scaler.inverse_transform(scaled_predict)
        return predict
    except Exception as e:
        logging.error("Exception occurred at load_df()", exc_info=True)
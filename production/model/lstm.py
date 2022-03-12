import numpy as np
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import logging

# normalize data
def normalize(close):
    logging.info(f'LSTM normalize')
    try:
        scaler = MinMaxScaler(feature_range = (0,1))
        scaled_data = scaler.fit_transform(close.values.reshape(-1, 1))
        return scaler, scaled_data
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        
# test-train split
def create_dataset(dataset, time_step):
    logging.info(f'LSTM create_dataset')
    try:
        x_data, y_data = [], []
        for i in range(len(dataset)-time_step-1):
            x_data.append(dataset[i:(i+time_step), 0])
            y_data.append(dataset[i + time_step, 0])
        return np.array(x_data), np.array(y_data)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Preparing train and test data
def test_train_split(scaled_data, train_size, time_step):
    logging.info('LSTM test_train_split({},{},{})'.format(scaled_data.shape,train_size,time_step))
    try:
        training_size = int(len(scaled_data) * train_size)
        test_size = len(scaled_data) - training_size
        test_size = test_size if test_size > 105 else 105
        training_size = len(scaled_data) - test_size
        logging.debug("-> training_size:{}\ttest_size:{}".format(training_size, test_size))
        train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        logging.debug("x_train:{}, y_train:{}, X_test:{}, y_test:{}".format(X_train.shape,y_train.shape,X_test.shape,y_test.shape))
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# Compile Model
def build_model(hp):
    logging.info(f'LSTM build_mode')
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
        logging.error("Exception occurred", exc_info=True)

# Tune Model Parameters
def k_tuner(symbol, X_train, y_train, X_test, ytest):
    logging.info(f'LSTM k_tuner')
    try:
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials= 5,
            executions_per_trial=3,
            directory='lstm_tuner', project_name = f'{symbol}')
        tuner.search(X_train, y_train,
                     epochs= 5,
                     validation_data=(X_test, ytest))
        model = tuner.get_best_models(num_models=1)[0]
        return model
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# LSTM Tune
def tune(symbol, close):
    logging.info(f'Tune LSTM')
    try:
        scaler, scaled_data = normalize(close)
        X_train, y_train, X_test, y_test = test_train_split(scaled_data, train_size=0.9, time_step=100)
        model = k_tuner(symbol, X_train, y_train, X_test, y_test)
        return model
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# LSTM prediction    
def predict(model, close):
    logging.info(f'LSTM Predict')
    try:
        scaler, scaled_data = normalize(close)
        X_train, y_train, X_test, y_test = test_train_split(scaled_data, train_size=0.9, time_step=100)
        scaled_predict = model.predict(X_test)
        predict = scaler.inverse_transform(scaled_predict)
        return predict[-1]
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

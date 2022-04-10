import numpy as np
import keras_tuner
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
import logging

DEFAULT_STEP = 100
MIN_D_SIZE = DEFAULT_STEP * 2.1

# normalize data
def normalize(close):
    logging.info(f'LSTM normalize')
    try:
        scaler = MinMaxScaler(feature_range = (0,1))
        scaled_data = scaler.fit_transform(close.values.reshape(-1, 1))
        return scaler, scaled_data
    except Exception as e:
        logging.error("normalize: Exception occurred", exc_info=True)
        
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
        logging.error("create_dataset: Exception occurred", exc_info=True)

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
        if train_size > 0:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error("test_train_split: Exception occurred", exc_info=True)

# Custom Loss function
def custom_loss(y_true, y_pred):
    """Customized loss function that takes into account directional loss.
    
    ARGS:
    y_true: tensor of true price
    y_pred: tensor of predicted price
    
    RETURN:
    custom loss output
    """
    try:
        #the "next day's price" of tensor
        y_true_next = y_true[1:]
        y_pred_next = y_pred[1:]

        #the "today's price" of tensor
        y_true_tdy = y_true[:-1]
        y_pred_tdy = y_pred[:-1]

        #substract to get up/down movement of the two tensors
        y_true_diff = tf.subtract(y_true_next, y_true_tdy)
        y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)

        #create a standard tensor with zero value for comparison
        standard = tf.zeros_like(y_pred_diff)

        #compare with the standard; if true, UP; else DOWN
        y_true_move = tf.greater_equal(y_true_diff, standard)
        y_pred_move = tf.greater_equal(y_pred_diff, standard)
        y_true_move = tf.reshape(y_true_move, [-1])
        y_pred_move = tf.reshape(y_pred_move, [-1])


        #find indices where the directions are not the same
        condition = tf.not_equal(y_true_move, y_pred_move)
        indices = tf.where(condition)

        #move one position later
        ones = tf.ones_like(indices)
        indices = tf.add(indices, ones)
        indices = K.cast(indices, dtype='int32')


        #create a tensor to store directional loss and put it into custom loss output
        direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32')
        updates = K.cast(tf.ones_like(indices), dtype='float32')
        alpha = 1000
        direction_loss = tf.scatter_nd_update(direction_loss, indices, alpha*updates)

        custom_loss = K.mean(tf.multiply(K.square(y_true - y_pred), direction_loss), axis=-1)

        return custom_loss
    except Exception as e:
        logging.error("Exception occurred at get_price_movement()", exc_info=True)

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
        model.compile(loss=custom_loss, optimizer='adam')
        return model
    except Exception as e:
        logging.error("build_model: Exception occurred", exc_info=True)

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
        logging.error("k_tuner: Exception occurred", exc_info=True)

# LSTM Tune
def tune(symbol, close):
    logging.info(f'Tune LSTM')
    try:
        dsize = len(close)
        if dsize < MIN_D_SIZE:
            raise ValueError(f'data size {dsize} is smaller than minium data size:{MIN_D_SIZE}')
        scaler, scaled_data = normalize(close)
        X_train, y_train, X_test, y_test = test_train_split(scaled_data, train_size=0.9, time_step=DEFAULT_STEP)
        model = k_tuner(symbol, X_train, y_train, X_test, y_test)
        return model
    except Exception as e:
        logging.error("tune: Exception occurred", exc_info=True)

# LSTM prediction    
def predict(model, close):
    logging.info(f'LSTM Predict')
    try:
        dsize = len(close)
        if dsize < MIN_D_SIZE:
            raise ValueError(f'data size {dsize} is smaller than minium data size:{MIN_D_SIZE}')
        scaler, scaled_data = normalize(close)
        X_train, y_train, X_test, y_test = test_train_split(scaled_data, train_size=0, time_step=DEFAULT_STEP)
        scaled_predict = model.predict(X_test)
        predict = scaler.inverse_transform(scaled_predict)
        logging.debug(" In Predict: {}".format(predict[-1]))
        return predict[-1][0]
    except Exception as e:
        logging.error("predict: Exception occurred", exc_info=True)

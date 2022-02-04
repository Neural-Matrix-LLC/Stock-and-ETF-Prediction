import numpy as np 
import pandas as pd
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


# Normalize Data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Train-Test Split
def create_dataset(dataset, time_step = 1):
    x_data, y_data = [], []
    
    for i in range(len(dataset) - time_step - 1):
        x_data.append(dataset[i:(i + time_step), 0])
        y_data.append(dataset[i + time_step, 0])
    return np.array(x_data), np.array(y_data)

# Preparing train and test data
training_size = int(len(scaled_data)*0.65)
test_size = len(scaled_data)-training_size
train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:1]

#Taking data for past 100 days for next prediction
time_step = 100

X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)

# Build LSTM model
def build_model(hp):
    model = Sequential()
    model.add(layers.LSTM(units = hp.Choice('layer1_units', [10,20,30,40,50,60,70,80,90,100]), 
                          return_sequences=True, 
                          input_shape=(X_train.shape[1],1)))
    for i in range(hp.Int('num_layers', 2, 15)):                        
        model.add(layers.LSTM(units =  hp.Int('units' + str(i), min_value=10, max_value=150, step=10), return_sequences=True))
    
    model.add(LSTM(units = hp.Choice('last_lstm_units', [50, 100, 150])))
    model.add(Dropout(rate = hp.Choice('rate', [0.3, 0.4, 0.5, 0.6, 0.7])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam' )
    return model

# Define Callbacks
tuner = keras_tuner.RandomSearch(
    build_model,
    objective = 'val_loss',
    max_trials = 5,
    executions_per_trial = 3,
    save_best_only = True,
    directory = 'tuner', project_name = f'{datetime.now()}')

tuner.search(X_train, y_train,
             epochs= 5,
             validation_data=(X_test, ytest))

# Loss: Training vs Validation
#model_history = model.fit(X_train,y_train, epochs=100, validation_data=(X_test,ytest), callbacks=callbacks)
#loss = model_history.history['loss']
#validation_loss = model_history.history['val_loss']

# Prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
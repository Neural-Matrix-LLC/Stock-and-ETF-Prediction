import numpy as np 
import pandas as pd
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import logging

# normalize data
def normalize(close):
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(close.values.reshape(-1, 1))
    return scaled_data

# test-train split
def create_dataset(dataset, time_step=1):
    x_data, y_data = [], []
    
    for i in range(len(dataset)-time_step-1):
        x_data.append(dataset[i:(i+time_step), 0])
        y_data.append(dataset[i + time_step, 0])
    return np.array(x_data), np.array(y_data)

# Preparing train and test data
def test_train_split(scaled_data, train_size=0.8, time_step=100):
    training_size = int(len(scaled_data) * train_size)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)

def lstm(close):
    normalize(close)
    
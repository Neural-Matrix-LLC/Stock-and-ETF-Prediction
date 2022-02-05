from sklearn.preprocessing import MinMaxScaler

# Normalize Data

def normalize(data):
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
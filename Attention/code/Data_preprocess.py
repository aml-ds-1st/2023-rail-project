
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from DARNN_model import *
from keras.layers import LSTM



# df = pd.read_csv('../../data/Rail_data.csv')
df = pd.read_csv('C:/Users/AML2/Desktop/TIL/data/Rail_data.csv')

# scaler = MinMaxScaler()
# scaled_col = ['air_temp','TSI','azimuth','altitude','solar_rad','High_solar_rad', 'casi', 'humidity', 'rain', 'wind_speed','wind_direction','rail_direction']
# df[scaled_col]= scaler.fit_transform(df[scaled_col])

# scaler1 = MinMaxScaler()
# df['rail_temp'] = scaler1.fit_transform(df['rail_temp'].values.reshape(-1,1))


X = df.iloc[:,:12].values
y = df.iloc[:,12].values


# X_train = X[:split]
# y_train = y[:split]
# X_test = X[split:]
# y_test = y[split:]


def sequence_data(X, y, sequence_size):
    enc_data = []
    dec_data = []
    target_list = []
    for idx in range(1, len(X) - sequence_size): #len(X)가 7000이고 seq_size가 5라면?
        enc_data.append(X[idx:idx + sequence_size])
        dec_data.append(y[idx:idx + sequence_size-1])                             
        target_list.append(y[idx + sequence_size])
        
    return torch.tensor(enc_data, dtype=torch.float32), torch.tensor(dec_data, dtype=torch.float32), torch.tensor(target_list, dtype=torch.float32).view(-1,1)

enc_data, dec_data, target_list = sequence_data(X, y, 5)

encoder_sequence = np.array(enc_data)
decoder_sequence = np.array(dec_data)
decoder_sequence = np.reshape(decoder_sequence, (-1, 4, 1))  # (?, 4, 1)로 shape을 바꾸는 이유는?
target = np.array(target_list)


np.save("data/encoder_data.npy", encoder_sequence)
np.save("data/decoder_data.npy", decoder_sequence)
np.save("data/target.npy", target)

    
# dec_data = dec_data.permute((-1, 4, 1))

# encoder_data = enc_data.to_csv("encoder.data.csv")
# decoder_data = dec_data.to_csv("decoder_data.csv")
# target = target.to_csv("target.csv")

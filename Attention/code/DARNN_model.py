import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class InputAttention(nn.Module): # Encoder hidden state를 이용하여 중요한 Feature 뽑아내기
    def __init__(self, T, m):
        super().__init__()
        self.m = m
        self.w1 = nn.Linear(T,T)
        self.w2 = nn.Linear(2*m,T)
        self.v = nn.Linear(T,1)
        
        
    def forward(self, h_s, c_s, x):
        query = torch.cat([h_s, c_s], dim=1) # shape: (batch, 2*m) 
        query = query.repeat(x.shape[2],1) # shape: (batch, n, 2*m)
        x_perm = x.permute((0, 2, 1))             # shape: (batch, n, T)
        score = torch.tanh(self.w1(x_perm) + self.w2(query)) # shape: (batch, n, T)
        score = self.v(score)                   # shape: (batch, n, 1)
        score = score.permute((0, 2, 1))    # shape: (batch, 1, n)
        attention_weights = nn.softmax(score) 
        
        return attention_weights
    '''
    n = encoder input data number
    m = encdoer lstm features
    p = decoder lstm features
    T = time series length >> sequence length
    
    h_s:shape >> (batch, m)
    c_s:shape >> (batch, m)
    x = time series encoder inputs:shape >> (batch, T, n)
    RepeatVector: 차원 수 더해주기
    Permute: shape에서 위치 바꾸기
    '''
    
class Encoderlstm(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers ):
        super(Encoderlstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                  
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
                
        _, [h_s, c_s] = self.lstm(x, (h0, c0))  # batch, m
        
        return h_s, c_s
        

class Encoder(nn.Module):
    def __init__(self, T, m):
        super().__init__()
        self.T = T
        self.input_att = InputAttention(T,m)
        self.lstm = Encoderlstm(m, 12, T, 2)
        self.alpha_t = None
        
    def forward(self, data, h0, c0, n=12):
        '''
        data: encoder data(batch, T, n)
        n: data feature num
        '''
        
        alpha_seq = torch.tensor(self.T, dtype = torch.float32) 
        for t in range(self.T):
            Lambda = lambda x: data[:,t,:]
            x = Lambda(data) # batch, 1~T, n
            x = x.unsqueeze(1) # (batch, 1, n)
            
            h_s, c_s = self.lstm(x)
            
            self.alpha_t = self.input_att(h_s, c_s, data) # (batch, 1, n)
            
            alpha_seq = alpha_seq.write(t, self.alpha_t)
            alpha_seq = alpha_seq.stack()
            
        alpha_seq = alpha_seq.reshape((-1, self.T, n))
        
        output = torch.multiply(data, alpha_seq)
        
        return output
        
class TemporalAttention(nn.Module):
    def __init__(self, p ,m):
        super().__init__()
        self.p = p
        self.m = m
        self.w1 = nn.Linear(p,m)
        self.w2 = nn.Linear(p,m)
        self.v = nn.Linear(m,1)
        
    def forward(self, h_s, c_s, enc_h):
        query = torch.cat([h_s, c_s], dim=1)
        query = query.repeat(enc_h.shape[1],1)
        score = torch.tanh(self.w1(enc_h) + self.w2(query))
        score = self.v(score)
        attention_weights = nn.softmax(score, axis=1)
        
        return attention_weights
            
class Decoderlstm(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers):
        
        super(Decoderlstm, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)     
        _, [h_s, c_s] = self.lstm(x, (h0, c0))  # batch, m
        
        return h_s, c_s
        

        
class Decoder(nn.Module):
    def __init__(self, T, p, m):
        super().__init__()
        self.T = T
        self.temp_att = TemporalAttention(p,m)
        self.linear = nn.Linear(m+1,1)
        self.lstm = Decoderlstm(p, 12, T, 2)
        self.enc_lstm_dim = m
        self.dec_lstm_dim = p
        self.context_v = None
        self.dec_h_s = None
        self.beta_t = None
        
        
    def forward(self, data, enc_h, h0=None, c0=None):
        h_s = None
        self.context_v = torch.zeros((enc_h.shape[0], 1, self.enc_lstm_dim))
        self.dec_h_s = torch.zeros((enc_h.shape[0], self.dec_lstm_dim))
        
        for t in range(self.T-1):
            Lambda = lambda x: data[:,t,:]
            x = Lambda(data)  # batch, 1~T-1, 1
            x = x.unsqueeze(1)
            x = torch.cat([x, self.context_v], dim=1)
            x = self.linear(x)
            
            h_s, c_s = self.lstm(x)
            
            self.beta_t = self.temp_att(h_s, c_s, enc_h)
            
            self.context_v = torch.matmul(self.beta_t, enc_h, transpose_a=True)
            
            return torch.cat([h_s[:, np.newaxis, :], self.context_v], dim=1)
        
class DARNN(nn.Module):
    def __init__(self, T, m, p):
        super().__init__()       
        self.m = m
        self.encoder = Encoder(T=T, m=m)
        self.decoder = Decoder(T=T, p=p, m=m)
        self.lstm = nn.LSTM(m, 12)
        self.linear1 = nn.Linear(m+p,p)
        self.linear2 = nn.Linear(p,1)
       
        
    def forward(self, inputs):
        '''
        input: [enc, dec]
        enc_data: batch, T, n
        dec_data: batch, T-1 , 1
        '''
        enc_data, dec_data = inputs
        batch = enc_data.shape[0]
        h0 = torch.zeros((batch, self.m))
        c0 = torch.zeros((batch, self.m))
        enc_output = self.encoder(enc_data, n = 12, h0=h0, c0=c0) # batch, T, n
        enc_h = self.lstm(enc_output)   # batch, T, m
        dec_output = self.decoder(dec_data, enc_h, h0=h0, c0=c0) # batch, 1, m+p
        output = self.linear2(self.linear1(dec_output))
        output = torch.squeeze(output)
        
        return output
        
        
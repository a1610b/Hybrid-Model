# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:18:26 2020

@author: Tony She

E-mail: tony_she@yahoo.com
"""

import numpy as np
import torch
from torch import nn
import functions.get_data as get_data
import matplotlib.pyplot as plt

# (Hyper parameters)
learning_rate = 1e-4
num_epochs = 1000
CHARACTER_FOR_TRAIN = 72

class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x

model = LSTM_Regression(CHARACTER_FOR_TRAIN, 8, output_size=1, num_layers=2).cuda()
loss_function = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

industry_dict = get_data.load_obj('industry_dict')
data = get_data.get_from_sql(name='LSTM_data')

for industry in industry_dict:
    for stock in industry_dict[industry]['con_code']:
        data_train = torch.from_numpy(data[stock+'_data_train'].values.astype(np.float32).reshape(-1, 1, CHARACTER_FOR_TRAIN)).cuda()
        target_train = torch.from_numpy(data[stock+'_target_train'].values.astype(np.float32)[:,0].reshape(-1, 1, 1)).cuda()
        if data_train.shape[0] == 0:
            continue
        for i in range(num_epochs):                   
            out = model(data_train)
            loss = loss_function(out, target_train)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if (i+1) % 100 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
        
model = LSTM_Regression(CHARACTER_FOR_TRAIN, 8, output_size=1, num_layers=2).cuda()
loss_function = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for j in range(100):
    for stock in industry_dict[industry]['con_code']:
        data_train = torch.from_numpy(data[stock+'_data_train'].values.astype(np.float32).reshape(-1, 1, CHARACTER_FOR_TRAIN)).cuda()
        target_train = torch.from_numpy(data[stock+'_target_train'].values.astype(np.float32)[:,0].reshape(-1, 1, 1)).cuda()
        if data_train.shape[0] == 0:
            continue
        for i in range(10000):                   
            out = model(data_train)
            loss = loss_function(out, target_train)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if (i+1) % 100 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))

for i in model.parameters():
    print(i)
model = model.eval() # 转换成测试模式
data_test = torch.from_numpy(get_data.get_from_sql(
    stock_id='600419.SH_data_test',
    name='LSTM_data').values.astype(np.float32).reshape(-1, 1, CHARACTER_FOR_TRAIN)).cuda()
target_test = get_data.get_from_sql(
    stock_id='600419.SH_target_test',
    name='LSTM_data').values[:,0]

pred_test = model(data_test).cpu() # 全量训练集的模型输出 (seq_size, batch_size, output_size)
pred_test = pred_test.view(-1).data.numpy()

plt.plot(pred_test, 'r', label='prediction')
plt.plot(target_test, 'b', label='real')
plt.legend(loc='best')
plt.show()
np.mean(np.power((pred_test-target_test)*100, 2))

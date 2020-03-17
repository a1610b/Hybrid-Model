"""
Created on Fri Mar 13 20:35:29 2020

@author: Tony She

E-mail: tony_she@yahoo.com
"""

import sqlite3 as db
from torch import nn
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import functions.get_data as get_data
import numpy as np
import pandas as pd

# (Hyper parameters)
batch_size = 256
learning_rate = 1e-1
num_epochs = 20


class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), 
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), 
            nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class StockDataSet(Dataset):
    def __init__(self, data, target):
        self.data = data.float()
        self.target = target.float()
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    industry_dict = get_data.load_obj('industry_dict')
    model = SimpleNet(232, 50, 20, 10)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 0
    count = 0

    for i in range(num_epochs):
        for industry in industry_dict:
            data_train = get_data.get_from_sql(
                stock_id=industry+'_data_train',
                name='CNN_industry').values.astype(np.float32)
            if data_train.shape[0] == 0:
                continue
            target_train = get_data.get_from_sql(
                stock_id=industry+'_target_train',
                name='CNN_industry').values.astype(np.float32)
            characteristic = torch.from_numpy(data_train)
            label = torch.from_numpy(target_train)
            if characteristic.shape[0] < 2:
                continue
            if torch.cuda.is_available():
                characteristic = characteristic.cuda()
                label = label.cuda()
            else:
                characteristic = Variable(characteristic)
                label = Variable(label)
            for j in range(1000):
                out = model(characteristic)
                loss = criterion(out, label)
                print_loss = loss.data.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch += 1
                if epoch % 100 == 0:
                    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

                # train_dataset = StockDataSet(torch.from_numpy(data_train),
                #                              torch.from_numpy(target_train))
    
                # train_loader = DataLoader(train_dataset,
                #                           batch_size=batch_size,
                #                           shuffle=True
                #                           )
                # for data in train_loader:
                #     characteristic, label = data
                #     if characteristic.shape[0] < 2:
                #         continue
                #     if torch.cuda.is_available():
                #         characteristic = characteristic.cuda()
                #         label = label.cuda()
                #     else:
                #         characteristic = Variable(characteristic)
                #         label = Variable(label)
                #     out = model(characteristic)
                #     loss = criterion(out, label)
                #     print_loss = loss.data.item()
    
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                #     epoch += 1
                #     if epoch % 100 == 0:
                #         print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    # model.eval()
    # eval_loss = 0
    # eval_acc = 0
    # for data in test_loader:
    #     characteristic, label = data
    #     if torch.cuda.is_available():
    #         characteristic = characteristic.cuda()
    #         label = label.cuda()

    #     out = model(characteristic)
    #     loss = criterion(out, label)
    #     eval_loss += loss.data.item() * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     # num_correct = (pred == label).sum()
    #     # eval_acc += num_correct.item()
    # print('Test Loss: {:.6f}'.format(
    #     eval_loss / (len(test_dataset))
    # ))

"""
Created on Fri Mar 13 20:35:29 2020

@author: Tony She

E-mail: tony_she@yahoo.com
"""

from torch import nn
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import functions.get_data as get_data
import numpy as np

# (Hyper parameters)
batch_size = 100
learning_rate = 1e-4
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
        # 定义好 image 的路径
        self.data = data.float()
        self.target = target.float()
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    stockdata = get_data.get_from_sql(minimum_data=500)
    # for stock in stockdata:
    stock = '600419.SH'
    example = stockdata[stock]
    norm_factor = {}
    adjust_factor = ['low', 'close', 'open', 'high']
    for item in adjust_factor:
        example[item+'_adj'] = example[item] * example['adj_factor']
    example_1 = example[['low_adj', 'close_adj', 'open_adj', 'high_adj',
                         'pct_chg', 'pe_ttm','vol', 'turnover_rate', 'dv_ttm',
                         'float_share', 'turnover_rate_f', 'pb', 'ps_ttm',
                         'volume_ratio', 'adj_factor']]
    for i in range(1, 51):
        example_1['close_last_'+str(i)+"_adj"] = example_1['close_adj'].shift(i)
        example_1['open_last_'+str(i)+"_adj"] = example_1['open_adj'].shift(i)
        example_1['high_last_'+str(i)+"_adj"] = example_1['high_adj'].shift(i)
        example_1['low_last_'+str(i)+"_adj"] = example_1['low_adj'].shift(i)

    for i in range(1, 11):
        example_1['return_next_'+str(i)] = example_1['close_adj'].shift(-i)\
                                           / example_1['close_adj']
                                    
    example_1.dropna(axis=0, inplace=True)
    target = example_1.iloc[:, -10:]
    data = example_1.iloc[:, :-10]

    target_train = target.iloc[:-200]
    data_train = data.iloc[:-200]
    
    target_test = target.iloc[-200:]
    data_test = data.iloc[-200:]
   
    norm_factor[stock] = {}
    norm_factor[stock]['high'] = np.max(data_train['high_adj'])
    norm_factor[stock]['low'] = np.min(data_train['low_adj'])
    for item in data_train:
        if item[-3:] == 'adj':
            data_train[item] = (data_train[item]\
                               - norm_factor[stock]['low'])\
                               / (norm_factor[stock]['high']\
                               - norm_factor[stock]['low'])
            data_test[item] = (data_test[item]\
                              - norm_factor[stock]['low'])\
                              / (norm_factor[stock]['high']\
                              - norm_factor[stock]['low'])
        else:
            norm_factor[stock][item+'_high'] = np.max(data_train[item])
            norm_factor[stock][item+'_low'] = np.min(data_train[item])
            data_train[item] = (data_train[item]\
                               - norm_factor[stock][item+'_low'])\
                               / (norm_factor[stock][item+'_high']\
                               - norm_factor[stock][item+'_low'])

            data_test[item] = (data_test[item]\
                              - norm_factor[stock][item+'_low'])\
                              / (norm_factor[stock][item+'_high']\
                              - norm_factor[stock][item+'_low'])
            
    train_dataset = StockDataSet(torch.from_numpy(data_train.values),
                                 torch.from_numpy(target_train.values))
    test_dataset = StockDataSet(torch.from_numpy(data_test.values),
                                torch.from_numpy(target_test.values))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False
                             )

    model = SimpleNet(data_train.shape[1], 50, 20, 10)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    epoch = 0
    for data in train_loader:
        characteristic, label = data
        if torch.cuda.is_available():
            characteristic = characteristic.cuda()
            label = label.cuda()
        else:
            characteristic = Variable(characteristic)
            label = Variable(label)
        out = model(characteristic)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        characteristic, label = data
        if torch.cuda.is_available():
            characteristic = characteristic.cuda()
            label = label.cuda()

        out = model(characteristic)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))


from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler      # 数据预处理
from sklearn import metrics
import functions.get_data as get_data

class HiddenLayer:
    def __init__(self, x, num):
        """
        Define initialization.

        Args:
            x (TYPE): input data.
            num (TYPE): number of hidden layers.

        Returns:
            None.
        """
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        h = self.sigmoid(np.dot(x, self.w)+self.b)
        self.H_ = np.linalg.pinv(h)  # pseudo-inverse of matrix h
        # print(self.H_.shape)

    def sigmoid(self, x):
        """
        Define an activation function.

        Args:
            x: input.

        Returns:
            double: sigmoid function.

        """
        print(x)
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        T = T.reshape(-1, 1)
        self.beta = np.dot(self.H_, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()
        # 独热编码之后一定要用toarray()转换成正常的数组
        # T = np.asarray(T)
        print(self.H_.shape)
        print(T.shape)
        self.beta = np.dot(self.H_, T)
        print(self.beta.shape)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w)+self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w)+self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result



stdsc = StandardScaler()
stockdata = get_data.get_from_sql(minimum_data=500)
example = stockdata['600419.SH']
adjust_factor = ['low', 'close', 'open', 'high']
for item in adjust_factor:
    example[item+'_adj'] = example[item] * example['adj_factor']
example_1 = example[['low_adj', 'close_adj', 'open_adj', 'high_adj', 'pct_chg',
                     'pe_ttm','vol', 'turnover_rate', 'dv_ttm', 'float_share',
                     'turnover_rate_f', 'pb', 'ps_ttm', 'volume_ratio']]
for i in range(1, 11):
    example_1['close_adj_last_'+str(i)] = example_1['close_adj'].shift(i)
    example_1['open_adj_last_'+str(i)] = example_1['open_adj'].shift(i)
    example_1['high_adj_last_'+str(i)] = example_1['high_adj'].shift(i)
    example_1['low_adj_last_'+str(i)] = example_1['low_adj'].shift(i)

for i in range(1, 11):
    example_1['close_adj_next_'+str(i)] = example_1['close_adj'].shift(-i)

example_1.dropna(axis=0, inplace=True)
target = example_1.iloc[:, -10:]
data = example_1.iloc[:, :-10]

x, y = stdsc.fit_transform(data), target
x_train, x_test, y_train, y_test\
    = train_test_split(x, y, test_size=0.2, random_state=0)

a = HiddenLayer(x_train, 100)
a.regressor_train(y_train.iloc[:,1].values)
result = a.regressor_test(x_test)

print(result)
print(metrics.mean_squared_error(y_test.iloc[:,1].values, result))

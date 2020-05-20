# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:57:55 2020

@author: Tony She

E-mail: tony_she@yahoo.com
"""
import sys
sys.path.append('D:\Code\Hybrid Model')
import sqlite3 as db
from multiprocessing import Pool
import pandas as pd
import tushare as ts
import numpy as np
import functions.get_data as get_data
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
import talib
import warnings

warnings.filterwarnings("ignore")

def compute_discrete():
    data_dict = get_data.get_sql_key(name='rf_data_d')
    correct_rate = {}
    correct_rate[5] = []
    correct_rate[1] = []
    correct_rate[2] = []
    correct_rate[3] = []
    correct_rate[4] = []
    count = 1
    length = len(data_dict)
    unit = int(length / 100)
    for stock in data_dict:
        if count % unit == 0:
            print(count / unit)
        count += 1
        data = get_data.get_from_sql(name='rf_data_d', stock_id=stock)
        data = data[data['date'] > '20120101']
        if data.shape[0] < 1500:
            continue
        temp_data = data[['rsi_3', 'rsi_14', 'rsi_28']]
        temp_data[temp_data < 30] = 1
        temp_data[(temp_data >= 30) & (temp_data < 50)] = -1
        temp_data[(temp_data >= 50) & (temp_data < 70)] = 1
        temp_data[temp_data >= 70] = -1
        data[['rsi_3', 'rsi_14', 'rsi_28']] = temp_data
        index = [
            'rsi_3', 'ma20_price','return_last_1d', 'return_month',
            'return_month', 't_6_t_12', 'rsi_28', 'ma60_ma20', 'SO_k_d',
            'rsi_14', 'obv', 't_t_1', 't_12_t_18', 't_1_t_6',
            'corr_vol_close_month', 'ma120_price', 'ma120_ma40', 'ma60_price', 't_12_t_36', 'ma5_price',
            'corr_vol_close_year', 'ma20_ma5'
        ]
        data_x = data[index]
        data_x[data_x > 0] = 1
        data_x[data_x < 0] = -1
        data_y = (data['return_rate_1m'] > 0).astype(int)
        data_x[np.isnan(data_x)] = 0
        # print('*' * 50)
        # print(stock)
        for i in range(5, 0, -1):
            x_train = data_x.iloc[:- i * 200]
            y_train = data_y.iloc[:- i * 200]
            x_test = data_x.iloc[- i * 200 :-(i-1) * 200 - 1]
            y_test = data_y.iloc[- i * 200 :-(i-1) * 200 - 1]
            classifier = RandomForestClassifier(min_samples_leaf = 100, n_estimators=200,random_state=0, n_jobs=-1, class_weight='balanced_subsample')
            classifier.fit(x_train, y_train)
            y_pre = classifier.predict(x_test)
            # print(i)
            # print(np.nanmean((y_pre == y_test)))
            correct_rate[i].append(np.nanmean((y_pre == y_test)))
            '''
            importances = list(classifier.feature_importances_)
    
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(x_train.columns), importances)]
    
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)[:10]
    
            # Print out the feature and importances 
            [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
            '''
    for i in correct_rate:
        print(np.mean(correct_rate[i]))
        
def compute_talib():
    data_dict = get_data.get_sql_key(name='data')
    correct_rate = {}
    correct_rate[5] = []
    correct_rate[1] = []
    correct_rate[2] = []
    correct_rate[3] = []
    correct_rate[4] = []
    count = 1
    length = len(data_dict)
    unit = int(length / 100)
    for stock in data_dict:
        if count % unit == 0:
            print(count / unit)
        count += 1
        data = get_data.get_from_sql(stock_id=stock)
        # data = data[data['trade_date'] > '20120101']
        if data.shape[0] < 1500:
            continue
        adjusted_close = data['close'] * data['adj_factor']
        adjusted_high = data['high'] * data['adj_factor']
        adjusted_low = data['low'] * data['adj_factor']
        adjusted_open = data['open'] * data['adj_factor']
        data_x = pd.DataFrame()
        data_x['MACD'] = talib.MACD(adjusted_close)[1]
        data_x['RSI'] = talib.RSI(adjusted_close)
        data_x['WILLR'] = talib.WILLR(adjusted_high, adjusted_low, adjusted_open)
        data_y = (data['pct_chg'].shift(-1) > 0).astype(int)
        data_x[np.isnan(data_x)] = 0
        # print('*' * 50)
        # print(stock)
        for i in range(5, 0, -1):
            x_train = data_x.iloc[100:- i * 200]
            y_train = data_y.iloc[100:- i * 200]
            x_test = data_x.iloc[- i * 200 :-(i-1) * 200 - 1]
            y_test = data_y.iloc[- i * 200 :-(i-1) * 200 - 1]
            classifier = RandomForestClassifier(min_samples_leaf = 100, n_estimators=200,random_state=0, n_jobs=-1, class_weight='balanced_subsample')
            classifier.fit(x_train, y_train)
            y_pre = classifier.predict(x_test)
            # print(i)
            # print(np.nanmean((y_pre == y_test)))
            correct_rate[i].append(np.nanmean((y_pre == y_test)))
            '''
            importances = list(classifier.feature_importances_)
    
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(x_train.columns), importances)]
    
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)[:10]
    
            # Print out the feature and importances 
            [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
            '''
    for i in correct_rate:
        print(np.mean(correct_rate[i]))
        
        
if __name__ == '__main__':
    compute_talib()
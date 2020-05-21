# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:09:23 2020

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
# import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import talib
from sklearn.ensemble import RandomForestClassifier
import warnings


def divide_list(given_list, num):
    result = []
    length = int(len(given_list) / num)
    for i in range(num):
        if i == num - 1:
            result.append(given_list[i*length:])
        else:
            result.append(given_list[i*length: (i+1)*length])
    return result


def calc_accr_rf(stock_list, d):
    warnings.filterwarnings("ignore")
    result_stock = pd.DataFrame()
    con = db.connect('D:\\Data\\rf_data_research_target.sqlite')
    target_list = [ 'industry_return_1d',
        'industry_return_5d', 'industry_return_20d', 'index_return_1d',
        'index_return_5d', 'index_return_20d', 'all_return_1d', 'all_return_5d',
        'all_return_20d'
    ]
    target_list_abs = [
        'return_1d', 'return_5d', 'return_20d']
    train_end_date = '20180101'
    
    for stock in stock_list:
        try:
            data_train_x = get_data.get_from_sql(stock_id=stock,
                                                 name='rf_data_research')
            data_train_x.rename(columns={'index': 'date'}, inplace=True)
            data_train_x.drop(['return_1d', 'return_5d', 'return_20d'],
                              inplace=True,
                              axis=1)
            data_train_y = pd.read_sql_query(
                sql="SELECT * FROM Processed_data WHERE tick = '" + stock + "'",
                con=con)
            data_set = pd.merge(data_train_x,
                                data_train_y,
                                left_on='date',
                                right_on='date',
                                how='inner')
            data_set.fillna(0, inplace=True)
            temp_dict = {'tick': stock}
            temp_dict['index'] = data_train_y['index'].iloc[-1]
            temp_dict['industry'] = data_train_y['industry'].iloc[-1]
            
            for target in target_list:
                data_set_working = data_set[list(data_train_x.columns) + [target, 'return_1d']]
                data_set_working = data_set_working[data_set_working[target] != 0]
        
                train_data_set = data_set_working[
                    data_set_working['date'] < train_end_date]
                if train_data_set.shape[0] < 500:
                    break
                test_data_set = data_set_working[
                    data_set_working['date'] >= train_end_date]
                if train_data_set.shape[0] < 100:
                    break        
                train_data_set['target'] = -1
                train_data_set.loc[train_data_set[
                    train_data_set[target] >= 0.7].index, 'target'] = 1
                train_data_set.loc[train_data_set[
                    train_data_set[target] <= 0.3].index, 'target'] = 0
                train_data_set = train_data_set[train_data_set['target'] >= 0]
                train_x = train_data_set[data_train_x.columns].drop('date', axis=1)
                train_x[np.isinf(train_x)] = 0
                train_y = train_data_set['target']
        
                test_x = test_data_set[data_train_x.columns].drop('date', axis=1)
                test_x[np.isinf(test_x)] = 0
                test_y = (test_data_set[target] > 0.5).astype(int)
                
                classifier = RandomForestClassifier(min_samples_leaf = 100, n_estimators=200,random_state=0, n_jobs=-1, class_weight='balanced_subsample')
                classifier.fit(train_x, train_y)
                y_pre = classifier.predict(test_x)
                temp_dict[target] = round(np.nanmean(y_pre == test_y), 4)
                
                prob_y = classifier.predict_proba(test_x)
                prob_y = prob_y[:, 1]
                if not (prob_y <= 0.5).all():
                    prob_y[prob_y >= np.percentile(prob_y[prob_y>0.5], 70)] = 1
                if not (prob_y >= 0.5).all():
                    prob_y[prob_y <= np.percentile(prob_y[prob_y<0.5], 30)] = 0
                temp_dict[target+'_enhanced'] = round(np.nanmean((prob_y == test_y)) / 0.3, 4)
                
                if 'return_1d' in target:
                    temp_dict[target+'_return'] = np.sum(y_pre * test_data_set['return_1d'])
                    temp_dict[target+'_return_compound'] = np.prod(y_pre * test_data_set['return_1d']+1)
                    y_pre[y_pre==0] = -1
                    temp_dict[target+'_return_ls'] = np.sum(y_pre * test_data_set['return_1d'])
                    temp_dict[target+'_return_compound_ls'] = np.prod(y_pre * test_data_set['return_1d']+1)
                    
                    stor_prob_y = prob_y.copy()
                    prob_y[prob_y < 1] = 0
                    temp_dict[target+'_enhanced_return'] = np.sum(prob_y * test_data_set['return_1d'])
                    temp_dict[target+'_enhanced_return_compound'] = np.prod(prob_y * test_data_set['return_1d']+1)
                    stor_prob_y[stor_prob_y == 0] = -1
                    stor_prob_y[(stor_prob_y >= 0) & (stor_prob_y < 1)] = 0
                    temp_dict[target+'_enhanced_return_ls'] = np.sum(stor_prob_y * test_data_set['return_1d'])
                    temp_dict[target+'_enhanced_return_compound_ls'] = np.prod(stor_prob_y * test_data_set['return_1d']+1)

            for target in target_list_abs:
                data_set_working = data_set[list(data_train_x.columns) + [target]]
                data_set_working = data_set_working[data_set_working[target] != 0]
        
                train_data_set = data_set_working[
                    data_set_working['date'] < train_end_date]
                if train_data_set.shape[0] < 500:
                    break
                test_data_set = data_set_working[
                    data_set_working['date'] >= train_end_date]
                if train_data_set.shape[0] < 100:
                    break        
                train_data_set['target'] = -1
                train_data_set.loc[train_data_set[
                    train_data_set[target] >= 0].index, 'target'] = 1
                train_data_set.loc[train_data_set[
                    train_data_set[target] <= 0].index, 'target'] = 0
                train_data_set = train_data_set[train_data_set['target'] >= 0]
                train_x = train_data_set[data_train_x.columns].drop('date', axis=1)
                train_x[np.isinf(train_x)] = 0
                train_y = train_data_set['target']
        
                test_x = test_data_set[data_train_x.columns].drop('date', axis=1)
                test_x[np.isinf(test_x)] = 0
                test_y = (test_data_set[target] >= 0).astype(int)
                
                classifier = RandomForestClassifier(min_samples_leaf = 200, n_estimators=500,random_state=0, n_jobs=-1, class_weight='balanced_subsample')
                classifier.fit(train_x, train_y)
                y_pre = classifier.predict(test_x)
                temp_dict[target] = round(np.nanmean(y_pre == test_y), 4)
                
                prob_y = classifier.predict_proba(test_x)
                prob_y = prob_y[:, 1]
                if not (prob_y <= 0.5).all():
                    prob_y[prob_y >= np.percentile(prob_y[prob_y>0.5], 70)] = 1
                if not (prob_y >= 0.5).all():
                    prob_y[prob_y <= np.percentile(prob_y[prob_y<0.5], 30)] = 0
                temp_dict[target+'_enhanced'] = round(np.nanmean((prob_y == test_y)) / 0.3, 4)

                if 'return_1d' in target:
                    temp_dict[target+'_return'] = np.sum(y_pre * test_data_set['return_1d'])
                    temp_dict[target+'_return_compound'] = np.prod(y_pre * test_data_set['return_1d']+1)
                    y_pre[y_pre==0] = -1
                    temp_dict[target+'_return_ls'] = np.sum(y_pre * test_data_set['return_1d'])
                    temp_dict[target+'_return_compound_ls'] = np.prod(y_pre * test_data_set['return_1d']+1)
                    
                    stor_prob_y = prob_y.copy()
                    prob_y[prob_y < 1] = 0
                    temp_dict[target+'_enhanced_return'] = np.sum(prob_y * test_data_set['return_1d'])
                    temp_dict[target+'_enhanced_return_compound'] = np.prod(prob_y * test_data_set['return_1d']+1)
                    stor_prob_y[stor_prob_y == 0] = -1
                    stor_prob_y[(stor_prob_y >= 0) & (stor_prob_y < 1)] = 0
                    temp_dict[target+'_enhanced_return_ls'] = np.sum(stor_prob_y * test_data_set['return_1d'])
                    temp_dict[target+'_enhanced_return_compound_ls'] = np.prod(stor_prob_y * test_data_set['return_1d']+1)
            result_stock = result_stock.append(temp_dict, ignore_index=True)
        except Exception as e:
            print(stock)
            print(repr(e))
    result_stock.to_csv('D:\output\rf_result' + str(d) + '.csv')
    


def calc_accr():
    warnings.filterwarnings("ignore")


    result_stock = pd.DataFrame()
    stock_list = get_data.get_sql_key(name='rf_data_research')
    stock_list_list = divide_list(stock_list, 24)
    result = []
    
    p = Pool()
    for i in range(24):
        result.append(p.apply_async(calc_accr_rf, args=(stock_list_list[i], i, )))
    p.close()
    p.join()


    final_result = pd.DataFrame()
    for i in range(24):
        temp = pd.read_csv('D:\output\rf_result' + str(i) + '.csv')
        final_result = final_result.append(temp)
    final_result.to_csv('D:\output\rf_result.csv')
    print(final_result.mean())
    print(final_result.groupy('industry').mean())
    print(final_result.groupy('index').mean())


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    calc_accr()

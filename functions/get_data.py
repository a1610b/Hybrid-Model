# -*- coding: utf-8 -*-
"""
Created on 2019/7/10

@author: Tony She

E-mail: tony_she@yahoo.com

This module provides socket operations and some related functions.
On Unix, it supports IP (Internet Protocol) and Unix domain sockets.
On other systems, it only supports IP. Functions specific for a
socket are available as methods of the socket object.

Functions:

socket() -- create a new socket object
"""

import pickle
import logging
import sqlite3 as db
import time
from multiprocessing import Pool

import pandas as pd
import tushare as ts
import numpy as np


def get_from_sql(item: str = '*',
                 stock_id: str = 'all',
                 name: str = 'data',
                 start_date: str = '19910101',
                 minimum_data: int = 0
                 ):
    """
    Get dict of Dataframe data from the given sql_path and item.

    Args:
        item (str, optional): Column that return. Defaults to '*'.
        stock_id(str, optional): The stock data that returns.
        name (str, optional): SQL's path. Defaults to 'data'.
        start_date (str, optional): The starting date of data.
            Defaults to '19910101'.
        minimum_data (int, optional): Reject any stock with data number
            less than minimum_data. Defaults to 0.

    Returns:
        data (dict): Dict of Dataframe if all data were required. If only one
        stock is required, then return a single dataframe.

    """
    con = db.connect('D:\\Data\\' + name + '.sqlite')
    cur = con.cursor()
    data = {}
    cur.execute("select name from sqlite_master where type='table'")
    stock_list = cur.fetchall()
    stock_list = [line[0] for line in stock_list]
    if stock_id == 'all':
        # get all table from the database

        # read the dataframe one by one from the stock_list
        for stock in stock_list:
            data[stock] = pd.read_sql_query(
                sql="SELECT " + item + " FROM '" + stock + "'",
                con=con
                )
            # print(data[stock])
            if data[stock].shape[0] < minimum_data:
                data.pop(stock)
            # data[stock] = data[stock][data[stock]['trade_date'] > start_date]
    else:
        if stock_id not in stock_list:
            return pd.DataFrame()
        data = pd.read_sql_query(
                sql="SELECT " + item + " FROM '" + stock_id + "'",
                con=con
                )
        if data.shape[0] < minimum_data:
            return pd.DataFrame()
    cur.close()
    con.close()
    return data


def get_industry_stock_list(level: int = 1) -> dict:
    """
    Return a dict consist with list of stock code in each SW Level 1 Industry.

    Args:
        level (int, optional): The level of sw industry we want.

    """
    ts.set_token('267addf63a14adcfc98067fc253fbd72a728461706acf9474c0dae29')
    pro = ts.pro_api()
    df = pro.index_classify(level='L' + str(level), src='SW')['index_code']
    industry_stock_list = {}

    count = 1
    for i in df:
        if count % 100 == 0:
            time.sleep(60)
        count += 1
        industry_stock_list[i] = pro.index_member(index_code=i)
    return industry_stock_list


# This is an old function that reads data from local csc files.
# Temporarily removed.
'''
def getDataFromeCSV():

    filename = input('filename: ')
    data = {}
    data['first'] = pd.read_csv(filename)
    data['first'].columns = ['close']
    data['first']['trade_date'] = pd.date_range(
        start='2019-1-09',
        periods=data['first'].shape[0],
        freq='-1D'
        )
    return data
'''


def download_all_market_data(sqlname: str = 'data'):
    """
    Download all stock's data exist in tushare database.

    Args:
        sqlname (str, optional): The name of the local db. Defaults to 'data'.

    """
    ts.set_token('267addf63a14adcfc98067fc253fbd72a728461706acf9474c0dae29')
    pro = ts.pro_api()
    LOG_FORMAT = "%(asctime)s====%(levelname)s++++%(message)s"
    logging.basicConfig(filename="download.log",
                        level=logging.ERROR,
                        format=LOG_FORMAT)

    # Get stock ID of the stock that are now trading(L), suspend trading(P)
    # and delisted(D)
    stock_list = set(pro.stock_basic(exchange='',
                                     list_status='D',
                                     fields='ts_code')['ts_code']) \
        | set(pro.stock_basic(exchange='',
                              list_status='L',
                              fields='ts_code')['ts_code']) \
        | set(pro.stock_basic(exchange='',
                              list_status='P',
                              fields='ts_code')['ts_code'])

    con = db.connect('D:\\Data\\'+sqlname+'.sqlite')
    cur = con.cursor()

    count = 0
    stock_list = list(stock_list)[::-1]
    one_percent = int(len(stock_list) / 100)
    for i in stock_list:
        try:
            # Avoid calling tushare too frequent
            count += 1
            '''
            if count % 10 == 0:
                time.sleep(10)
            '''

            # Show the progress of the downloading
            if count % one_percent == 0:
                print(count / one_percent)

            df1 = pro.daily(ts_code=i)           # Basic price data
            df2 = pro.daily_basic(ts_code=i)     # Fundamental data
            df3 = pro.adj_factor(ts_code=i)      # Price correction factor
            df4 = pro.moneyflow(ts_code=i)       # Cashflow trend
            df5 = pro.margin_detail(ts_code=i)   # Margin trading data

            stock_data = df1[set(df1.columns) - {'ts_code'}].merge(
                    df2[set(df2.columns) - {'ts_code', 'close'}],
                    how='left',
                    right_on='trade_date',
                    left_on='trade_date'
                    )
            stock_data = stock_data.merge(
                    df3[['trade_date', 'adj_factor']],
                    how='left',
                    right_on='trade_date',
                    left_on='trade_date'
                    )
            stock_data = stock_data.merge(
                    df4[set(df4.columns) - {'ts_code'}],
                    how='left',
                    right_on='trade_date',
                    left_on='trade_date'
                    )
            stock_data = stock_data.merge(
                    df5[set(df5.columns) - {'ts_code'}],
                    how='left',
                    right_on='trade_date',
                    left_on='trade_date'
                    )
            stock_data.dropna(how='all', axis=1, inplace=True)
            stock_data = stock_data.iloc[::-1]
            stock_data.reset_index(drop=True, inplace=True)

            # Ignore stock that don't have data yet
            if stock_data.shape[0] == 0:
                continue

            # Write in the database
            stock_data.to_sql(
                name=i,
                con=con,
                if_exists='replace',
                index=False
                )
            con.commit()
        except Exception:
            print(i)
            logging.error('%s', i)
    cur.close()
    con.close()
    return 'Done'


def prep_data_for_cnn(industry_list, industry):
    using_factor = ['low_adj', 'close_adj', 'open_adj', 'high_adj', 'pct_chg',
                    'pe_ttm', 'vol', 'turnover_rate', 'float_share',
                    'turnover_rate_f', 'pb', 'ps_ttm', 'volume_ratio',
                    'adj_factor', 'buy_md_amount', 'buy_lg_vol',
                    'sell_md_amount', 'sell_sm_vol', 'buy_sm_amount',
                    'buy_md_vol', 'sell_sm_amount', 'sell_elg_vol',
                    'buy_elg_amount', 'sell_md_vol', 'sell_lg_vol',
                    'buy_elg_vol', 'sell_lg_amount', 'sell_elg_amount',
                    'net_mf_vol', 'buy_lg_amount', 'buy_sm_vol',
                    'net_mf_amount']
    adjust_factor = ['low', 'close', 'open', 'high']
    norm_factor = pd.DataFrame()
    first = True

    for stock in industry_list:
        df = get_from_sql(stock_id=stock)
        print(stock)
        # print(df.head())
        if df.shape[0] == 0:
            continue

        for item in adjust_factor:
            df[item+'_adj'] = df[item] * df['adj_factor']

        # If the factor required is not provided by the stock, then the stock
        # is removed from the list.
        if not set(using_factor).issubset(set(df.columns)):
            print(stock)
            continue
        df = df[using_factor]

        for i in range(1, 51):
            df['close_last_'+str(i)+"_adj"] = df['close_adj'].shift(i)
            df['open_last_'+str(i)+"_adj"] = df['open_adj'].shift(i)
            df['high_last_'+str(i)+"_adj"] = df['high_adj'].shift(i)
            df['low_last_'+str(i)+"_adj"] = df['low_adj'].shift(i)

        for i in range(1, 11):
            df['return_next_'+str(i)] = df['close_adj'].shift(-i)\
                                        / df['close_adj']

        df.dropna(axis=0, inplace=True)
        target = df.iloc[:, -10:]
        data = df.iloc[:, :-10]
        target_train_temp = target.iloc[:-200]
        data_train_temp = data.iloc[:-200]
        target_test_temp = target.iloc[-200:]
        data_test_temp = data.iloc[-200:]

        norm_factor_temp = {}
        norm_factor_temp['stock'] = stock
        norm_factor_temp['high'] = np.max(data_train_temp['high_adj'])
        norm_factor_temp['low'] = np.min(data_train_temp['low_adj'])
        for item in data_train_temp:
            if item[-3:] == 'adj':
                data_train_temp[item] = (data_train_temp[item]
                                    - norm_factor_temp['low'])\
                                    / (norm_factor_temp['high']
                                       - norm_factor_temp['low'])
                data_test_temp[item] = (data_test_temp[item]
                                   - norm_factor_temp['low'])\
                                  / (norm_factor_temp['high']
                                     - norm_factor_temp['low'])
            else:
                norm_factor_temp[item+'_high'] = np.max(data_train_temp[item])
                norm_factor_temp[item+'_low'] = np.min(data_train_temp[item])
                data_train_temp[item] = (data_train_temp[item]
                                    - norm_factor_temp[item+'_low'])\
                                   / (norm_factor_temp[item+'_high']
                                      - norm_factor_temp[item+'_low'])
                data_test_temp[item] = (data_test_temp[item]
                                   - norm_factor_temp[item+'_low'])\
                                  / (norm_factor_temp[item+'_high']
                                     - norm_factor_temp[item+'_low'])
        norm_factor = norm_factor.append(norm_factor_temp, ignore_index=True)
        
        if first:
            target_train = target_train_temp.values
            data_train = data_train_temp.values
            target_test = target_test_temp.values
            data_test = data_test_temp.values
            first = False
        else:
            target_train = np.vstack((target_train, target_train_temp.values))
            data_train = np.vstack((data_train, data_train_temp.values))
            target_test = np.vstack((target_test, target_test_temp.values))
            data_test = np.vstack((data_test, data_test_temp.values))

    # writing the information to database
    con = db.connect('D:\\Data\\CNN_industry.sqlite')
    pd.DataFrame(data_train).dropna().to_sql(name=industry+"_data_train",
                                             con=con,
                                             if_exists='replace',
                                             index=False
                                             )
    pd.DataFrame(data_test).dropna().to_sql(name=industry+"_data_test",
                                            con=con,
                                            if_exists='replace',
                                            index=False
                                            )
    pd.DataFrame(target_train).dropna().to_sql(name=industry+"_target_train",
                                               con=con,
                                               if_exists='replace',
                                               index=False
                                               )
    pd.DataFrame(target_test).dropna().to_sql(name=industry+"_target_train",
                                              con=con,
                                              if_exists='replace',
                                              index=False
                                              )
    pd.DataFrame(norm_factor).dropna().to_sql(name=industry+"_norm_factor",
                                              con=con,
                                              if_exists='replace',
                                              index=False
                                              )
    con.commit()
    con.close()
    print('Done')
    return None


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_industry_dict(level = 3):
    industry_dict = get_industry_stock_list(level)
    save_obj(industry_dict, 'industry_dict')


def construct_data_for_cnn():
    industry_dict = load_obj('industry_dict')
    p = Pool()
    print('start pooling')
    for industry in industry_dict:
        p.apply_async(prep_data_for_cnn,
                      args=(industry_dict[industry]['con_code'], industry,))
    p.close()
    p.join()


def main():
    download_all_market_data()
    print('done')


if __name__ == '__main__':
    construct_data_for_cnn()

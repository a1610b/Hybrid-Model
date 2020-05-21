# -*- coding: utf-8 -*-
import os
import sqlite3 as db
from multiprocessing import Pool
import pandas as pd
import tushare as ts
import numpy as np
import functions.get_data as get_data
import math
import talib


def prep_data_for_rf_improve(stock_list, const):
    """
    Calculate the daily indicators for stocks in the given stock_list.

    Args:
        stock_list (list): The stocks that need to be calculated.
        const (TYPE): Multiprocessing id.

    Returns:
        None.

    """
    con = db.connect('D:\\Data\\rf_data_research_'+str(const)+'.sqlite')
    for stock in stock_list:
        try:
            print(stock, const)
            df = get_data.get_from_sql(stock_id=stock)
            df_f = get_data.get_from_sql(stock_id=stock, name='data_finance')
            df.set_index('trade_date', inplace=True)
            df_f.set_index('ann_date', inplace=True)
            df_f = df_f[::-1]
            df_f = df_f.drop(['ts_code', 'end_date', 'dt_eps'], axis=1)
            df_m = pd.merge(df_f, df, right_index = True, left_index=True, how='outer')
            df_m.fillna(method='ffill', inplace=True)
            df_m['pct_chg'] = (df_m['close'] / df_m['pre_close'] - 1) * 100
            for i in ['high', 'low', 'close', 'open']:
                df_m[i + '_adj'] = df_m[i] * df_m['adj_factor']
            df_m['pre_close_adj'] = df_m['close_adj'] / (df_m['pct_chg'] / 100 + 1)
            df_m['MACD'] = talib.MACD(df_m['close_adj'])[1]
            df_m['WILLR'] = talib.WILLR(df_m['high_adj'], df_m['low_adj'],
                                        df_m['close_adj'])
            df_m['AD'] = talib.AD(df_m['high_adj'], df_m['low_adj'], df_m['close_adj'],
                                  df_m['vol'])
            df_m['RSI_3'] = talib.RSI(df_m['close_adj'], timeperiod=3)
            df_m['RSI_14'] = talib.RSI(df_m['close_adj'], timeperiod=14)
            df_m['RSI_28'] = talib.RSI(df_m['close_adj'], timeperiod=28)
            df_m['CCI_3'] = talib.CCI(df_m['high_adj'],
                                      df_m['low_adj'],
                                      df_m['close_adj'],
                                      timeperiod=3)
            df_m['CCI_14'] = talib.CCI(df_m['high_adj'],
                                       df_m['low_adj'],
                                       df_m['close_adj'],
                                       timeperiod=14)
            df_m['CCI_28'] = talib.CCI(df_m['high_adj'],
                                       df_m['low_adj'],
                                       df_m['close_adj'],
                                       timeperiod=28)
            df_m['WILLR'] = talib.WILLR(df_m['high_adj'], df_m['low_adj'],
                                        df_m['close_adj'])

            df_m['SMA5'] = (talib.MA(df_m['close_adj'], timeperiod=5) - df_m['close_adj']) / df_m['close_adj']
            df_m['SMA20'] = (talib.MA(df_m['close_adj'], timeperiod=20) - df_m['close_adj']) / df_m['close_adj']
            df_m['SMA60'] = (talib.MA(df_m['close_adj'], timeperiod=60) - df_m['close_adj']) / df_m['close_adj']
            df_m['SMA20_5'] = (talib.MA(df_m['close_adj'], timeperiod=20) - df_m['SMA5']) / df_m['SMA5']
            df_m['SMA60_5'] = (talib.MA(df_m['close_adj'], timeperiod=60) - df_m['SMA5']) / df_m['SMA5']
            df_m['SMA60_20'] = (talib.MA(df_m['close_adj'], timeperiod=60) - df_m['SMA20']) / df_m['SMA20']

            df_m['SMA5_tr'] = talib.MA(df_m['turnover_rate'], timeperiod=5)
            df_m['SMA20_tr'] = talib.MA(df_m['turnover_rate'], timeperiod=20)
            df_m['SMA60_tr'] = talib.MA(df_m['turnover_rate'], timeperiod=60)
            df_m['SMA250_tr'] = talib.MA(df_m['turnover_rate'], timeperiod=250)
            df_m['SMA5_tr_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=5)
                                  - df_m['turnover_rate']) / df_m['turnover_rate']
            df_m['SMA20_tr_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=20)
                                   - df_m['turnover_rate']) / df_m['turnover_rate']
            df_m['SMA60_tr_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=60)
                                   - df_m['turnover_rate']) / df_m['turnover_rate']
            df_m['SMA250_tr_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=250)
                                    - df_m['turnover_rate']) / df_m['turnover_rate']
            df_m['SMA20_SMA5_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=20)
                                     - df_m['SMA5_tr']) / df_m['SMA5_tr']
            df_m['SMA60_SMA5_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=60)
                                     - df_m['SMA5_tr']) / df_m['SMA5_tr']
            df_m['SMA250_SMA5_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=250)
                                      - df_m['SMA5_tr']) / df_m['SMA5_tr']
            df_m['SMA60_SMA20_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=60)
                                      - df_m['SMA20_tr']) / df_m['SMA20_tr']
            df_m['SMA250_SMA20_tr'] = (talib.MA(df_m['turnover_rate'], timeperiod=250)
                                       - df_m['SMA20_tr']) / df_m['SMA20_tr']

            df_m['corr_close_vol_20'] = talib.CORREL(df_m['close_adj'],
                                                     df_m['vol'],
                                                     timeperiod=20)
            df_m['corr_close_vol_60'] = talib.CORREL(df_m['close_adj'],
                                                     df_m['vol'],
                                                     timeperiod=60)
            df_m['pct_chg_STD_20'] = talib.STDDEV(df_m['pct_chg'], timeperiod=20)
            df_m['pct_chg_STD_60'] = talib.STDDEV(df_m['pct_chg'], timeperiod=60)
            df_m['pct_chg_STD_250'] = talib.STDDEV(df_m['pct_chg'], timeperiod=250)
            df_m['OBV'] = talib.OBV(df_m['pct_chg'], df_m['vol'])
            df_m['MOM_5'] = talib.MOM(df_m['close_adj'], timeperiod=5) / df_m['close_adj'].shift(5)
            df_m['MOM_20'] = talib.MOM(df_m['close_adj'], timeperiod=20) / df_m['close_adj'].shift(20)
            df_m['MOM_60'] = talib.MOM(df_m['close_adj'], timeperiod=60) / df_m['close_adj'].shift(60)
            df_m['MOM_250'] = talib.MOM(df_m['close_adj'], timeperiod=250) / df_m['close_adj'].shift(250)
            df_m['t_1_t_6'] = (talib.MOM(df_m['close_adj'], timeperiod=100) / df_m['close_adj'].shift(100)).shift(20)
            df_m['t_6_t_12'] = (talib.MOM(df_m['close_adj'], timeperiod=120) / df_m['close_adj'].shift(120)).shift(120)

            df_m['close_kurt_20'] = df_m['pct_chg'].rolling(20).kurt()
            df_m['close_kurt_60'] = df_m['pct_chg'].rolling(60).kurt()
            df_m['close_skew_20'] = df_m['pct_chg'].rolling(20).skew()
            df_m['close_skew_60'] = df_m['pct_chg'].rolling(60).skew()

            df_m['open_close_5'] = (df_m['close_adj'] / df_m['open_adj'] -
                                    1).rolling(5).mean()
            df_m['open_close_20'] = (df_m['close_adj'] / df_m['open_adj'] -
                                     1).rolling(20).mean()
            df_m['open_close_60'] = (df_m['close_adj'] / df_m['open_adj'] -
                                     1).rolling(60).mean()
            df_m['open_high_5'] = (df_m['high_adj'] / df_m['open_adj'] -
                                   1).rolling(5).mean()
            df_m['open_high_20'] = (df_m['high_adj'] / df_m['open_adj'] -
                                    1).rolling(20).mean()
            df_m['open_close_60'] = (df_m['high_adj'] / df_m['open_adj'] -
                                     1).rolling(60).mean()
            df_m['close_low_5'] = (df_m['close_adj'] / df_m['open_adj'] -
                                   1).rolling(5).mean()
            df_m['close_low_20'] = (df_m['close_adj'] / df_m['open_adj'] -
                                    1).rolling(20).mean()
            df_m['close_low_60'] = (df_m['close_adj'] / df_m['open_adj'] -
                                    1).rolling(60).mean()
            df_m['open_pre_close_5'] = (df_m['open_adj'] / df_m['pre_close_adj'] -
                                        1).rolling(5).mean()
            df_m['open_pre_close_20'] = (df_m['open_adj'] / df_m['pre_close_adj'] -
                                         1).rolling(20).mean()
            df_m['open_pre_close_60'] = (df_m['open_adj'] / df_m['pre_close_adj'] -
                                         1).rolling(60).mean()

            df_m['trend_strength_20'] = df_m['MOM_20'] / np.abs(
                df_m['close_adj'] - df_m['pre_close_adj']).rolling(20).sum()

            df_m['return_1d'] = (df_m['close_adj'].shift(-1) - df_m['close_adj']) / df_m['close_adj']
            df_m['return_5d'] = (df_m['close_adj'].shift(-5) - df_m['close_adj']) / df_m['close_adj']
            df_m['return_20d'] = (df_m['close_adj'].shift(-20) - df_m['close_adj']) / df_m['close_adj']

            df_m.to_sql(
                name=stock,
                con=con,
                if_exists='replace',
                index_label='date'
            )
            con.commit()
        except Exception as e:
            print(stock)
            print(repr(e))
    con.close()


def gather_target_df():
    """
    Gather the return information from the downloaded sql and give each day a label.

    Returns:
        None.

    """
    con = db.connect('D:\\Data\\rf_data_research_target.sqlite')

    dict_industry = get_data.get_industry_stock_list()
    data_list = get_data.get_sql_key()
    data_f_list = get_data.get_sql_key(name='data_finance')
    data_l = set(data_list)

    for i in dict_industry:
        dict_industry[i] = list(dict_industry[i]['con_code'])
        data_l = data_l - set(dict_industry[i]['con_code'])
    dict_industry['None'] = list(data_l)

    result = pd.DataFrame()
    for industry in dict_industry:
        for stock in dict_industry[industry]['con_code']:
            if (stock not in data_f_list) or (stock not in data_list):
                continue
            df = get_data.get_from_sql(stock_id=stock)
            df_m = pd.DataFrame()
            df_m['close_adj'] = df['close'] * df['adj_factor']
            df_m['return_1d'] = (df_m['close_adj'].shift(-1) - df_m['close_adj']) / df_m['close_adj']
            df_m['return_5d'] = (df_m['close_adj'].shift(-5) - df_m['close_adj']) / df_m['close_adj']
            df_m['return_20d'] = (df_m['close_adj'].shift(-20) - df_m['close_adj']) / df_m['close_adj']
            df_m['tick'] = stock
            df_m = df_m[['return_1d', 'return_5d', 'return_20d', 'tick']]
            df_m['industry'] = industry
            df_m['date'] = df['trade_date']
            result = result.append(df_m)

    result.to_sql(
        name='All_Data',
        con=con,
        if_exists='replace',
        index=False
    )
    con.commit()
    con.close()
    ts.set_token('267addf63a14adcfc98067fc253fbd72a728461706acf9474c0dae29')
    pro = ts.pro_api()
    dict_300 = {}
    for i in range(14):
        dict_300[str(2007+i)+'0101'] = list(pro.index_weight(index_code='399300.SZ',
                                                             start_date=str(2007+i)+'0101',
                                                             end_date=str(2007+i)+'0110')['con_code'].iloc[:300])
        dict_300[str(2007+i)+'0701'] = list(pro.index_weight(index_code='399300.SZ',
                                                             start_date=str(2007+i)+'0625',
                                                             end_date=str(2007+i)+'0701')['con_code'].iloc[:300])
    dict_500 = {}
    for i in range(14):
        dict_500[str(2007+i)+'0101'] = list(pro.index_weight(index_code='000905.SH',
                                                             start_date=str(2007+i)+'0101',
                                                             end_date=str(2007+i)+'0201')['con_code'].iloc[:500])
        dict_500[str(2007+i)+'0701'] = list(pro.index_weight(index_code='000905.SH',
                                                             start_date=str(2007+i)+'0625',
                                                             end_date=str(2007+i)+'0710')['con_code'].iloc[:500])
    calendar = pro.trade_cal(exchange='')
    calendar = calendar[calendar['is_open'] == 1]['cal_date']
    dict_industry = get_data.get_industry_stock_list()
    stock_list = get_data.get_sql_key()
    # prep_data_for_rf(stock_list, dict_industry, calendar, 1, dict_300, dict_500)
    stock_list_list = []
    length = int(len(stock_list) / 24)
    for i in range(24):
        if i == 23:
            stock_list_list.append(stock_list[i*length:])
        else:
            stock_list_list.append(stock_list[i*length: (i+1)*length])
    p = Pool()
    for i in range(24):
        p.apply_async(prep_data_for_rf, args=(stock_list_list[i], dict_industry, calendar, i, dict_300, dict_500, ))
    p.close()
    p.join()
    data = {}
    for i in range(24):
        data_temp = get_data.get_from_sql(name='rf_temp_' + str(i))
        data = {**data, **data_temp}
    con = db.connect('D:\\Data\\rf_data_d.sqlite')
    cur = con.cursor()
    for stock in data:
        data[stock].to_sql(
            name=stock,
            con=con,
            if_exists='replace',
            index=False
            )
    con.commit()
    cur.close()
    con.close()
    return None


def divide_list(given_list, num):
    """
    Divide the given list to a list that has num element, each element is a sub list of the given list.

    Args:
        given_list (list): The list that want to be separated.
        num (int): The number of element in the new list.

    Returns:
        result (list): The list that has been separated.

    """
    result = []
    length = int(len(given_list) / num)
    for i in range(num):
        if i == num - 1:
            result.append(given_list[i*length:])
        else:
            result.append(given_list[i*length: (i+1)*length])
    return result


def cal_relative_return(df, dict_industry, d):
    """
    Calculate daily relative return cross-sectionally based on the data gathered.

    Args:
        df (DataFrame): Dataframe that has labeled daily return.
        dict_industry (dict): A dict that contain stock list for every industry.
        d (int): Multiprocessing id.

    Returns:
        None.

    """
    date_list = df['date'].drop_duplicates()
    for i in ['industry_return_1d', 'industry_return_5d', 'industry_return_20d', 'index_return_1d',
              'index_return_5d', 'index_return_20d', 'all_return_1d', 'all_return_5d', 'all_return_20d']:
        df[i] = 0
    for date in date_list:
        try:
            print(date, d)
            temp = df[df['date'] == date]
            df.loc[temp.index, 'all_return_1d'] = temp['return_1d'].rank(pct=True)
            df.loc[temp.index, 'all_return_5d'] = temp['return_5d'].rank(pct=True)
            df.loc[temp.index, 'all_return_20d'] = temp['return_20d'].rank(pct=True)
            for industry in dict_industry:
                temp_i = temp[temp['industry'] == industry]
                df.loc[temp_i.index, 'industry_return_1d'] = temp_i['return_1d'].rank(pct=True)
                df.loc[temp_i.index, 'industry_return_5d'] = temp_i['return_5d'].rank(pct=True)
                df.loc[temp_i.index, 'industry_return_20d'] = temp_i['return_20d'].rank(pct=True)
            for index in ['zz500', 'hs300', 'None']:
                temp_i = temp[temp['index'] == index]
                df.loc[temp_i.index, 'index_return_1d'] = temp_i['return_1d'].rank(pct=True)
                df.loc[temp_i.index, 'index_return_5d'] = temp_i['return_5d'].rank(pct=True)
                df.loc[temp_i.index, 'index_return_20d'] = temp_i['return_20d'].rank(pct=True)
        except Exception as e:
            print(d)
            print(repr(e))
    con = db.connect('D:\\Data\\rf_data_research_target.sqlite')
    df.to_sql(name='Processed_data_'+str(d),
              con=con,
              if_exists='replace',
              index=False
              )
    con.commit()
    con.close()


def give_relative_return():
    """
    Calculate daily relative return cross-sectionally based on the data gathered.

    Multiprocessing is used in this function to speed up the process.

    Returns:
        None.

    """
    result = get_data.get_from_sql(stock_id='All_Data', name='rf_data_research_target')
    result = result[result['date'] >= '20070101']
    result = result.sort_values(by=['date']).reset_index(drop=True)
    sub_result = {}
    cutting = list(result['date'].drop_duplicates().index)
    unit = int(len(cutting) / 24)
    for i in range(24):
        if i != 23:
            sub_result[i] = result.loc[cutting[i * unit]: cutting[(i + 1) * unit] - 1]
        else:
            sub_result[i] = result.loc[cutting[i * unit]:]
    dict_industry = get_data.get_industry_stock_list()
    p = Pool()
    print('Start pooling')
    for i in range(24):
        p.apply_async(cal_relative_return, args=(sub_result[i], dict_industry, i, ))
    p.close()
    p.join()
    print('Done pooling')
    final = pd.DataFrame()
    con = db.connect('D:\\Data\\rf_data_research_target.sqlite')
    for i in range(24):
        final = final.append(get_data.get_from_sql(stock_id='Processed_data_'+str(i), name='rf_data_research_target'))
    final.to_sql(
                name='Processed_data',
                con=con,
                if_exists='replace',
                index=False
                )
    con.commit()
    con.close()


def prep_data_rf_improved():
    """
    Calculate daily indicators and store them in rf_data_research.

    Multiprocessing is used in this function to speed up the process.

    Returns:
        None.

    """
    stock_list = get_data.get_sql_key()
    stock_list_f = get_data.get_sql_key(name='data_finance')
    stock_list = list(set(stock_list) & set(stock_list_f))
    stock_list_list = divide_list(stock_list, 24)
    print('Start')
    p = Pool()
    for i in range(24):
        p.apply_async(prep_data_for_rf_improve, args=(stock_list_list[i], i,))
    p.close()
    p.join()
    con = db.connect('D:\\Data\\rf_data_research.sqlite')
    cur = con.cursor()
    for i in range(24):
        data_temp = get_data.get_from_sql(name='rf_data_research_'+str(i))
        for stock in data_temp:
            data_temp[stock].to_sql(
                name=stock,
                con=con,
                if_exists='replace',
                index=False
                )
        os.remove('D:\\Data\\rf_data_research_' + str(i) + '.sqlite')
    con.commit()
    cur.close()
    con.close()


def main():
    """
    The order function should be called for a new computer.

    Returns:
        None.

    """
    # if the computer have not download the basic and fundamental data
    # get_data.download_all_market_data()
    # get_data.download_all_market_data_finance()

    # calculate relative return cross-sectionally and store them in rf_data_target.sql
    gather_target_df()
    cal_relative_return()

    # calculate daily indicators and store them in rf_data_research
    prep_data_rf_improved()


if __name__ == '__main__':
    gather_target_df()

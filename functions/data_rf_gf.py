# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:05:50 2020

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


def prep_data_for_rf(stock_list, dict_industry, calendar, i, dict_300, dict_500):
    con = db.connect('D:\\Data\\rf_temp_'+str(i)+'.sqlite')
    cur = con.cursor()
    for stock in stock_list:
        try:
            df = get_data.get_from_sql(stock_id=stock)
            df_f = get_data.get_from_sql(stock_id=stock, name='data_finance')
            df.set_index('trade_date', inplace=True)
            df_f.set_index('ann_date', inplace=True)
            listed_date = df.index[0]
            usable = max(change_month(listed_date, 13), '20070101')
            if (usable > '20200101') or (usable > str(df.index[-1])):
                continue
            if 'turnover_rate_f' not in df.columns:
                df['turnover_rate_f'] = df['turnover_rate']
            first_date = df.index[df.index >= usable][0]
            current_date = first_date
            df['pct_chg'] = (df['close'] / df['pre_close'] - 1) * 100
            data = pd.DataFrame()
            while (current_date <= '20200101') and (current_date < df.index[-30]):
                # stock_date = df.index[df.index >= current_date][0]
                '''
                if stock_date != current_date:
                    # If the stock is not trading in the first trading day of the month,
                    # we don't collect its data and won't do trading on that stock this month.
                    continue
                '''
                last_trading_day = df.index[df.index < current_date][-1]
                existing_data = df.loc[: last_trading_day]
                last_year = change_month(current_date, -12)
                last_month = change_month(current_date, -1)
                last_year = df.index[df.index >= last_year][0]
                last_month = df.index[df.index >= last_month][0]
                last_year_df = df.loc[last_year: last_trading_day]
                last_month_df = df.loc[last_month: last_trading_day]
                f_date = df_f.index[df_f.index <= current_date][0]
                next_date = df.index[df.index > current_date][0]
                next_5d = df.index[df.index > current_date][4]
                last_5d = df.index[df.index < current_date][-5]
            
                price = df.loc[last_trading_day, 'close'] * df.loc[last_trading_day, 'adj_factor']
                if df_f.loc[f_date, 'rd_exp'] and df.loc[last_trading_day, 'total_mv'] and df.loc[last_trading_day, 'pe_ttm']:
                    rd_exp_to_earning = df_f.loc[f_date, 'rd_exp']\
                        / df.loc[last_trading_day, 'total_mv']\
                        * df.loc[last_trading_day, 'pe_ttm']
                else:
                    rd_exp_to_earning = np.nan
                if df_f.loc[f_date, 'fcfe'] and df.loc[last_trading_day, 'total_mv']:
                    fcfe = df_f.loc[f_date, 'fcfe'] / df.loc[last_trading_day, 'total_mv'] / 10000
                else:
                    fcfe = np.nan
            
                if df.index[df.index > next_date].shape[0] == 0:
                    break
                return_rate_1d = df.loc[next_date, 'close'] / df.loc[next_date, 'pre_close'] - 1
                return_rate_5d = df.loc[next_5d, 'close'] * df.loc[next_5d, 'adj_factor'] / (df.loc[current_date, 'close'] * df.loc[current_date, 'adj_factor']) - 1
                return_rate_1m = cal_return(current_date, 1, 0, df)
            
                return_last_1d = df.loc[last_trading_day, 'close'] / df.loc[last_trading_day, 'pre_close'] - 1
                return_last_5d = price / (df.loc[last_5d, 'close'] * df.loc[last_5d, 'adj_factor']) - 1
            
                if np.mean(np.abs(existing_data['close'][-3:] - existing_data['pre_close'][-3:])) == 0:
                    rsi_3 = 100
                else:
                    rsi_3 = 100 * np.mean(existing_data['adj_factor'][-3:] * np.maximum(existing_data['close'][-3:] - existing_data['pre_close'][-3:], 0)) / np.mean(existing_data['adj_factor'][-3:] * np.abs(existing_data['close'][-3:] - existing_data['pre_close'][-3:]))
                if rsi_3 > 70:
                    rsi_3_adj = 50 - rsi_3
                elif rsi_3 > 50:
                    rsi_3_adj = rsi_3 - 50
                elif rsi_3 > 30:
                    rsi_3_adj = 30 - rsi_3 
                else:
                    rsi_3_adj = 20 + rsi_3 
            
                if np.mean(np.abs(existing_data['close'][-14:] - existing_data['pre_close'][-14:])) == 0:
                    rsi_14 = 100
                else:
                    rsi_14 = 100 * np.mean(existing_data['adj_factor'][-14:] * np.maximum(existing_data['close'][-14:] - existing_data['pre_close'][-14:], 0)) / np.mean(existing_data['adj_factor'][-14:] * np.abs(existing_data['close'][-14:] - existing_data['pre_close'][-14:]))
                if rsi_14 > 70:
                    rsi_14_adj = 50 - rsi_14
                elif rsi_14 > 50:
                    rsi_14_adj = rsi_14 - 50
                elif rsi_14 > 30:
                    rsi_14_adj = 30 - rsi_14 
                else:
                    rsi_14_adj = 20 + rsi_14 
            
                rsi_28 = 100 * np.mean(existing_data['adj_factor'][-28:] * np.maximum(existing_data['close'][-28:] - existing_data['pre_close'][-28:], 0)) / np.mean(existing_data['adj_factor'][-28:] * np.abs(existing_data['close'][-28:] - existing_data['pre_close'][-28:]))
                if rsi_28 > 70:
                    rsi_28_adj = 50 - rsi_28
                elif rsi_28 > 50:
                    rsi_28_adj = rsi_28 - 50
                elif rsi_28 > 30:
                    rsi_28_adj = 30 - rsi_28 
                else:
                    rsi_28_adj = 20 + rsi_28 
            
                obv = np.sum(np.sign(last_month_df['close'] - last_month_df['pre_close']) * last_month_df['vol'])
            
                if last_month_df.shape[0] <= 10:
                    return_var_month_realized = np.nan
                    return_skew_month_realized = np.nan
                    return_kurt_month_realized = np.nan
                    avg_tr_last_month = np.nan
                    avg_tr_last_month_avg_tr_last_year = np.nan
                    return_var_month = np.nan
                    return_skew_month = np.nan
                    return_kurt_month = np.nan
                    return_d_var_month = np.nan
                    return_u_var_month = np.nan
                    return_d_var_var_month = np.nan
                    t_t_1 = np.nan
                    max_return_last_month = np.nan
                    corr_vol_close_month = np.nan
                    corr_vol_high_month = np.nan
                    corr_vol_open_month = np.nan
                    corr_vol_low_month = np.nan
                    high_open_month = np.nan
                    close_low_month = np.nan
                    trend_strength_month = np.nan
                else:
                    if np.isnan(last_month_df['turnover_rate_f']).all():
                        last_month_df.loc['turnover_rate_f'] = last_month_df['turnover_rate']
                    return_var_month_realized = np.nanmean(last_month_df['pct_chg'] ** 2)
                    return_skew_month_realized = np.nanmean(last_month_df['pct_chg'] ** 3)\
                        / (return_var_month_realized ** 1.5)
                    return_kurt_month_realized = np.nanmean(last_month_df['pct_chg'] ** 4)\
                        / (return_var_month_realized ** 2)
                    avg_tr_last_month = np.nanmean(last_month_df['turnover_rate_f'])
                    avg_tr_last_month_avg_tr_last_year = np.nanmean(
                        last_month_df['turnover_rate_f']) / np.nanmean(last_year_df['turnover_rate_f'])
                    return_var_month = last_month_df['pct_chg'].var()
                    return_skew_month = last_month_df['pct_chg'].skew()
                    return_kurt_month = last_month_df['pct_chg'].kurt()
                    return_d_var_month = d_var(last_month_df['pct_chg'])
                    return_u_var_month = u_var(last_month_df['pct_chg'])
                    return_d_var_var_month = return_d_var_month / return_var_month
                    t_t_1 = cal_return(current_date, 0, -1, df)
                    max_return_last_month = np.nanmax(last_month_df['pct_chg'])
                    corr_vol_close_month = corrcoef(last_month_df['vol'],
                                                    last_month_df['adj_factor'] * last_month_df['close'])
                    corr_vol_high_month = corrcoef(last_month_df['vol'],
                                                   last_month_df['adj_factor'] * last_month_df['high'])
                    corr_vol_open_month = corrcoef(last_month_df['vol'],
                                                   last_month_df['adj_factor'] * last_month_df['open'])
                    corr_vol_low_month = corrcoef(last_month_df['vol'],
                                                  last_month_df['adj_factor'] * last_month_df['low'])
                    high_open_month = np.nanmean(last_month_df['high'] / last_month_df['open'])
                    close_low_month = np.nanmean(last_month_df['close'] / last_month_df['low'])
                    trend_strength_month = (last_month_df['close'][-1] - last_month_df['pre_close'][0])\
                        / np.nansum(np.abs(last_month_df['close'] - last_month_df['pre_close']))
                if last_year_df.shape[0] <= 20:
                    return_var_year_realized = np.nan
                    return_skew_year_realized = np.nan
                    return_kurt_year_realized = np.nan
                    return_var_year = np.nan
                    return_skew_year = np.nan
                    return_kurt_year = np.nan
                    return_d_var_year = np.nan
                    return_u_var_year = np.nan
                    return_d_var_var_year = np.nan
                    std_tr_last_year = np.nan
                    avg_abs_return_tr_last_year = np.nan
                    close_last_year_high = np.nan
                    max_return_last_year = np.nan
                    corr_vol_close_year = np.nan
                    corr_vol_high_year = np.nan
                    corr_vol_open_year = np.nan
                    corr_vol_low_year = np.nan
                    high_open_year = np.nan
                    close_low_year = np.nan
                    trend_strength_year = np.nan
                    ma20_price = np.nan
                    ma20_ma5 = np.nan
                    SO_k = np.nan
                else:
                    if np.isnan(last_year_df['turnover_rate_f']).all():
                        last_year_df.loc['turnover_rate_f'] = last_year_df['turnover_rate']
                    return_var_year_realized = np.nanmean(last_year_df['pct_chg'] ** 2)
                    return_skew_year_realized = np.nanmean(last_year_df['pct_chg'] ** 3)\
                        / (return_var_year_realized ** 1.5)
                    return_kurt_year_realized = np.nanmean(last_year_df['pct_chg'] ** 4)\
                        / (return_var_year_realized ** 2)
                    return_var_year = last_year_df['pct_chg'].var()
                    return_skew_year = last_year_df['pct_chg'].skew()
                    return_kurt_year = last_year_df['pct_chg'].kurt()
                    return_d_var_year = d_var(last_year_df['pct_chg'])
                    return_u_var_year = u_var(last_year_df['pct_chg'])
                    return_d_var_var_year = return_d_var_year / return_var_year
                    std_tr_last_year = np.nanstd(last_year_df['turnover_rate_f'])
                    avg_abs_return_tr_last_year = np.nanmean(np.abs(last_year_df['pct_chg'])
                                                             / last_year_df['turnover_rate_f'])
                    close_last_year_high = df.loc[last_trading_day, 'close']\
                        * df.loc[last_trading_day, 'adj_factor']\
                        / np.nanmax(last_year_df['high'] * last_year_df['adj_factor'])
                    max_return_last_year = np.nanmax(last_year_df['pct_chg'])
                    corr_vol_close_year = corrcoef(last_year_df['vol'],
                                                   last_year_df['adj_factor'] * last_year_df['close'])
                    corr_vol_high_year = corrcoef(last_year_df['vol'],
                                                  last_year_df['adj_factor'] * last_year_df['high'])
                    corr_vol_open_year = corrcoef(last_year_df['vol'],
                                                  last_year_df['adj_factor'] * last_year_df['open'])
                    corr_vol_low_year = corrcoef(last_year_df['vol'],
                                                 last_year_df['adj_factor'] * last_year_df['low'])
                    high_open_year = np.nanmean(last_year_df['high'] / last_year_df['open'])
                    close_low_year = np.nanmean(last_year_df['close'] / last_year_df['low'])
                    trend_strength_year = (last_year_df['close'][-1] - last_year_df['pre_close'][0])\
                        / np.nansum(np.abs(last_year_df['close'] - last_year_df['pre_close']))
                    ma5_price = (np.nanmean(last_year_df['close'][-5:] * last_year_df['adj_factor'][-5:])
                                  - price)\
                        / price    
                    ma20_price = (np.nanmean(last_year_df['close'][-20:] * last_year_df['adj_factor'][-20:])
                                  - price)\
                        / price
                    ma20_ma5 = (np.nanmean(last_year_df['close'][-20:] * last_year_df['adj_factor'][-20:])
                                - np.nanmean(last_year_df['close'][-5:] * last_year_df['adj_factor'][-5:]))\
                        / price
                    SO_k = SO(last_year_df.iloc[-20:])
                    if last_year_df.shape[0] > 120:
                        ma120_price = (np.nanmean(last_year_df['close'][-120:] * last_year_df['adj_factor'][-120:])
                                       - price)\
                            / price
                        ma120_ma40 = (np.nanmean(last_year_df['close'][-120:] * last_year_df['adj_factor'][-120:])
                                      - np.nanmean(last_year_df['close'][-40:] * last_year_df['adj_factor'][-40:]))\
                            / price
                    else:
                        ma120_price = np.nan
                        ma120_ma40 = np.nan
                    if last_year_df.shape[0] > 60:
                        ma60_price = (np.nanmean(last_year_df['close'][-60:] * last_year_df['adj_factor'][-60:])
                                      - price)\
                            / price
                        ma60_ma20 = (np.nanmean(last_year_df['close'][-60:] * last_year_df['adj_factor'][-60:])
                                     - np.nanmean(last_year_df['close'][-20:] * last_year_df['adj_factor'][-20:]))\
                            / price
                        SO_d = np.nanmean([SO(last_year_df.iloc[-20:]),
                                           SO(last_year_df.iloc[-40:-20]),
                                           SO(last_year_df.iloc[-60:-20])])
                        SO_k_d = SO_k - SO_d
                    else:
                        ma60_price = np.nan
                        ma60_ma20 = np.nan
                        SO_d = np.nan
                        SO_k_d = np.nan
            
                info = {
                    'tick': stock,
                    'industry': industry_stock(stock, dict_industry),
                    'stock_value_cat': stock_to_cat(stock, current_date, dict_300, dict_500),
                    'date': current_date,
                    'return_rate_1d': return_rate_1d,
                    'return_rate_5d': return_rate_5d,
                    'return_rate_1m': return_rate_1m,
                    'close': price
                    # Trend factors
                    # Reversal
                    'return_last_1d': return_last_1d,
                    'return_last_5d': return_last_5d,
                    't_6_t_12': cal_return(current_date, -6, -12, df),
                    't_12_t_36': cal_return(current_date, -12, -36, df),
                    't_12_t_18': cal_return(current_date, -12, -18, df),
                    # Momentum
                    't_1_t_6': cal_return(current_date, -1, -6, df),
                    't_t_1': t_t_1,
                    'return_month': cal_hist_month(df, current_date),
                    'SO_k': SO_k,
                    'SO_d': SO_d,
                    'SO_k_d': SO_k_d,
                    'ma5_price': ma5_price,
                    'ma20_price': ma20_price,
                    'ma20_ma5': ma20_ma5,
                    'ma60_price': ma60_price,
                    'ma60_ma20': ma60_ma20,
                    'ma120_price': ma120_price,
                    'ma120_ma40': ma120_ma40,
                    # Liquidity
                    'std_tr_last_year': std_tr_last_year,
                    'avg_tr_last_month': avg_tr_last_month,
                    'avg_tr_last_month_avg_tr_last_year': avg_tr_last_month_avg_tr_last_year,
                    # Technical
                    'rsi_3': rsi_3,
                    'rsi_3_adj': rsi_3_adj,
                    'rsi_14': rsi_14,
                    'rsi_14_adj': rsi_14_adj,
                    'rsi_28': rsi_28,
                    'rsi_28_adj': rsi_28_adj,
                    'obv': obv,
                    'close_last_year_high': close_last_year_high,
                    'max_return_last_month': max_return_last_month,
                    'max_return_last_year': max_return_last_year,
                    'avg_abs_return_tr_last_year': avg_abs_return_tr_last_year,
                    'ln_mv_t': math.log(df.loc[last_trading_day, 'total_mv']),
                    'ln_mv_c': math.log(df.loc[last_trading_day, 'circ_mv']),
                    'mv_c_mv_t': df.loc[last_trading_day, 'circ_mv'] / df.loc[last_trading_day, 'total_mv'],
                    'return_var_month_realized': return_var_month_realized,
                    'return_skew_month_realized': return_skew_month_realized,
                    'return_kurt_month_realized': return_kurt_month_realized,
                    'return_var_month': return_var_month,
                    'return_skew_month': return_skew_month,
                    'return_kurt_month': return_kurt_month,
                    'return_d_var_month': return_d_var_month,
                    'return_u_var_month': return_u_var_month,
                    'return_d_var_var_month': return_d_var_var_month,
                    'return_var_year_realized': return_var_year_realized,
                    'return_skew_year_realized': return_skew_year_realized,
                    'return_kurt_year_realized': return_kurt_year_realized,
                    'return_var_year': return_var_year,
                    'return_skew_year': return_skew_year,
                    'return_kurt_year': return_kurt_year,
                    'return_d_var_year': return_d_var_year,
                    'return_u_var_year': return_u_var_year,
                    'return_d_var_var_year': return_d_var_var_year,
                    'corr_vol_close_month': corr_vol_close_month,
                    'corr_vol_high_month': corr_vol_high_month,
                    'corr_vol_open_month': corr_vol_open_month,
                    'corr_vol_low_month': corr_vol_low_month,
                    'high_open_month': high_open_month,
                    'close_low_month': close_low_month,
                    'trend_strength_month': trend_strength_month,
                    'corr_vol_close_year': corr_vol_close_year,
                    'corr_vol_high_year': corr_vol_high_year,
                    'corr_vol_open_year': corr_vol_open_year,
                    'corr_vol_low_year': corr_vol_low_year,
                    'high_open_year': high_open_year,
                    'close_low_year': close_low_year,
                    'trend_strength_year': trend_strength_year,      
            
                    # Value factors
                    'pe': df.loc[last_trading_day, 'pe_ttm'],
                    'ps': df.loc[last_trading_day, 'ps_ttm'],
                    'pb': df.loc[last_trading_day, 'pb'],
                    'current_ratio': df_f.loc[f_date, 'current_ratio'],
                    'quick_ratio': df_f.loc[f_date, 'quick_ratio'],
                    'cash_ratio': df_f.loc[f_date, 'cash_ratio'],
                    'inv_turn': df_f.loc[f_date, 'inv_turn'],
                    'ar_turn': df_f.loc[f_date, 'ar_turn'],
                    'ca_turn': df_f.loc[f_date, 'ca_turn'],
                    'fa_turn': df_f.loc[f_date, 'fa_turn'],
                    'assets_turn': df_f.loc[f_date, 'assets_turn'],
                    'fcfe': fcfe,
                    'tax_to_ebt': df_f.loc[f_date, 'tax_to_ebt'],
                    'ocf_to_or': df_f.loc[f_date, 'ocf_to_or'],
                    'ocf_to_opincome': df_f.loc[f_date, 'ocf_to_opincome'],
                    'ca_to_assets': df_f.loc[f_date, 'ca_to_assets'],
                    'tbassets_to_totalassets': df_f.loc[f_date, 'tbassets_to_totalassets'],
                    'int_to_talcap': df_f.loc[f_date, 'int_to_talcap'],
                    'currentdebt_to_debt': df_f.loc[f_date, 'currentdebt_to_debt'],
                    'longdeb_to_debt': df_f.loc[f_date, 'longdeb_to_debt'],
                    'ocf_to_shortdebt': df_f.loc[f_date, 'ocf_to_shortdebt'],
                    'debt_to_eqt': df_f.loc[f_date, 'debt_to_eqt'],
                    'tangibleasset_to_debt': df_f.loc[f_date, 'tangibleasset_to_debt'],
                    'tangasset_to_intdebt': df_f.loc[f_date, 'tangasset_to_intdebt'],
                    'tangibleasset_to_netdebt': df_f.loc[f_date, 'tangibleasset_to_netdebt'],
                    'ocf_to_debt': df_f.loc[f_date, 'ocf_to_debt'],
                    'ocf_to_interestdebt': df_f.loc[f_date, 'ocf_to_interestdebt'],
                    'longdebt_to_workingcapital': df_f.loc[f_date, 'longdebt_to_workingcapital'],
                    'ebitda_to_debt': df_f.loc[f_date, 'ebitda_to_debt'],
                    'cash_to_liqdebt': df_f.loc[f_date, 'cash_to_liqdebt'],
                    'cash_to_liqdebt_withinterest': df_f.loc[f_date, 'cash_to_liqdebt_withinterest'],
                    'q_netprofit_margin': df_f.loc[f_date, 'q_netprofit_margin'],
                    'q_gsprofit_margin': df_f.loc[f_date, 'q_gsprofit_margin'],
                    'q_exp_to_sales': df_f.loc[f_date, 'q_exp_to_sales'],
                    'q_profit_to_gr': df_f.loc[f_date, 'q_profit_to_gr'],
                    'q_saleexp_to_gr': df_f.loc[f_date, 'q_saleexp_to_gr'],
                    'q_adminexp_to_gr': df_f.loc[f_date, 'q_adminexp_to_gr'],
                    'q_finaexp_to_gr': df_f.loc[f_date, 'q_finaexp_to_gr'],
                    'q_impair_to_gr_ttm': df_f.loc[f_date, 'q_impair_to_gr_ttm'],
                    'q_gc_to_gr': df_f.loc[f_date, 'q_gc_to_gr'],
                    'q_op_to_gr': df_f.loc[f_date, 'q_op_to_gr'],
                    'q_roe': df_f.loc[f_date, 'q_roe'],
                    'q_dt_roe': df_f.loc[f_date, 'q_dt_roe'],
                    'q_npta': df_f.loc[f_date, 'q_npta'],
                    'q_opincome_to_ebt': df_f.loc[f_date, 'q_opincome_to_ebt'],
                    'q_investincome_to_ebt': df_f.loc[f_date, 'q_investincome_to_ebt'],
                    'q_dtprofit_to_profit': df_f.loc[f_date, 'q_dtprofit_to_profit'],
                    'q_salescash_to_or': df_f.loc[f_date, 'q_salescash_to_or'],
                    'q_ocf_to_sales': df_f.loc[f_date, 'q_ocf_to_sales'],
                    'q_ocf_to_or': df_f.loc[f_date, 'q_ocf_to_or'],
                    'ocf_yoy': df_f.loc[f_date, 'ocf_yoy'],
                    'roe_yoy': df_f.loc[f_date, 'roe_yoy'],
                    'q_gr_yoy': df_f.loc[f_date, 'q_gr_yoy'],
                    'q_sales_yoy': df_f.loc[f_date, 'q_sales_yoy'],
                    'q_op_yoy': df_f.loc[f_date, 'q_op_yoy'],
                    'q_profit_yoy': df_f.loc[f_date, 'q_profit_yoy'],
                    'q_netprofit_yoy': df_f.loc[f_date, 'q_netprofit_yoy'],
                    'equity_yoy': df_f.loc[f_date, 'equity_yoy'],
                    'rd_exp_to_earning': rd_exp_to_earning
                }
                data = data.append(info, ignore_index=True)
                current_date = next_date
            if data.shape[0] > 0:
                data.to_sql(
                    name=stock,
                    con=con,
                    if_exists='replace',
                    index=False
                    )
                con.commit()
        except Exception as e:
            print(stock)
            print(repr(e))
    cur.close()
    con.close()
    print(str(i)+" done")
    return None


def cal_return(date, delta_1, delta_2, df):
    # calculate the return from date + delta2 to date + delta1
    date_1 = change_month(date, delta_1)
    date_2 = change_month(date, delta_2)
    df_1 = df.index[df.index <= date_2]
    if df_1.shape[0] == 0:
        return np.nan
    stock_date_1 = df.index[df.index >= date_1][0]
    stock_date_2 = df.index[df.index >= date_2][0]
    return df.loc[stock_date_1, 'open'] * df.loc[stock_date_1, 'adj_factor']\
        / (df.loc[stock_date_2, 'open'] * df.loc[stock_date_2, 'adj_factor']) - 1


def change_month(date, delta):
    # Change the date given month
    month = int(date[-4:-2]) + delta
    year = int(date[:4])
    if month <= 0:
        if month % 12 == 0:
            year += math.floor(month / 12) - 1
            month = 12
        else:
            year += math.floor(month / 12)
            month = month % 12
    elif month > 12:
        year += math.ceil(month / 12) - 1
        month = month % 12
    if month < 10:
        return str(year) + '0' + str(month) + date[-2:]
    else:
        return str(year) + str(month) + date[-2:]


def cal_hist_month(df, date):
    return_rate = []
    while 1:
        ret = cal_return(date, -11, -12, df)
        if np.isnan(ret):
            break
        else:
            return_rate.append(ret)
        date = change_month(date, -12)
    if len(return_rate) == 0:
        return np.nan
    else:
        return np.nanmean(return_rate)


def d_var(df):
    return np.sum((df[df < 0] - np.mean(df)) ** 2 / (len(df) - 1))


def u_var(df):
    return np.sum((df[df > 0] - np.mean(df)) ** 2 / (len(df) - 1))


def corrcoef(df_1, df_2):
    return (np.nanmean(df_1 * df_2) - np.nanmean(df_1) * np.nanmean(df_2))\
           / (np.nanstd(df_1) * np.nanstd(df_2))


def industry_stock(stock, dict_industry):
    for industry in dict_industry:
        # print(dict_industry[industry]['con_code'])
        if stock in list(dict_industry[industry]['con_code']):
            return industry


def stock_to_cat(stock, date, dict_300, dict_500):
    if date < '20070101':
        return 'No index yet'
    if date[4:6] > '06':
        key = date[:4] + '0701'
    else:
        key = date[:4] + '0101'
    if stock in dict_300[key]:
        return 'hs300'
    elif stock in dict_500[key]:
        return 'zz500'
    else:
        return 'other'


def SO(df):
    return (df['close'][-1] * df['adj_factor'][-1] - np.nanmax(df['high'] * df['adj_factor']))\
        / (np.nanmax(df['high'] * df['adj_factor']) - np.nanmin(df['low'] * df['adj_factor']))


def main():
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


if __name__ == '__main__':
    main()

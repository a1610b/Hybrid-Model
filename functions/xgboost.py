# -*- coding: utf-8 -*-

import  xgboost as xgb
from Utils import  pathUtils
import pandas as pd
from sklearn.model_selection import  train_test_split
from Process.genBasicData import genData
from Utils import  feaUtils
import sys
sys.path.append('D:\Code\Hybrid Model')
import sqlite3 as db
import time
from multiprocessing import Pool
import pandas as pd
import tushare as ts
import numpy as np
import functions.get_data as get_data
import math


data = get_data.get_from_sql(stock_id = 'test_data_801010.SI', name = 'rf_data_industry')
train_data = genData(pathUtils.train_path)
test_data = genData(pathUtils.test_path)

param = {'max_depth': 3,
         'learning_rate ': 0.01,
         'silent': 1,
         'objective': 'binary:logistic',
         "eval_metric":"auc",
         "scale_pos_weight":10,
         "subsample":0.8,
         "min_child_weight":1,
         "n_estimators": 1}

# features = [i for i in list(train_data.columns) if i not in ["ID","y"]]
features = feaUtils.train_fea
x_train, x_valid, y_train, y_valid = train_test_split(train_data[features],train_data["y"],
                                                      test_size=0.2, random_state=66)

dtrain = xgb.DMatrix(x_train, y_train)
dvalid = xgb.DMatrix(x_valid, y_valid)
dtest  = xgb.DMatrix(test_data[features])


evallist = [(dtrain,"train"),(dvalid,"valid")]
num_round = 20000
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=30)


y_pre = bst.predict(dtest, ntree_limit = bst.best_ntree_limit)

res = pd.concat([test_data[["ID"]],pd.DataFrame(y_pre,columns=["pred"])],axis=1)
res.to_csv(pathUtils.predict_root_path+"3.csv",index=False)
————————————————
版权声明：本文为CSDN博主「很吵请安青争」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/dpengwang/java/article/details/86290505
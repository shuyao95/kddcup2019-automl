import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from automl import predict, train, validate,timetrain,timepredict
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer
from util import Config, log, show_dataframe, timeit
import random
import time

class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.targets = None

        self.Time_data_info={
            #time
            'total_time':0,
            'time_ramain_so_far':0,
            'time_for_feature_engineering':0,
            'time_for_hyperparams_searching':0,
            'time_for_model_train':0,
            'time_for_model_prediction':0,

            #size
            'feature_engineering_input_size':0,
            'data_rows_for_hp':0,
            'data_cols_for_hp':0,
            'test_data_rows':0,
            'test_data_columns':0,
            'For_safe':50,
        }
        self.randomintvalue = random.randint(1, 100)

    @timeit
    def fit(self, Xs, y, time_ramain):
        # self.tables = copy.deepcopy(Xs)
        self.Time_data_info['total_time'] = time_ramain
        self.Time_data_info['For_safe'] = (self.Time_data_info['total_time'] / 12)

        self.tables = Xs
        self.targets = y

    @timeit
    def predict(self, X_test, time_remain):
        self.Time_data_info['time_ramain_so_far'] = time_remain

        start_feature = time.time()

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]

        #index = main_table.sort_values(by=self.config['time_col']).index
        #split = int(0.6*len(index))
        #train_index, test_index = index[:split], index[split:]

        #log(f"Merge train and test tables...")
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        log(f"Feature engineering...")
        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        X = clean_df(X)
        X = feature_engineer(X, self.config)


        X_train = X[X.index.str.startswith("train")]
        X_train.index = X_train.index.map(lambda x: int(x.split('_')[1]))
        X_train.sort_index(inplace=True)
        y_train = self.targets

        end_feature = time.time()

        self.Time_data_info['time_for_feature_engineering'] = (end_feature - start_feature)

        self.Time_data_info['time_ramain_so_far'] = self.Time_data_info['time_ramain_so_far'] - self.Time_data_info[
            'time_for_feature_engineering']

        #self.Time_data_info['data_cols_for_hp'] = X.shape[1]
        #self.Time_data_info['data_rows_for_hp'] = X.shape[0]
        print(f"TIME info:", self.Time_data_info)

        # train model
        log(f"Training...")
        train_start = time.time()
        #train(X_train.iloc[train_index], y_train.iloc[train_index], self.config)

        timetrain(X_train, y_train, self.config,self.Time_data_info)

        #train with time limitation
        #timetrain(X_train.iloc[train_index], y_train.iloc[train_index], self.config, self.Time_data_info)

        train_end = time.time()

        self.Time_data_info['time_ramain_so_far'] = self.Time_data_info['time_ramain_so_far']-(train_end-train_start)
        self.Time_data_info['time_for_model_train'] = (train_end-train_start)

        print("TIME info:", self.Time_data_info)

        #r = predict(X_train.iloc[test_index], self.config)
        #r = timepredict(X_train.iloc[test_index], self.config)

        #print('Test auc: ', roc_auc_score(y_train.iloc[test_index], r))

        #importance = self.config["model"].feature_importance(importance_type='split')
        #feature_name = np.array(self.config["model"].feature_name())
        #feature_importance = pd.DataFrame({'feature_importance': feature_name[np.argsort(-importance)], 'importnace':-np.sort(-importance)})
        #feature_importance.to_csv('feature_importance.csv', index=False)

        # predict
        log(f"Predicting...")
        X_test = X[X.index.str.startswith("test")]
        X_test.index = X_test.index.map(lambda x: int(x.split('_')[1]))
        X_test.sort_index(inplace=True)
        result = predict(X_test, self.config)

        return pd.Series(result)

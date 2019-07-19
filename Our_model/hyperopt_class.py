import pandas as pd
from sklearn.model_selection import  train_test_split,cross_val_score,KFold
import lightgbm as lgb

from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe

from util import Config,log,timeit
from typing import Dict,List
from sklearn.metrics import roc_auc_score,auc

import time

import hyperopt
import numpy as np


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


class train_hyperopt:

    def __init__(self,Time_info):
        self.Time_info = Time_info

    @timeit
    def train_lightgbm(self,X: pd.DataFrame, y: pd.Series, config: Config,time_limitation):

        params = {
            'boosting_type':'gbdt',
            #'boosting_type': 'dart',
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4,
            'feature_fraction':0.9,
        }

        n_samples = int(0.1 * len(X))
        print('number of sample for hyperopt', n_samples)
        X_sample, y_sample = data_sample(X, y, n_samples)


        time_for_hp = (1 * time_limitation / 2)

        HYPEROPT_START=time.time()
        hyperparams = self.hyperopt_lightgbm(X_sample, y_sample, params, config,time_for_hp)
        HYPEROPT_end=time.time()

        print("time hyperopt:",HYPEROPT_end-HYPEROPT_START)


        time_for_train = time_limitation - (HYPEROPT_end - HYPEROPT_START)

        X_train, X_val, y_train, y_val = data_split(X, y, 0.1)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        # set it to a big value

        train_time_start = time.time()
        clf1= lgb.train({**params, **hyperparams},
                                    train_data,
                                    30,
                                    valid_data,
                                    early_stopping_rounds=30,
                                    verbose_eval=100)
        train_time_end = time.time()
        del clf1

        _30_boost_rounds_for_train_time = train_time_end - train_time_start

        leave_num_boost_rounds = int(0.9*(30*((time_for_train-_30_boost_rounds_for_train_time)/_30_boost_rounds_for_train_time)))

        print("leave_num_boost_rounds",leave_num_boost_rounds)

        #for test
        clf = lgb.train({**params, **hyperparams},
                                    train_set=train_data,
                                    num_boost_round=leave_num_boost_rounds,
                                    valid_sets=[train_data, valid_data],
                                    early_stopping_rounds=200,
                                    verbose_eval=100)



        config["model"]=clf

        #importance = config["model"].feature_importance(importance_type='split')
        #feature_name = config["model"].feature_name()
        #feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
        #feature_importance.to_csv('feature_importance.csv', index=False)

    @timeit
    def predict_lightgbm(self,X: pd.DataFrame, config: Config) -> List:

        return config["model"].predict(X)

    @timeit
    def hyperopt_lightgbm(self,X: pd.DataFrame, y: pd.Series, params: Dict, config: Config,time_limitation):

        time_start_dfl = time.time()

        X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        if params['boosting_type']=='dart':
            space = {
                "max_depth": hp.choice("max_depth", np.arange(2, 8, 1, dtype=int)),
                # smaller than 2^(max_depth)
                "num_leaves": hp.choice("num_leaves", np.arange(4, 400, 4, dtype=int)),
                "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.9, 0.1),
                "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.9, 0.1),
                "bagging_freq": hp.choice("bagging_freq", np.linspace(1,10,2, dtype=int)),
                # "scale_pos_weight":hp.uniform('scale_pos_weight',1.0,10.0),
                # "colsample_by_tree":hp.uniform("colsample_bytree",0.5,1.0),
                "min_child_weight": hp.quniform('min_child_weight', 2, 50, 2),
                "reg_alpha": hp.uniform("reg_alpha", 0.5, 5.0),
                "reg_lambda": hp.uniform("reg_lambda", 0.5, 5.0),
                "learning_rate": hp.quniform("learning_rate", 0.05, 0.2, 0.02),
                # "learning_rate": hp.loguniform("learning_rate", np.log(0.04), np.log(0.5)),
                #
                "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(200, 2000, 200, dtype=int)),
                "is_unbalance": hp.choice("is_unbalance", [True])
            }
        else:
            space = {
                "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                # smaller than 2^(max_depth)
                "num_leaves": hp.choice("num_leaves", np.arange(4, 160, 4, dtype=int)),
                # "feature_fraction": hp.quniform("feature_fraction", 0.8, 0.9, 0.05),
                # "bagging_fraction": hp.quniform("bagging_fraction", 0.2, 0.8, 0.1),
                # "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 10, 2, dtype=int)),
                # "scale_pos_weight": hp.uniform('scale_pos_weight',1.0, 10.0),
                "scale_pos_weight": hp.choice("scale_pos_weight", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                # "colsample_by_tree":hp.uniform("colsample_bytree",0.5,1.0),

                "min_child_weight": hp.uniform('min_child_weight', 1, 10),
                # "min_child_weight": hp.quniform('min_child_weight', 0.0002, 0.005, 0.0005),

                "reg_alpha": hp.uniform("reg_alpha", 0.0, 10.0),
                "reg_lambda": hp.uniform("reg_lambda", 0.0, 10.0),
                "learning_rate": hp.quniform("learning_rate", 0.01, 0.1, 0.01),
                # "learning_rate": hp.loguniform("learning_rate", np.log(0.04), np.log(0.5)),
                #
                "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(10, 2000, 10, dtype=int)),
                # "is_unbalance": hp.choice("is_unbalance", [True])
            }

            """
            space = {
                "max_depth": hp.choice("max_depth", np.arange(2, 11, 1, dtype=int)),
                # smaller than 2^(max_depth)
                #160
                "num_leaves": hp.choice("num_leaves", np.arange(8, 200, 8, dtype=int)),
                "feature_fraction": hp.quniform("feature_fraction", 0.6, 0.9, 0.05),
                # "bagging_fraction": hp.quniform("bagging_fraction", 0.2, 0.8, 0.1),
                # "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 10, 2, dtype=int)),
                # "scale_pos_weight":hp.uniform('scale_pos_weight',1.0, 10.0),
                # "colsample_by_tree":hp.uniform("colsample_bytree",0.5,1.0),

                "min_child_weight": hp.quniform('min_child_weight', 0.5, 80, 0.5),
                #"min_child_weight": hp.quniform('min_child_weight', 0.0002, 0.005, 0.0005),

                "reg_alpha": hp.uniform("reg_alpha", 3.0, 11.0),
                "reg_lambda": hp.uniform("reg_lambda", 3.0, 11.0),
                "learning_rate": hp.quniform("learning_rate", 0.02, 0.4, 0.01),
                # "learning_rate": hp.loguniform("learning_rate", np.log(0.04), np.log(0.5)),
                #
                "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(200, 2000, 80, dtype=int)),
                # "is_unbalance": hp.choice("is_unbalance", [True])
            }
            """

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=45, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]
            # in classification, less is better
            return {'loss': -score, 'status': STATUS_OK}


        trials = Trials()
        time_10evals_start = time.time()

        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=1,
                             rstate=np.random.RandomState(1))

        time_10evals_end = time.time()

        time_10_eval = (time_10evals_end - time_10evals_start)
        time_end_dfl = time.time()


        #400 = dim*50
        #avoid error
        #0.8*20=16

        #for test
        #400
        evals_num = min(int(8 * ((time_limitation - (time_end_dfl - time_start_dfl)) / time_10_eval)),1000)

        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=evals_num, verbose=1,
                             rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        del trials

        return hyperparams

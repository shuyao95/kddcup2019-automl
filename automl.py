from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from Our_model.hyperopt_class import train_hyperopt
from util import Config, log, timeit


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        # "scale_pos_weight": 5
    }

    n_samples = int(0.1*len(X))
    print('number of sample for hyperopt', n_samples)
    X_sample, y_sample = data_sample(X, y, n_samples)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    X_train, X_val, y_train, y_val = data_split(X, y, 0.1)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                1200,
                                [train_data, valid_data],
                                early_stopping_rounds=45,
                                verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "max_depth": hp.choice("max_depth", np.arange(2, 10, 1, dtype=int)),
        # smaller than 2^(max_depth)
        "num_leaves": hp.choice("num_leaves", np.arange(4, 200, 4, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.2, 0.8, 0.1),
        # "bagging_fraction": hp.quniform("bagging_fraction", 0.2, 0.8, 0.1),
        # "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 10, 2, dtype=int)),
        # "scale_pos_weight":hp.uniform('scale_pos_weight',1.0, 10.0),
        # "colsample_by_tree":hp.uniform("colsample_bytree",0.5,1.0),
        "min_child_weight": hp.quniform('min_child_weight', 2, 50, 2),
        "reg_alpha": hp.uniform("reg_alpha", 2.0, 8.0),
        "reg_lambda": hp.uniform("reg_lambda", 2.0, 8.0),
        "learning_rate": hp.quniform("learning_rate", 0.05, 0.4, 0.01),
        # "learning_rate": hp.loguniform("learning_rate", np.log(0.04), np.log(0.5)),
        #
        "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(200, 2000, 100, dtype=int)),
        #"is_unbalance": hp.choice("is_unbalance", [True])
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300,
                          valid_data, early_stopping_rounds=45, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=150, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


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


######################################with time limitation

@timeit
def timetrain(X: pd.DataFrame, y: pd.Series, config: Config,Time_info):
    #baseline
    #train_lightgbm(X, y, config)
    #leave for prediction

    time_limitation_for_hp= Time_info['time_ramain_so_far']-Time_info['For_safe']

    new_y = train_hyperopt(Time_info)

    new_y.train_lightgbm(X,y,config,time_limitation_for_hp)


@timeit
def timepredict(X: pd.DataFrame, config: Config) -> List:

    preds = predict_configmodel(X, config)

    return preds

@timeit
def predict_configmodel(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)

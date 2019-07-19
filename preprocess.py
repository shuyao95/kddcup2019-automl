import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool

import CONSTANT
from util import log, timeit


uni_ops = {
    CONSTANT.TIME_PREFIX: {
        'week': lambda df: df.dt.week,
        'year': lambda df: df.dt.year,
        'month': lambda df: df.dt.month,
        'day': lambda df: df.dt.day,
        'hour': lambda df: df.dt.hour,
        # 'minute': lambda df: df.dt.minute,
        'dayofweek': lambda df: df.dt.dayofweek,
        'dayofyear': lambda df: df.dt.dayofyear,
    },
}

@timeit
def compress_df(df, num=True, cat=True):
    if num:
        num_cols = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].astype('float32')
    if cat:
        cat_cols = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].astype('category')


@timeit
def parallelize_apply(func, df, cols):
   num_threads=4
   pool = Pool(processes=num_threads)
   col_num = int(np.ceil(len(cols) / num_threads))
   res1 = pool.apply_async(func, args=(df,cols[:col_num]))
   res2 = pool.apply_async(func, args=(df,cols[col_num:2 * col_num]))
   res3 = pool.apply_async(func, args=(df,cols[2 * col_num:3 * col_num]))
   res4 = pool.apply_async(func, args=(df,cols[3 * col_num:]))
   pool.close()
   pool.join()
   df = pd.concat([df,res1.get(),res2.get(),res3.get(),res4.get()],axis=1)
   return df


@timeit
def normal_apply(func, df, cols):
    return pd.concat([df, func(df, cols)], axis=1)


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        df = tables[tname]
        fillna(df)
        num_cols = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
        cat_cols = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        m_cat_cols = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
        time_cols = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

        print("num colums...")

        #if len(num_cols)>3:
        #    df = parallelize_apply(num_log_p,df,num_cols)
        #elif len(num_cols)>0:
        #    df = normal_apply(num_log_p,df,num_cols)

        print('Category columns...')
        # if len(cat_cols) > 3:
        #     df = parallelize_apply(count_cat, df, cat_cols)
        if len(cat_cols) > 0:
            df = normal_apply(count_cat, df, cat_cols)
        print('Multi category columns...')
        if len(m_cat_cols) > 3:
            df = parallelize_apply(count_m_cat, df, m_cat_cols)
        elif len(m_cat_cols) > 0:
            df = normal_apply(count_m_cat, df, m_cat_cols)
        print('Time columns...')
        # if len(time_cols) > 3:
        #     df = parallelize_apply(transform_datetime, df, time_cols)
        if len(time_cols) > 0:
            df = normal_apply(transform_datetime, df, time_cols)
        # drop columns
        df.drop(m_cat_cols+time_cols, axis=1, inplace=True)

        compress_df(df)
        tables[tname] = df
        # print(tname, '\n', df.info(memory_usage='deep'))

@timeit
def clean_df(df):
    compress_df(df, num=False)
    df_fillna_with_mean(df)
    hash_cat(df)
    return df


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)
    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def df_fillna_with_mean(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(df[c].mean(), inplace=True)
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        if "0" not in df[c].cat.categories:
            df[c] = df[c].cat.add_categories(["0"])
        df[c].fillna("0")
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        mean = pd.to_timedelta(df[c]).mean() + pd.Timestamp(0)
        df[c].fillna(mean, inplace=True)
    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


def num_log_p(df,num_cols):
    op = "log"
    eps = 1e-8
    prefix = CONSTANT.NUMERICAL_PREFIX
    new_df = pd.DataFrame()
    for c in num_cols:
        n_min = df[c].min()
        new_df[f"{prefix}{op.upper()}({c})"] = df[c].apply(lambda x: np.log(x-n_min+eps))
    return new_df

@timeit
def num_log(df):
    op = "log"
    eps = 1e-8
    prefix = CONSTANT.NUMERICAL_PREFIX
    num_cols = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    for c in num_cols:
        n_min = df[c].min()
        df[f"{prefix}{op.upper()}({c})"] = df[c].apply(lambda x: np.log(x-n_min+eps))


@timeit
def feature_engineer(df, config):
   # print(df.info(memory_usage='deep'))
   #num_log(df)
   return df


def count_cat(df, cat_cols):
    prefix_n = CONSTANT.NUMERICAL_PREFIX
    prefix_c = CONSTANT.CATEGORY_PREFIX
    op = "frequency"
    new_df=pd.DataFrame()
    for c in cat_cols:
        dic = df[c].value_counts().to_dict()
        new_df[f"{prefix_n}{op.upper()}({c})"] = df[c].apply(lambda x: dic[x])
    return new_df

def hash_cat(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

def frequent_cat(x):
    data = x.split(',')
    item, freq = np.unique(data, return_counts=True)
    return item[np.argmax(freq)]

def weighted_cat(dic):
    def freq(x):
        data = x.split(',')
        item, freq = np.unique(data, return_counts=True)
        global_freq = np.array([dic[i] for i in item])
        return item[np.argmax(global_freq*freq)]
    return freq

def count_m_cat(df,m_cat_cols):
    prefix_n = CONSTANT.NUMERICAL_PREFIX
    prefix_c = CONSTANT.CATEGORY_PREFIX
    op_l = 'length'
    op_f = 'frequent_cat'
    op_fw = 'frequent_weighted_cat'
    new_df=pd.DataFrame()
    for c in m_cat_cols:
        new_df[f"{prefix_c}{op_f.upper()}RANK(1)({c})"] = df[c].apply(frequent_cat)
        new_df[f"{prefix_n}{op_l.upper()}({c})"] = df[c].apply(lambda x: len(x.split(',')))
        all_item = ','.join(df[c].values).split(',')
        item, freq = np.unique(all_item, return_counts=True)
        dic = dict(zip(item, freq))
        new_df[f"{prefix_c}{op_fw.upper()}RANK(1)({c})"] = df[c].apply(weighted_cat(dic))
    return new_df


def count_m_cat_merge(df,m_cat_cols):
    prefix_c = CONSTANT.CATEGORY_PREFIX
    op_f = 'frequent_cat'
    new_df=pd.DataFrame()
    for c in m_cat_cols:
        new_df[f"{prefix_c}{op_f.upper()}RANK(1)({c})"] = df[c].apply(frequent_cat)
    return new_df


def transform_datetime(df, time_cols):
    prefix_n = CONSTANT.NUMERICAL_PREFIX
    ops = uni_ops[CONSTANT.TIME_PREFIX]
    new_dfs = []
    for c in time_cols:
        new_df = df[c].agg(ops.values())
        new_df.columns = [f"{prefix_n}{op.upper()}({c})" for op in ops]
        new_dfs += [new_df]
    return pd.concat(new_dfs, axis=1)


# @timeit
# def transform_categorical_hash(df):
#     for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
#         df[c] = df[c].apply(lambda x: int(x))

#     for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
#         df.drop(c, axis=1, inplace=True)


# # @timeit
# def rational_feature(df, col, windows=1):
#     if col.startswith(CONSTANT.NUMERICAL_PREFIX):
#         prefix = CONSTANT.NUMERICAL_PREFIX
#     elif col.startswith(CONSTANT.CATEGORY_PREFIX):
#         prefix = CONSTANT.CATEGORY_PREFIX
#     elif col.startswith(CONSTANT.MULTI_CAT_PREFIX):
#         prefix = CONSTANT.MULTI_CAT_PREFIX
#     else:
#         prefix = CONSTANT.TIME_PREFIX
#     data = df[col].values
#     new_data = [np.concatenate([np.repeat(data[0], i + 1), data[(i + 1):]]) for i in range(windows)]
#     new_data = np.stack(new_data, axis=1)
#     new_df = pd.DataFrame(new_data, columns=[f"{prefix}PREV_{i+1}({col})" for i in range(windows)]).astype(df[col].dtypes)
#     return pd.concat([df, new_df], axis=1)


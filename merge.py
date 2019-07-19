import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import CONSTANT
from util import Config, Timer, log, timeit

NUM_OP = [np.std, np.mean]

def join_name(a):
    if 'm_cat' in a[1]:
        prefix = CONSTANT.MULTI_CAT_PREFIX
    elif 'cat' in a[1]:
        prefix = CONSTANT.CATEGORY_PREFIX
    else:
        prefix = CONSTANT.NUMERICAL_PREFIX
    return f"{prefix}JOIN_{a[1].upper()}({a[0]})"


def bfs(root_name, graph, tconfig):
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, v_name, key, type_):
    if type_.split("_")[2] == 'many':
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(join_name)
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")
    return u.join(v, on=key)

# @timeit
# def join(u, v, v_name, key, type_):
# #    if isinstance(key, list):
# #        assert len(key) == 1
# #        key = key[0]
#    if type_.split("_")[2] == 'many':
#        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
#                     and not col.startswith(CONSTANT.TIME_PREFIX)
#                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
#        rehash_key = f'rehash_{key}'
#        v[rehash_key] = v[key].apply(lambda x: hash(x) % 200000)
#        v = v.groupby(rehash_key).agg(agg_funcs)
#        v.columns = v.columns.map(join_name)
#    else:
#        v = v.set_index(key)
#    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")
#    return u.join(v, on=key)


@timeit
def temporal_join(u, v, v_name, key, time_col):
    timer = Timer()

    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    timer.check("select")

    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    timer.check("concat")

    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    timer.check("rehash_key")

    tmp_u.sort_values(time_col, inplace=True)
    timer.check("sort")

    agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                 and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}

    tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    timer.check("group & rolling & agg")

    tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    timer.check("reset_index")

    tmp_u.columns = tmp_u.columns.map(lambda a:
        f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")

    if tmp_u.empty:
        log("empty tmp_u, return u")
        return u

    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    timer.check("final concat")

    del tmp_u

    return ret

def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            u = temporal_join(u, v, v_name, key, config['time_col'])
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)

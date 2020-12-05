"""Apply experiment with Sobol."""

import pandas as pd
import numpy as np
import SALib
from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import saltelli as sample_saltelli
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank

from settings import SOBOL_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
seed = 123
cache_file = f'{SOBOL_DATA_DIR}sobol_{seed}.json'

# calculate results with fixed parameters
x_all = sample_latin.sample(problem, 10000, seed=101)
y_true = evaluate(x_all, a)
y_true_ave = np.average(y_true)
x_default = 0.25
rand = np.random.randint(0, y_true.shape[0], size=(1000, y_true.shape[0]))
error_dict = {}; pool_res = {}

out_path = f'{SOBOL_DATA_DIR}{x_default}'
if not os.path.exists(out_path):
    os.makedirs(out_path)

file_exist = os.path.exists(cache_file)
dummy = True; error_cal = False

if not file_exist:
    partial_order, sa_total, sa_total_conf, sa_main, sa_main_conf = {}, pd.DataFrame(), \
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    n_start, n_end, n_step = 200, 5200, 200
    x_large_size = sample_saltelli.sample(problem, n_end, calc_second_order=False)
    
    for i in range(n_start, n_end, n_step):
        # partial ordering
        x_sobol = x_large_size[:i * (len_params + 2)]
        if i == n_start:
            y_sobol = evaluate(x_sobol, a)
        else:
            x_eval = evaluate(x_sobol[-(len_params + 2) * n_step:], a)
            x_sobol =  np.append(x_sobol, x_eval)
            y_sobol = evaluate(x_sobol, a)

        sa_sobol = analyze_sobol.analyze(problem, y_sobol, 
                            calc_second_order=False, num_resamples=1000, conf_level=0.95, dummy=dummy)
        # use toposort find parameter sa block
        conf_lower = sa_sobol['total_rank_ci'][0]
        conf_upper = sa_sobol['total_rank_ci'][1]
        if dummy: 
            len_dummy = len_params + 1
            abs_sort = partial_rank(len_dummy, conf_lower, conf_upper)
        else:
            abs_sort = partial_rank(len_params, conf_lower, conf_upper)

        rank_list = list(toposort(abs_sort))
        key = 'result_'+str(i)
        partial_order[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}
        # Save results
        sa_total.loc[:, key] = sa_sobol['ST']
        sa_total_conf.loc[:, key] = sa_sobol['ST_conf']
        sa_main.loc[:, key] = sa_sobol['S1']
        sa_main_conf.loc[:, key] = sa_sobol['S1_conf']

        if error_cal:
            error_dict[key], pool_res = group_fix(partial_order[key], evaluate, 
                            x_all, y_true, x_default, rand, pool_res, a, file_exist)

    sa_all = pd.concat([sa_total, sa_total_conf, sa_main, sa_main_conf], axis = 0)
    sa_all.loc[:, 'Type'] = [*['ST']*len_dummy, *['ST_conf'] * len_dummy, *['S1'] * len_dummy, *['S1_conf'] * len_dummy]
    sa_all.to_csv(SOBOL_DATA_DIR + 'Sobol_indices.csv')

    with open(cache_file, 'w') as fp:
        json.dump(partial_order, fp, indent=2)
        
else:
    with open(cache_file, 'r') as fp:
        partial_order = json.load(fp)

    if error_cal:
        for key, value in partial_order.items():
            error_dict[key], pool_res = group_fix(value, evaluate, x_all, y_true, 
                                        x_default, rand, pool_res, a, file_exist)

# convert the result into dataframe
if error_cal:
    key_outer = list(error_dict.keys())
    f_names = list(error_dict[key_outer[0]].keys())
    for ele in f_names:
        dict_measure = {key: error_dict[key][ele] for key in key_outer}
        df = to_df(partial_order, dict_measure)
        df.to_csv(f'{SOBOL_DATA_DIR}{x_default}/{ele}.csv')



            
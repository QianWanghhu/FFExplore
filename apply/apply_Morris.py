"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import morris as sample_morris
from SALib.analyze import morris as analyze_morris
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank

from settings import MORRIS_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = f'{MORRIS_DATA_DIR}morris_{8}.json'

# calculate results with fixed parameters
x_all = sample_latin.sample(problem, 10000)#, seed=101
y_true = evaluate(x_all, a)
y_true_ave = np.average(y_true)
rand = np.random.randint(0, y_true.shape[0], size=(1000, y_true.shape[0]))
defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
# defaults_list = [0.25]
defaults_list.sort()

for x_default in defaults_list:
    error_dict = {}
    pool_res = {}
    out_path = f'{MORRIS_DATA_DIR}{x_default}'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_exists = os.path.exists(cache_file)
    if not file_exists:
        # Loop of Morris
        partial_order = {}
        mu_st, sigma_dt = {}, {}
        rank_lower_dt, rank_upper_dt = {}, {}
        n_start, n_end, n_step = 10, 150, 10
        x_large_size = sample_morris.sample(problem, n_end, num_levels=8, seed=1010) #
        for i in range(n_start, n_end, n_step):
            # partial ordering
            x_morris = x_large_size[:i * (len_params + 1)]
            if i == n_start:
                y_morris = evaluate(x_morris, a)
            else:
                y_eval = evaluate(x_morris[-(len_params + 1) * n_step:], a)
                y_morris =  np.append(y_morris, y_eval)
            sa_dict = analyze_morris.analyze(problem, x_morris, y_morris, num_resamples=1000, conf_level=0.95, seed=123)
            mu_star_rank_dict = sa_dict['mu_star'].argsort().argsort()

            # use toposort to find parameter sa block
            conf_lower = sa_dict['mu_star_rank_conf'][0]
            conf_upper = sa_dict['mu_star_rank_conf'][1]

            abs_sort = partial_rank(len_params, conf_lower, conf_upper)
            
            rank_list = list(toposort(abs_sort))
            key = 'result_'+str(i)     
            partial_order[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}

            #save results returned from Morris if needed
            mu_st[key] = sa_dict['mu_star']
            rank_lower_dt[key] = conf_lower
            rank_upper_dt[key] = conf_upper
            sigma_dt[key] = sa_dict['sigma']

            # error_dict[key], pool_res = group_fix(partial_order[key], evaluate, 
            #                                     x_all, y_true, x_default, rand, 
            #                                     pool_res, a, file_exists)

        with open(cache_file, 'w') as fp:
            json.dump(partial_order, fp, indent=2)
    else:
        with open(cache_file, 'r') as fp:
            partial_order = json.load(fp)

        # for key, value in partial_order.items():
        #     error_dict[key], pool_res = group_fix(value, evaluate, x_all, y_true, 
        #                                         x_default, rand, pool_res, a, file_exists)
# End

    # # convert the result into dataframe
    # key_outer = list(error_dict.keys())
    # f_names = list(error_dict[key_outer[0]].keys())
    # for ele in f_names:
    #     dict_measure = {key: error_dict[key][ele] for key in key_outer}
    #     df = to_df(partial_order, dict_measure)
    #     df.to_csv(f'{MORRIS_DATA_DIR}{x_default}/{ele}.csv')

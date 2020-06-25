import pandas as pd
import numpy as np
import SALib
from toposort import toposort, toposort_flatten
import json

%load_ext autoreload
%autoreload 2

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G import g_func
from utils.Sobol_G_setting import  set_sobol_g_func

#sensitivity analysis and partial sorting 
from SALib.sample import morris as sample_morris
from SALib.sample import saltelli as sample_saltelli
from SALib.analyze import morris as analyze_morris
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

from saffix.utils.group_fix import group_fix
from saffix.utils.format_convert import to_df

import os

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
f_dir = '../../output/sobol_g/morris/symmetry/'
cache_file = '{}{}'.format(f_dir, 'morris_2.json')

if not os.path.exists(cache_file):
    # Loop of Morris
    partial_order = {}
    file_exist = False
    mu_st = {}
    for i in range(10, 310, 10):
        # partial ordering
        x_morris= sample_morris.sample(problem, i, num_levels=4)
        y_morris = evaluate(x_morris, a)
        sa_dict= analyze_morris.analyze(problem, x_morris, y_morris, num_resamples=1000, conf_level=0.95, seed=88)
        mu_star_rank_dict = sa_dict['mu_star'].argsort().argsort()
        # use toposort find parameter sa block
        conf_low = sa_dict['mu_star_rank_conf'][0]
        conf_up = sa_dict['mu_star_rank_conf'][1]
        rank_conf = {j:None for j in range(len_params)}
        for j in range(len_params):
            rank_conf[j] = [conf_low[j], conf_up[j]]  
        abs_sort= {j:None for j in range(len_params)}

        for m in range(len_params):
            list_temp = np.where(conf_low >= conf_up[m])
            set_temp = set()
            if len(list_temp) > 0:
                for ele in list_temp[0]:
                    set_temp.add(ele)
            abs_sort[m] = set_temp
        order_temp = list(toposort(abs_sort))

        partial_order['result_'+str(i)] = {j: list(order_temp[j]) for j in range(len(order_temp))}
#         #save results of mu_star
        mu_st['result_'+str(i)] = sa_dict['mu_star']         
        
    with open(cache_file, 'w') as fp:
        json.dump(partial_order, fp, indent=2)
else:
    file_exist = True
    with open(cache_file, 'r') as fp:
        partial_order = json.load(fp)


# calculate results with fixed parameters

# x_all = sample_latin.sample(problem, 10000, seed=101)
# y_true = g_func(x_all, a)
# y_true_ave = np.average(y_true)
# keys_list = list(partial_order.keys())[0:29]
# partial_order = {k:v for k, v in partial_order.items() if k in keys_list}
# # run more for default values evaluation
# # put all default values to test into a list
# defaults_list = np.append([0, 0.1, 0.4, 0.5], np.linspace(0.2, 0.3, 11))
# defaults_list.sort()
# # defaults_list = [0.75]
# for x_default in defaults_list: 
#     mean_fix = {key: None for key in keys_list}
#     var_fix = {key: None for key in keys_list}
#     mae_fix = {key: None for key in keys_list}
#     prsn_fix = {key: None for key in keys_list}
#     result_fix = {key: None for key in keys_list}
#     for key in keys_list:
#         mean_fix[key], mae_fix[key], var_fix[key], prsn_fix[key]= group_fix(partial_order[key], 
#         g_func, x_all, y_true, x_default, a, file_exist)

#     # convert the result into dataframe

#     dict_lists = [mean_fix, mae_fix, var_fix, prsn_fix]
#     f_names = ['mean', 'mae', 'var', 'pearsonr']
#     for ele in range(len(dict_lists)):
#         df = to_df(partial_order, dict_lists[ele], y_true_ave)
#         df.to_csv('{}{}{}{}{}{}'.format(f_dir, 'default_eval_0127/', str(round(x_default, 2)), '/', f_names[ele], '.csv'))
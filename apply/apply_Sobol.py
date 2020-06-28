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
from SALib.test_functions.Sobol_G import evaluate, \
    total_sensitivity_index as total_sa, sensitivity_index as main_sa,\
    partial_first_order_variance as pfoa, \
    total_variance
# import settings for Sobol G-function and returns necessary elements
%load_ext autoreload
%autoreload 2
from utils.Sobol_G_setting import set_sobol_g_func
from utils.group_fix import group_fix, to_df
a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

f_dir = '../../../Research/G_func_ff/output/sobol/revision/'
cache_file = '{}{}'.format(f_dir, 'sobol.json')

if not os.path.exists(cache_file):
    # Loop of Morris
    file_exist = False
    partial_order = {}
    sa_total = {}
    sa_main = {}
    for i in range(600, 1200, 200):
        # partial ordering
        x_sobol = sample_saltelli.sample(problem, i, calc_second_order=True)
        print(x_sobol.shape)
        y_sobol = g_func(x_sobol, a)
        sa_sobol = analyze_sobol.analyze(problem, 
                    y_sobol, calc_second_order=True, num_resamples=1000, conf_level=0.95)
        # mu_star_rank_dict = sa_dict['mu_star'].argsort().argsort()
        # use toposort find parameter sa block
        conf_low = sa_sobol['total_rank_ci'][0]
        conf_up = sa_sobol['total_rank_ci'][1]

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
#         #save results of sobol indices
        sa_total['result_'+str(i)] = sa_sobol['ST']         
        sa_main['result_'+str(i)] = sa_sobol['S1']


    with open(cache_file, 'w') as fp:
        json.dump(partial_order, fp, indent=2)
else:
    file_exist = True
    with open(cache_file, 'r') as fp:
        partial_order = json.load(fp)



# calculate results with fixed parameters
x_all = sample_latin.sample(problem, 10000, seed=101)
y_true = evaluate(x_all, a)
y_true_ave = np.average(y_true)
keys_list = list(partial_order.keys())[:12]
partial_order = {k:v for k, v in partial_order.items() if k in keys_list}
rand = np.random.randint(0, y_true.shape[0], size=(1000, y_true.shape[0]))
defaults_list = [0.25]
for x_default in defaults_list: 
    mae_conf, var_conf = {key: None for key in keys_list}, {key: None for key in keys_list}
    var_fix = {key: None for key in keys_list}
    mae_fix = {key: None for key in keys_list}
    prsn_fix = {key: None for key in keys_list}
    pearsonr_conf = {key: None for key in keys_list}
    for key in keys_list:
        mae_fix[key], var_fix[key], prsn_fix[key], mae_conf[key], var_conf[key], pearsonr_conf[key] \
        = group_fix(partial_order[key], evaluate, x_all, y_true, x_default, rand, a, file_exist)

    # convert the result into dataframe
    dict_lists = [mae_fix, var_fix, prsn_fix, mae_conf, var_conf, pearsonr_conf]
    f_names = ['mae', 'var', 'pearsonr', 'mae_conf', 'var_conf', 'pearsonr_conf']
    for ele in range(len(dict_lists)):
        df = to_df(partial_order, dict_lists[ele])
        df.to_csv(f'{f_dir}{f_names[ele]}.csv')
# End for()        
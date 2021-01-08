"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
import json
import os

#sensitivity analysis and partial sorting 
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.partial_sort import to_df
from utils.test_function_setting import set_sobol_g_func
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import MORRIS_DATA_DIR, SOBOL_DATA_DIR
a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = [f'../output/reuse_sample/morris/morris_8_123.json', f'../output/reuse_sample/sobol/sobol_123.json' ]
file_exists = True
pool_res = {}; x_default = 0.25

for fn in cache_file:
    # select which cache file to use
    with open(fn, 'r') as fp:
        partial_order = json.load(fp)

    if 'morris' in fn:
        out_path = f'../output/reuse_sample/morris/compare/{x_default}'
        key = list(partial_order.keys())[6]; partial_order = {key: partial_order[key]}
    else:
        key = list(partial_order.keys())[3]; partial_order = {key: partial_order[key]}
        out_path = f'../output/reuse_sample/sobol/compare/{x_default}'

    if not os.path.exists(out_path): os.makedirs(out_path)
    error_dict = {} 
    y_fix = np.array([])
    # get input samples
    file_sample = f'../output/reuse_sample/morris/metric_samples.csv'
    if os.path.exists(file_sample):
        y_true_exist = True
        samples = pd.read_csv(file_sample, index_col = 'Unnamed: 0').values
        x_all = samples[:, 0:-1]
        y_true = samples[:, -1]  
        x_subset = x_all[:100]
        y_subset = y_true[:100]

    for key, value in partial_order.items():
        num_group = len(value) - 1
        mae = {i: None for i in range(num_group)}
        var, ppmc = dict(mae), dict(mae)
        mae_upper, var_upper, ppmc_upper = dict(mae), dict(mae), dict(mae)
        mae_lower, var_lower, ppmc_lower = dict(mae), dict(mae), dict(mae)
        ind_fix = []
        # loop to fix parameters and calculate the error metrics
        for i in range(num_group, -1, -1):
            ind_fix = index_fix(value, i, file_exists, ind_fix)    
            
            skip_calcul = results_exist(ind_fix, pool_res)
            if skip_calcul == False:
                x_copy = np.copy(x_subset)
                x_copy[:, ind_fix] = [x_default]
                y_fix = evaluate_wrap(evaluate, x_copy, a)
                y_true_ave = np.average(y_subset)
                rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
                error_temp, pool_res, _ = group_fix(ind_fix, y_subset, y_fix, rand, pool_res, file_exists)
            
            else:
                # map index to calculated values
                error_temp = skip_calcul

            [mae[i], var[i], ppmc[i], mae_lower[i], var_lower[i], ppmc_lower[i], 
            mae_upper[i],var_upper[i], ppmc_upper[i]] = error_temp

        error_dict[key] = {'mae': mae, 'var': var, 'ppmc': ppmc,
                    'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                    'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}
        # # End for
        
    # convert the result into dataframe
    key_outer = list(error_dict.keys())
    f_names = list(error_dict[key_outer[0]].keys())
    for ele in f_names:
        dict_measure = {key: error_dict[key][ele] for key in key_outer}
        df = to_df(partial_order, dict_measure)
        df.to_csv(f'{out_path}/{ele}.csv')
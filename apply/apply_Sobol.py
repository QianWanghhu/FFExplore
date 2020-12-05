"""Apply experiment with Sobol."""

import pandas as pd
import numpy as np
from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import saltelli as sample_saltelli
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.partial_sort import to_df, partial_rank
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import SOBOL_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = f'../output/sobol/sobol_123.json'

seed = 123; dummy = False
file_exists = os.path.exists(cache_file)
if not file_exists:
    # Loop of Morris
    partial_order = {}
    sa_main, sa_main_conf = {}, {}
    sa_total, sa_total_conf = {}, {}
    n_start, n_end, n_step = 200, 1200, 200
    x_large_size = sample_saltelli.sample(problem, n_end, calc_second_order=False, seed=seed)
    for i in range(n_start, n_end, n_step):
        # partial ordering
        x_sobol = x_large_size[:i * (len_params + 2)]
        if i == n_start:
            y_sobol = evaluate(x_sobol, a)
        else:
            y_eval = evaluate(x_sobol[-(len_params + 2) * n_step:], a)
            # y_eval = evaluate(x_morris[-(len_params + 1) * n_step:], a)
            y_sobol =  np.append(y_sobol, y_eval)
        sa_sobol = analyze_sobol.analyze(problem, y_sobol, 
                        calc_second_order=False, num_resamples=1000, conf_level=0.95, dummy=dummy)

        # use toposort to find parameter sa block
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

        #save results returned from Morris if needed
        sa_total[key] = sa_sobol['ST']
        sa_total_conf[key] = sa_sobol['ST_conf']
        sa_main[key] = sa_sobol['S1']
        sa_main_conf[key] = sa_sobol['S1_conf']

    with open(cache_file, 'w') as fp:
        json.dump(partial_order, fp, indent=2)
else:
    with open(cache_file, 'r') as fp:
        partial_order = json.load(fp)

        
# End
# End
# defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
defaults_list = [0.25]
defaults_list.sort()

for x_default in defaults_list:
    out_path = f'../output/sobol/{x_default}'
    if not os.path.exists(out_path): os.makedirs(out_path)

    error_dict = {}; pool_res = {}
    y_fix = np.array([])
    # get input samples
    file_sample = f'../output/morris/metric_samples.csv'
    if os.path.exists(file_sample):
        y_true_exist = True
        samples = pd.read_csv(file_sample, index_col = 'Unnamed: 0').values
        x_all = samples[:, 0:-1]
        y_true = samples[:, -1]  
        x_subset = x_all[:10]
        y_subset = y_true[:10]

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
        fpath = f'../output/sobol/{x_default}'
        if not os.path.exists(fpath): os.makedirs(fpath)
        df.to_csv(f'{fpath}/{ele}.csv')

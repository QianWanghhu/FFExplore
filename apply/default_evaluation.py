"""Evaluate the effects of using different default values for factor fixng."""
import pandas as pd
import numpy as np
import os
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import MORRIS_DATA_DIR, METRIC_NAME

a, x, x_bounds, _, _, problem = set_sobol_g_func()

# calculate results with fixed parameters
defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
# defaults_list = [0.1, 0.2]
file_exist = False
nstart = 2400

# store results from fixing parameters in dict
error_dict = {}
combs_fix = [[i for i in range(15, 21)], [i for i in range(12, 21)]]
file_sample = f'../output/morris/metric_samples.csv'
y_true_exist = True
samples = pd.read_csv(file_sample, index_col = 'Unnamed: 0').values
x_all = samples[:, 0:-1]
y_true = samples[:, -1]  
x_subset = x_all[:2400]
y_subset = y_true[:2400]
defaults_list.sort()

for ind_fix in combs_fix:
    if os.path.exists(file_sample):
        num_group = len(defaults_list) - 1
        mae = {i: None for i in defaults_list}
        var, ppmc = dict(mae), dict(mae)
        mae_upper, var_upper, ppmc_upper = dict(mae), dict(mae), dict(mae)
        mae_lower, var_lower, ppmc_lower = dict(mae), dict(mae), dict(mae)
    for i in defaults_list:
        pool_res = {}
        y_fix = np.array([])
        # loop to fix parameters and calculate the error metrics            
        x_copy = np.copy(x_subset)
        x_copy[:, ind_fix] = [i]
        y_fix = evaluate_wrap(evaluate, x_copy, a)
        y_true_ave = np.average(y_subset)
        rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
        error_temp, pool_res, _ = group_fix(ind_fix, y_subset, y_fix, rand, {}, file_exist)
        
        [mae[i], var[i], ppmc[i], mae_lower[i], var_lower[i], ppmc_lower[i], 
        mae_upper[i],var_upper[i], ppmc_upper[i]] = error_temp

    error_dict[f'fix_{len(ind_fix)}'] = {'mae': mae, 'var': var, 'ppmc': ppmc,
                        'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                        'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}
            # # End for

# convert the result into dataframe
for key, value in error_dict.items():
    out_path = f'../output/morris/'
    if not os.path.exists(out_path): os.makedirs(out_path)
    df= pd.DataFrame(value)
    df.to_csv(f'{out_path}/{key}.csv')

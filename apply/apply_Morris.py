"""Apply experiment with Morris."""
import pandas as pd
import numpy as np
import json
import os
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func
from utils.partial_sort import to_df
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import MORRIS_DATA_DIR
from sa_rankings import morris_ranking

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

# sensitivity analysis with Morris
cache_file = f'../output/replicates/morris/morris_level4.json'
file_exists = os.path.exists(cache_file)
if file_exists:
    partial_order = morris_ranking(cache_file)
else:
    partial_order, x_morris = morris_ranking(cache_file)
    np.savetxt('../output/replicates/morris/' + 'morris_sample.txt', x_morris)

# Generate a number of r=10 independent Sobol' sampling sequences
from SALib.sample import sobol_sequence
N = 800
base_sequence = sobol_sequence.sample(N + 1000, (x.shape[0]) * 10)
base_sequence = base_sequence[1000:]
metric_cache = '../output/replicates/sobol/metric_samples.txt'
if not os.path.exists(metric_cache): np.savetxt(metric_cache, base_sequence)

# Start from a small sample size
# Define the factor set to fix and the sample size to use
x_fix_set = [[i for i in range(11, 21)], [i for i in range(15, 21)]]
nstart = 10; x_default = 0.25

out_path = f'../output/replicates/morris/sobol_adaptive/'
if not os.path.exists(out_path): os.makedirs(out_path)

# get input samples
if os.path.exists(metric_cache): samples = np.loadtxt(metric_cache)
r = int(samples.shape[1] / len_params) # the number of repetitions

# The loop of calculating error metrics 
for ind_fix in x_fix_set:
    error_dict = {}; pool_res = {}
    # loop to fix parameters and calculate the error metrics    
    for i in range(r-1):
        nsubsets = int(samples.shape[0] / 10)
        mae = {i: None for i in range(nsubsets)}
        var, ppmc = dict(mae), dict(mae)
        mae_upper, var_upper, ppmc_upper = dict(mae), dict(mae), dict(mae)
        mae_lower, var_lower, ppmc_lower = dict(mae), dict(mae), dict(mae)
        x_sample = samples[:, (i * len_params):(i + 1) * len_params]
        y_true = evaluate(x_sample, a)
        
        # Loop of each subset 
        for n in range(nsubsets):
            y_subset = y_true[0:(n + 1)*10]
            x_copy = np.copy(x_sample[0: (n + 1) * 10, :])
            x_copy[:, ind_fix] = [x_default]
            y_fix = evaluate_wrap(evaluate, x_copy, a)
            y_true_ave = np.average(y_subset)
            rand = np.array([np.arange(0, x_copy.shape[0])])
            error_temp, pool_res, _ = group_fix(ind_fix, y_subset, y_fix, rand, pool_res, file_exists)
    
            [mae[n], var[n], ppmc[n], mae_lower[n], var_lower[n], ppmc_lower[n], 
            mae_upper[n],var_upper[n], ppmc_upper[n]] = error_temp

    error_dict[f'replicate{r}'] = {'mae': mae, 'var': var, 'ppmc': ppmc,
                'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}
    # # End for

# convert the result into dataframe
# key_outer = list(error_dict.keys())
# f_names = list(error_dict[key_outer[0]].keys())
# for ele in f_names:
#     dict_measure = {key: error_dict[key][ele] for key in key_outer}
#     df = to_df(partial_order, dict_measure)
#     df.to_csv(f'{out_path}/{ele}.csv')

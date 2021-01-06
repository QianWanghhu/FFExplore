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
from utils.sample_replicates import sample_repli

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

# sensitivity analysis with Morris
outer_path = f'../output/replicates/morris/'
cache_file = f'{outer_path}morris_level4.json'
file_exists = os.path.exists(cache_file)
if file_exists:
    partial_order = morris_ranking(cache_file)
else:
    partial_order, x_morris = morris_ranking(cache_file)
    np.savetxt(outer_path + 'morris_sample.txt', x_morris)

# Define the factor set to fix and the sample size to use
x_fix_set = [[i for i in range(4, 21)], [i for i in range(8, 21)], 
    [i for i in range(11, 21)], [i for i in range(12, 21)], [i for i in range(15, 21)]]
x_default = 0.25

desti_folder = 'sobol_horizontal'
out_path = f'{outer_path}{desti_folder}/'
if not os.path.exists(out_path): os.makedirs(out_path)

# get input samples
metric_cache = f'{outer_path}{desti_folder}/metric_samples.txt'
r = 10 # r is the number of repetitions
if os.path.exists(metric_cache): 
    samples = np.loadtxt(metric_cache)
else:
    samples = sample_repli(800, len_params, metric_cache, split_style = 'vertical', 
        skip_numbers = 1000, num_replicates = r)

nsubsets = int(samples.shape[0] / 10); nstart = 10; nboot=1000
boot = True

# The loop of calculating error metrics 
for ind_fix in x_fix_set:
    print(ind_fix)
    error_dict = {}; pool_res = {}
    # loop to fix parameters and calculate the error metrics    
    for i in range(r):
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
            if boot:
                rand = np.random.randint(0, x_copy.shape[0], size = (nboot, x_copy.shape[0]))
            else:
                rand = np.array([np.arange(0, x_copy.shape[0])]) 
            error_temp, pool_res, _ = group_fix(ind_fix, y_subset, \
                y_fix, rand, pool_res, file_exists, boot)
    
            [mae[n], var[n], ppmc[n], mae_lower[n], var_lower[n], ppmc_lower[n], 
            mae_upper[n],var_upper[n], ppmc_upper[n]] = error_temp

        error_dict[f'replicate{i}'] = {'mae': mae, 'var': var, 'ppmc': ppmc,
                        'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                        'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}

    # convert the result into dataframe
    key_outer = list(error_dict.keys())
    f_names = list(error_dict[key_outer[0]].keys())
    len_fix = len(ind_fix)
    fpath = f'{out_path}/fix_{len_fix}/'
    if not os.path.exists(fpath): os.mkdir(fpath)
    for key in key_outer:
        # dict_measure = {key: error_dict[key][ele] for key in key_outer}
        df = pd.DataFrame.from_dict(error_dict[key], orient='columns')
        df.to_csv(f'{fpath}{key}.csv')
    # End for

# Calculate the standard error and mean
from utils.group_fix import stderr, std_stderr
out_path = f'{outer_path}{desti_folder}/'
fps = [fp for fp in os.listdir(out_path) if 'fix' in fp]
err_metrics = ['mae', 'var', 'ppmc']
col_names = [i + '_mean' for i in err_metrics]
for i in err_metrics: col_names.append(i + '_std')
for fp in fps:
    print(fp)
    filenames = [f for f in os.listdir(f'{out_path}{fp}/') if 'repl' in f]
    std_estimation = np.zeros(shape = (nsubsets, 3 * 2))
    mean_estimation = np.zeros(shape = (nsubsets, 3 * 2)) # the number of columns is double the number of error metrics used
    for i in range(3):
        seq = np.zeros(shape=(nsubsets, r))
        for jj in range(len(filenames)):
            fn = filenames[jj]
            df_temp = pd.read_csv(f'{out_path}{fp}/{fn}', index_col='Unnamed: 0')
            seq_temp = df_temp.loc[:, df_temp.columns[i]].values
            seq[:, jj] = seq_temp
        mean_estimation[:, i], mean_estimation[:, i + 3] = stderr(seq)
        std_estimation[:, i], std_estimation[:, i + 3] = std_stderr(seq)
    df_mean = pd.DataFrame(data=mean_estimation, columns=col_names)
    df_std = pd.DataFrame(data=std_estimation, columns=col_names)
    df_mean.to_csv(f'{out_path}{fp}/mean_estimation.csv')
    df_std.to_csv(f'{out_path}{fp}/std_estimation.csv')

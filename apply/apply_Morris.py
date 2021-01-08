"""Apply experiment with Morris."""
import pandas as pd
import numpy as np
import json
import os
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func
from utils.partial_sort import to_df
from utils.group_fix import group_fix, index_fix, results_exist, loop_error_metrics, stderr
from utils.group_fix import evaluate_wrap

from settings import METRIC_NAME, MORRIS_DATA_DIR
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

desti_folder = 'vertical'
out_path = f'{outer_path}{desti_folder}/'
if not os.path.exists(out_path): os.makedirs(out_path)

# Define the factor set to fix and the sample size to use
x_fix_set = [[i for i in range(4, 21)], [i for i in range(8, 21)], 
    [i for i in range(11, 21)], [i for i in range(12, 21)], [i for i in range(15, 21)]]
x_default = 0.25

# get input samples
metric_cache = f'{outer_path}{desti_folder}/metric_samples.txt'
r = 10 # r is the number of repetitions
if os.path.exists(metric_cache): 
    samples = np.loadtxt(metric_cache)
else:
    samples = sample_repli(800, len_params, metric_cache, split_style = 'vertical', 
        skip_numbers = 1000, num_replicates = r)

nsubsets = int(samples.shape[0] / 10); nstart = 9; nboot=1
boot = False

loop_error_metrics(out_path, x_fix_set, x_default, nsubsets, r, len_params, 
    samples, evaluate, boot, file_exists, a = a, nboot = nboot, save_file=True)

# Calculate the standard error and mean
out_path = f'{outer_path}{desti_folder}/'
err_metrics = ['mae', 'var', 'ppmc']
col_names = [i + '_mean' for i in err_metrics]
for i in err_metrics: col_names.append(i + '_std')
from utils.sample_replicates import replicates_process, boot_process
if 'boot' in desti_folder:
    boot_process(out_path, col_names, nsubsets, r, save_file=True)
else:
    replicates_process(out_path, col_names, nsubsets, r, save_file=True)

# Default evaluation
defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
defaults_list.sort()
samples  = samples[0:100, :]
nsubsets = int(samples.shape[0] / 10)
for x_default in defaults_list:
    out_path = f'{outer_path}{desti_folder}/{x_default}/'
    if not os.path.exists(out_path): os.makedirs(out_path)
    loop_error_metrics(out_path, x_fix_set[-2:], x_default, nsubsets, r, len_params, 
        samples, evaluate, boot, file_exists, a = a, nboot = nboot, save_file=True, nstart=9)

num_defaults = len(defaults_list)
num_fix = ['fix_6', 'fix_9']
df = pd.DataFrame(data = np.zeros(shape=(num_defaults, len(METRIC_NAME))), index=defaults_list,
    columns=METRIC_NAME)
for i in range(num_defaults):
    replicate_collect  = np.array([])
    out_path = f'{outer_path}sobol_vertical/{defaults_list[i]}/{num_fix[0]}/'
    fps = os.listdir(out_path)
    for fp in fps:
        df_temp = pd.read_csv(out_path + fp, index_col = 'Unnamed: 0').dropna(axis=0)
        try:
            replicate_collect = np.vstack([replicate_collect, df_temp.iloc[0].values])
        except ValueError:
            replicate_collect = df_temp.iloc[0].values

    for jj in range(3):
        df.loc[defaults_list[i], METRIC_NAME[jj]], std_temp =\
            stderr(replicate_collect[:, jj].reshape(r, 1).transpose())
        df.loc[defaults_list[i], METRIC_NAME[jj+3]], df.loc[defaults_list[i], METRIC_NAME[jj + 6]] =\
            df.loc[defaults_list[i], METRIC_NAME[jj]] - std_temp, df.loc[defaults_list[i], METRIC_NAME[jj]] + std_temp
df.to_csv(f'{outer_path}sobol_vertical/{num_fix[0]}.csv')
    
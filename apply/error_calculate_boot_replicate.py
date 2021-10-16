"""Apply experiment with Morris."""
import pandas as pd
import numpy as np
import json
import os
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func
from utils.group_fix import loop_error_metrics, stderr

from settings import METRIC_NAME, MORRIS_DATA_DIR
from utils.sample_replicates import sample_repli, return_metric_samples
from utils.sample_replicates import replicates_process

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

# sensitivity analysis with Morris
outer_path = f'output/morris/'
cache_file = f'{outer_path}morris.json'
file_exists = os.path.exists(cache_file)

# Define the factor set to fix and the sample size to use
x_fix_set = [[i for i in range(4, 21)], [i for i in range(9, 21)], 
    [i for i in range(11, 21)], [i for i in range(12, 21)], [i for i in range(15, 21)]]
x_default = 0.25

# calculation with bootstrap
r_range = [1]
for r in r_range:
    print(f'--------------------- bootstrap------------------------')
    desti_folder = ['bootstrap_mc_test']
    out_path  = [f'{outer_path}{f}/' for f in desti_folder]
    for f in out_path: 
        if not os.path.exists(f): os.makedirs(f)

    boot = True
    np.random.seed(111)
    samples_boot = np.random.rand(10000, len_params)
    np.savetxt(f'{out_path[0]}samples_mc.txt', samples_boot)
    nstep = int(r * 100)
    nsubsets = int(samples_boot.shape[0] / nstep); nboot=1000
    # loop_error_metrics(out_path[0], x_fix_set, x_default, nsubsets, 1, len_params, 
    #     samples_boot, evaluate, boot, file_exists, a = a, nboot = nboot, save_file=True, nstep = nstep)

# calculate error metrics with replicate sampling.
# get input samples
r_range = [*np.arange(10, 100, 10), *np.arange(100, 201, 100)]
for r in r_range:
    print(f'--------------------- Replicate = {r}------------------------')
    desti_folder = ['vertical']
    out_path  = [f'{outer_path}{f}_test/r{r}/' for f in desti_folder]
    for f in out_path: 
        if not os.path.exists(f): os.makedirs(f)
    
    split_style = desti_folder[0]; size = 1000; skip_numbers = 1000; 
    metric_cache = f'{out_path[0]}metric_samples.txt'
    samples_vertical = return_metric_samples(metric_cache, size, len_params, split_style, skip_numbers, r)
    # calculation with replicate sampling
    boot = False
    nstep = 10
    nsubsets = int(samples_vertical.shape[0] / 10); nstart = 0; nboot=1
    loop_error_metrics(out_path[0], x_fix_set, x_default, nsubsets, r, len_params, 
        samples_vertical, evaluate, boot, file_exists, a = a, nboot = nboot, save_file=True, nstep=nstep)

    #Calculate the standard error and mean
    for f in out_path[1:]:
        err_metrics = ['mae', 'var', 'ppmc']
        col_names = [i + '_mean' for i in err_metrics]
        for i in err_metrics: col_names.append(i + '_std')
        replicates_process(f, col_names, nsubsets, r, save_file=True)

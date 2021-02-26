"""Evaluate the effects of using different default values for factor fixng."""
import pandas as pd
import numpy as np
import os
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func
from utils.group_fix import loop_error_metrics
from settings import MORRIS_DATA_DIR, METRIC_NAME

a, x, x_bounds, _, len_params, problem = set_sobol_g_func()
# calculate results with fixed parameters
# Default evaluation
defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
defaults_list.sort()
out_path = 'output/adaptive_replicates/morris/bootstrap_mc/'
samples = np.loadtxt(out_path+'samples_mc.txt')[0:1000]
nsubsets = int(samples.shape[0] / 1000)
ind_fix_set = [[i for i in range(15, 21)], [i for i in range(12, 21)]]
boot=True
file_exists = False

for i in defaults_list:
    nstep = 1000
    nsubsets = int(samples.shape[0] / nstep); nboot=1000
    save_path = f'{out_path}{i}/'
    if not os.path.exists(save_path): os.mkdir(save_path)
    loop_error_metrics(save_path, ind_fix_set, i, nsubsets, 1, len_params, 
    samples, evaluate, boot, file_exists, a = a, nboot = nboot, save_file=True, nstep = nstep)
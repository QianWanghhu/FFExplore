"""Apply experiment for Genz function."""
import json
import os
import numpy as np
import pandas as pd
import time
from toposort import toposort, toposort_flatten
#sensitivity analysis and partial sorting 
import pyapprox as pya
from pyapprox.models import genz
from pyapprox.approximate import approximate, compute_l2_error
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices_from_gaussian_process,\
    sampling_based_sobol_indices
from pyapprox.benchmarks import sensitivity_benchmarks
from pyapprox.benchmarks.benchmarks import setup_benchmark
# import settings for Sobol G-function and returns necessary elements
from utils.group_fix import loop_error_metrics, group_fix
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks, gp_ranking
from utils.test_function_setting import *
from utils.sample_replicates import *

# set the Genz-function
conf_level = 0.95
file_path = f'../output/genz/'
if not os.path.exists(file_path): os.mkdir(file_path)
benchmark, num_nvars = set_genz()

# save parameter rankings, parameter sensitivity indices, and independent random samples
# error_list, samples, rank_groups =  gp_partial_ranking(benchmark, num_nvars, conf_level, file_path)
# np.savetxt(f'{file_path}error_gp.txt', error_list)
# np.savetxt(f'{file_path}samples_gp.txt', samples)
# with open(f'{file_path}rankings.json', 'w') as fp: json.dump(rank_groups, fp, indent=2)

###===========================###=====================================###
# TRAIN GAUSSIAN PROCESS
all_samples = np.loadtxt(f'{file_path}samples_gp.txt')
all_vals = benchmark.fun(all_samples)
approx_dict, error_dict = {}, {}
for n in range(470, 480, 10):
    print(f'---------------{n}---------------')
    approx_list, error_list = bootstrap_gp(all_samples[:, 0:n], all_vals[:, 0:n], \
        all_samples[:, 1000:], all_vals[1000:].flatten(), 10)
    error_dict[f'ntrain_{n}'] = error_list
    approx_dict[f'ntrain_{n}'] = approx_list
    if np.mean(error_list) <= 0.01: break

## CALCULATE ERRORS DUE TO FACTOR FIXING
file_exists = True
x_fix_set =[[k for k in range(i)] for i in range(1, num_nvars)]
x_default = 0.5
desti_folder = 'sobol_vertical'
r = 100 # r is the number of repetitions
metric_cache = f'{file_path}/metric_samples_large.txt'
if os.path.exists(metric_cache): 
    samples = np.loadtxt(metric_cache)
else:
    samples = sample_repli(1000, num_nvars, metric_cache, split_style = 'vertical', 
        skip_numbers = 1000, num_replicates = r)

num_interp = 100
rand = np.array([np.arange(samples.shape[0])])
stats_dict = {}
for num_fix in [0, 8, 9, 10, 11]:
    print(f'---------------FIX {num_fix}----------------')
    k = 0
    for ii in range(r):
        stats_ff = pd.DataFrame()
        x_uncond = samples[:, (ii*num_nvars):(ii+1)*num_nvars].T
        x_cond = np.copy(x_uncond)
        if num_fix > 0:
            x_cond[0:num_fix, :] = x_default

        for fun in approx_dict['ntrain_470']:
            mu_uncond, std_uncond = fun(x_uncond, return_std = True)
            # mu_uncond = benchmark.fun(x_uncond)
            mu_cond, std_cond = fun(x_cond, return_std = True)
            mu_uncond_rand = np.array([np.random.normal(mu_uncond[j], std_uncond[j], num_interp) \
                for j in range(len(mu_uncond))])
            mu_cond_rand = np.array([np.random.normal(mu_cond[j], std_cond[j], num_interp) \
                for j in range(len(mu_cond))])  
            for j in range(mu_cond_rand.shape[1]):
                stats_temp, _, _ = \
                    group_fix(x_fix_set[num_fix], mu_uncond_rand[:, j], mu_cond_rand[:, j],
                        rand, {}, file_exist=True, boot=False) 
                stats_ff[k] = stats_temp
                k += 1
        stats_ff.index = ['mae_mean', 'var_mean', 'ppmc_mean', 'mae_lower', 'var_lower', 'ppmc_lower', 
            'mae_upper', 'var_upper', 'ppmc_upper']
        fpath_temp = f'{file_path}fix_{num_fix}'
        if not os.path.exists(fpath_temp): os.mkdir(fpath_temp)
        stats_ff.T.to_csv(f'{fpath_temp}/replicate_{ii}.csv')
    # stats_dict[f'fix_{num_fix}'] = stats_ff

# Calculate the standard error and mean
out_path = f'{file_path}/gp/'
for fn in os.listdir(out_path):
    df = {}
    for ftemp in os.listdir(out_path+fn+'/'):
        df = pd.read_csv(out_path+fn+'/'+ftemp, index_col='Unnamed: 0')
        df.loc['mean',:] = df.mean(axis=0)
        df.loc['std', :] = df.std(axis=0)
        df.to_csv(out_path+fn+'/'+ftemp, index_label='Unnamed: 0')
        
err_metrics = ['mae', 'var', 'ppmc']
col_names = [i + '_mean' for i in err_metrics]

for i in err_metrics: col_names.append(i + '_std')
nsubsets = int(samples.shape[1] / num_nvars)
replicates_process(out_path, col_names, nsubsets, r, save_file=True)

##====================test performances of the reduced model================================##
train_samples = np.loadtxt(f'{file_path}samples_gp.txt')
train_vals = benchmark.fun(train_samples)
samples_fix = np.copy(train_samples)
samples_fix[-11:, :] = 0.5
error_list = {'reduced':[], 'full': []}
for ntrain in range(10*num_nvars, 1000+1, 50):
    print(ntrain)
    # full-model
    x_subset = train_samples[:, 0:ntrain]
    validation_samples = samples_fix[:, -300:]
    y_subset = benchmark.fun(x_subset)
    # full_approx = approximate(x_subset, y_subset, 'gaussian_process', {'nu':np.inf}).approx
    # reduced_model
    x_subset_fix = samples_fix[:, 0:ntrain]
    y_subset_fix = benchmark.fun(x_subset_fix)

    reduced_approx = approximate(x_subset_fix[:-9, :], y_subset_fix, 
        'gaussian_process', {'nu':np.inf}).approx
    # compare the errors
    validation_vals = benchmark.fun(validation_samples).flatten()            
    # full_error = l2_compute(full_approx, validation_samples, validation_vals)
    reduced_error = l2_compute(reduced_approx, validation_samples, validation_vals)
    error_list['reduced'].append(reduced_error)
    # error_list['full'].append(full_error)
df = pd.DataFrame.from_dict(error_list['reduced'])
df.index = np.arange(10*num_nvars, 1000+1, 50)
df.to_csv(file_path+'reduced_compare.csv')

## test with a larger sample size
def test_large_size():
    samples = pya.generate_independent_random_samples(
                    benchmark.variable, 2400, random_state=121)
    # train_samples = np.loadtxt(f'{file_path}samples_gp.txt')[:, 0:900]
    train_samples = samples[:, 0:240]
    validation_samples = samples[:, 240:]
    train_vals = benchmark.fun(train_samples)
    full_approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
    validation_vals = benchmark.fun(validation_samples).flatten()
    full_error = l2_compute(full_approx, validation_samples, validation_vals)
    return full_error

full_error = test_large_size()
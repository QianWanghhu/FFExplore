"""Apply experiment for Genz function."""

from toposort import toposort, toposort_flatten
import json
import os
import numpy as np
import pandas as pd
import time
#sensitivity analysis and partial sorting 
import pyapprox as pya
from pyapprox.models import genz
from pyapprox.approximate import approximate
from pyapprox.approximate import compute_l2_error
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices_from_gaussian_process
from pyapprox.benchmarks import sensitivity_benchmarks
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices
# import settings for Sobol G-function and returns necessary elements
from utils.group_fix import loop_error_metrics
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks
from utils.test_function_setting import set_genz, gaussian_process
from utils.sample_replicates import *

# call the function

def gp_sa(benchmark, num_nvars):
    nvars = benchmark.variable.num_vars()
    order = 2
    interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
    interaction_terms = interaction_terms[:, 
        np.where(interaction_terms.max(axis=0)==1)[0]]

    time_start = time.time()
    num_samples = 1500; nvalidation = 300
    nsamples = 100 # this is the sample used for Sobol' sensitivity analysis
    total_effects_dict, error_list, samples = gaussian_process(benchmark, interaction_terms, 
        num_samples, nvalidation, num_nvars, nsamples, 
            nstart=(10 * num_nvars), nstop=200, nstep=10)
    print(f'Use {time.time() - time_start} seconds')                
    return total_effects_dict, error_list, samples

def gp_ranking(total_effects_dict, conf_level):
    rank_groups = {}
    for key, value in total_effects_dict.items():
        try:
            ranking_ci[key] = compute_bootstrap_ranks(value.T, conf_level)
        except NameError:
            ranking_ci = {}; rank_groups = {}
            ranking_ci[key] = compute_bootstrap_ranks(value.T, conf_level)

        rank_list = partial_rank(ranking_ci[key][0], ranking_ci[key][1])
        rank_groups[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}
    return rank_groups

# Rank parameters according to the sensitivity indices
conf_level = 0.95
file_path = f'../output/genz/'
benchmark, num_nvars = set_genz()

total_effects_dict, error_list, samples = gp_sa(benchmark, num_nvars)
rank_groups = gp_ranking(total_effects_dict, conf_level)
# Save results    
for key, value in total_effects_dict.items():
    total_effects_df = pd.DataFrame(data = value, columns=[f'x{i+1}' for i in range(num_nvars)])
    total_effects_df.to_csv(f'{file_path}{key}_st.csv')

np.savetxt(f'{file_path}error_gp.txt', error_list)
np.savetxt(f'{file_path}samples_gp.txt', samples)

if not os.path.exists(file_path): os.mkdir(file_path)
# save parameter rankings, parameter sensitivity indices, and independent random samples
with open(f'{file_path}rankings.json', 'w') as fp: json.dump(rank_groups, fp, indent=2)

# Calculate the total effects of parameters in the Genz-function
###===========================###=====================================###
x_fix_set =[[k for k in range(i)] for i in range(1, num_nvars)]
desti_folder = 'sobol_vertical'
r = 10 # r is the number of repetitions

# full_approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
if os.path.exists(metric_cache): 
    samples = np.loadtxt(metric_cache)
else:
    samples = sample_repli(800, num_nvars, metric_cache, split_style = 'vertical', 
        skip_numbers = 1000, num_replicates = r)

x_default = 0.5; boot = False; nsubsets = int(samples.shape[0] / 10)
file_exists = True
loop_error_metrics(file_path, x_fix_set, x_default, nsubsets, r, num_nvars, 
    samples, benchmark.fun, boot, file_exists, save_file=True)

# Calculate the standard error and mean
out_path = f'{file_path}/'
err_metrics = ['mae', 'var', 'ppmc']
col_names = [i + '_mean' for i in err_metrics]
for i in err_metrics: col_names.append(i + '_std')
if 'boot' in desti_folder:
    boot_process(out_path, col_names, nsubsets, r, save_file=True)
else:
    replicates_process(out_path, col_names, nsubsets, r, save_file=True)

##====================test performances of the reduced model================================##
metric_cache = f'{file_path}/metric_samples.txt'
train_samples = np.loadtxt(f'{file_path}samples_gp.txt')
train_vals = benchmark.fun(train_samples)
samples_fix = np.copy(train_samples)
samples_fix[-10:, :] = 0.5
error_list = {'reduced':[], 'full': []}
for ntrain in range(10*num_nvars, 1000+1):
    # full-model
    x_subset = train_samples[:, 0:ntrain]
    validation_samples = samples_fix[:, -300:]
    y_subset = benchmark.fun(x_subset)
    full_approx = approximate(x_subset, y_subset, 'gaussian_process', {'nu':np.inf}).approx

    # reduced_model
    x_subset_fix = samples_fix[:, 0:ntrain]
    y_subset_fix = benchmark.fun(x_subset_fix)

    reduced_approx = approximate(x_subset_fix[:-9, :], y_subset_fix, 'gaussian_process', {'nu':np.inf}).approx

    # compare the errors
    validation_vals = benchmark.fun(validation_samples).flatten()
    full_approx_vals = full_approx(validation_samples).flatten()
            
    full_error = np.linalg.norm(full_approx_vals - validation_vals, axis=0) 
    full_error /= np.linalg.norm(validation_vals, axis=0)

    reduced_approx_vals = reduced_approx(validation_samples[:-9, :]).flatten()
            
    reduced_error = np.linalg.norm(reduced_approx_vals - validation_vals, axis=0) 
    reduced_error /= np.linalg.norm(validation_vals, axis=0)

    error_list['reduced'].append(reduced_error)
    error_list['full'].append(full_error)

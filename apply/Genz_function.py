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
from utils.group_fix import loop_error_metrics
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks, gp_ranking
from utils.test_function_setting import set_genz, gaussian_process, l2_compute, gp_sa
from utils.sample_replicates import *

# set the Genz-function
conf_level = 0.95
file_path = f'../output/genz/'
if not os.path.exists(file_path): os.mkdir(file_path)
benchmark, num_nvars = set_genz()

# Rank parameters according to the sensitivity indices
total_effects_dict, error_list, samples = gp_sa(benchmark, num_nvars, 
    num_samples=1500, nvalidation=300, nsamples=100, nstart=(10 * num_nvars),
        nstop=200, nstep=10)
rank_groups = gp_ranking(total_effects_dict, conf_level)
# Save results    
for key, value in total_effects_dict.items():
    total_effects_df = pd.DataFrame(data = value, columns=[f'x{i+1}' for i in range(num_nvars)])
    total_effects_df.to_csv(f'{file_path}{key}_st.csv')

# save parameter rankings, parameter sensitivity indices, and independent random samples
np.savetxt(f'{file_path}error_gp.txt', error_list)
np.savetxt(f'{file_path}samples_gp.txt', samples)
with open(f'{file_path}rankings.json', 'w') as fp: json.dump(rank_groups, fp, indent=2)

###===========================###=====================================###
x_fix_set =[[k for k in range(i)] for i in range(1, num_nvars)]
x_default = 0.5
desti_folder = 'sobol_vertical'
r = 10 # r is the number of repetitions
metric_cache = f'{file_path}/metric_samples_large.txt'
if os.path.exists(metric_cache): 
    samples = np.loadtxt(metric_cache)
else:
    samples = sample_repli(1000, num_nvars, metric_cache, split_style = 'vertical', 
        skip_numbers = 1000, num_replicates = r)

train_samples = np.loadtxt(f'{file_path}samples_gp.txt')[:, 0:900]
train_vals = benchmark.fun(train_samples)
full_approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
boot = False; nsubsets = int(samples.shape[0] / 10)
file_exists = True
loop_error_metrics(file_path, x_fix_set[8:10], x_default, nsubsets, r, num_nvars, 
    samples, full_approx, boot, file_exists, save_file=True)

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
    # reduced_error = l2_compute(full_approx, validation_samples, validation_vals)
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
    train_samples = samples[:, 0:400]
    validation_samples = samples[:, 400:]
    train_vals = benchmark.fun(train_samples)
    full_approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
    validation_vals = benchmark.fun(validation_samples).flatten()
    full_error = l2_compute(full_approx, validation_samples, validation_vals)
    return full_error

full_error = test_large_size()
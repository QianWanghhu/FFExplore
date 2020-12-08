"""Apply experiment for Oakley function."""

from toposort import toposort, toposort_flatten
import json
import os
import numpy as np
import pandas as pd

#sensitivity analysis and partial sorting 
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.approximate import approximate
from pyapprox.approximate import compute_l2_error
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices_from_gaussian_process
from pyapprox.benchmarks.sensitivity_benchmarks import oakley_function_statistics
# import settings for Sobol G-function and returns necessary elements
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks

# Obtain the analytic statistics of Oakley-function
mean, var, main_effect = oakley_function_statistics()

# Sensitivity analysis of Sobol with Gaussain Process
benchmark = setup_benchmark("oakley")
num_samples = 1000
samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples)
vals = benchmark.fun(samples)
ntrains = 250
train_samples = samples[:, 0:ntrains]
train_vals = vals[0:ntrains]
validation_samples = samples[:, ntrains:]
validation_vals = vals[ntrains:]

approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':.5}).approx
# nsamples = 100
# error = compute_l2_error(approx, benchmark.fun, benchmark.variable, nsamples, rel=True)
approx_vals = approx(validation_samples)
error = np.linalg.norm(approx_vals-validation_vals, axis=0) 
error /= np.linalg.norm(validation_vals, axis=0)

approx_vals.mean() / validation_vals.mean()

nvars = benchmark.variable.num_vars()
order = 2
interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
interaction_terms = interaction_terms[:, 
    np.where(interaction_terms.max(axis=0)==1)[0]]

nsamples = 800; ngp = 100
mean_sobol_indices, mean_total_effects, mean_variance, \
    std_sobol_indices, std_total_effects, std_variance, all_total_effects = \
        sampling_based_sobol_indices_from_gaussian_process(
            approx, benchmark.variable, interaction_terms, nsamples,
            sampling_method='sobol', ngp_realizations=ngp,
            normalize=True)

# compute partial ranks
conf_level = 0.60
ranking_ci = compute_bootstrap_ranks(all_total_effects.reshape(ngp, nvars).T,
            conf_level)
abs_sort = partial_rank(nvars, ranking_ci[0], ranking_ci[1])
rank_list = list(toposort(abs_sort))

# Test the effects of factor fixing using the original Oakley function
benchmark = setup_benchmark("oakley")
num_samples = 10000
samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples)
vals = benchmark.fun(samples)

x_default = np.array([-2.5, 0, 2.5])
# x_default = np.array([])
vals_fix = np.zeros((x_default.size, vals.size))
rv = np.zeros(x_default.size)
rmae = np.zeros(x_default.size)
for i in range(x_default.size):
    samples_fix = np.copy(samples)
    samples_fix[:10, :] = x_default[i]
    vals_fix = benchmark.fun(samples_fix)

    rv[i] = vals_fix.var()
    rmae[i] = np.mean(np.abs(vals_fix - vals), axis=0) / np.abs(vals).mean()

df = pd.DataFrame(data = np.array([vals_fix.flatten(), vals.flatten()]).T, columns = ['vals', 'vals_fix'])

df.to_csv('../output/oakley.csv')



def fix_factor_oakley(x_subset, y_subset, x_default, ind_fix):
    samples_fix = np.copy(x_subset)
    samples_fix[ind_fix, :] = x_default
    y_fix = benchmark.fun(samples_fix).flatten()
    rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
    error_dict, pool_res, y_fix = \
        group_fix(ind_fix, y_subset, y_fix, rand, {}, False)
    error_dict.append(y_fix.var())
    return error_dict

error_dict = {}    
for x_default in defaults:
    error_dict[str(x_default)] = fix_factor_oakley(samples, y_subset, x_default, ind_fix)


samples_fix = np.copy(samples)
samples_fix[ind_fix, :] = 0
y_fix = benchmark.fun(samples_fix).flatten()
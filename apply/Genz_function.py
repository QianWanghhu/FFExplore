"""Apply experiment for Genz function."""

from toposort import toposort, toposort_flatten
import json
import os
import numpy as np
import pandas as pd

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
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks

def set_genz():
    a = np.array([1e-7, 0.0001, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.5, 1, 2, 2.5, 2.5, 3])
    # a = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1])
    # a = np.array([0, 0.0001, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 1, 1.5, 2, 2, 2.5, 3])
    num_nvars = a.shape[0]
    u = np.random.rand(num_nvars)
    cw = np.array([a, u])
    benchmark = setup_benchmark('genz', nvars=num_nvars, test_name='oscillatory', coefficients = cw)
    return benchmark, num_nvars

# Calculate the total effects of parameters in the Genz-function
benchmark, num_nvars = set_genz()

nvars = benchmark.variable.num_vars()
order = 2
interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
interaction_terms = interaction_terms[:, 
    np.where(interaction_terms.max(axis=0)==1)[0]]
###===========================###=====================================###
# # Test the effects of factor fixing using the original Genz-function
# def fix_factor_genz(x_subset, y_subset, x_default, ind_fix):
#     """
#     Function used to fix factors in Genz-function. 
#     This is used for testing the effects of factor fixing.
#     """
#     samples_fix = np.copy(x_subset)
#     samples_fix[ind_fix, :] = x_default
#     y_fix = benchmark.fun(samples_fix).flatten()
#     rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
#     error_dict, _, y_fix = \
#         group_fix(ind_fix, y_subset, y_fix, rand, {}, False)
#     error_dict.append(y_fix.var())
#     return error_dict

# num_samples = 10000
# seeds = [np.random.randint(1, 1000) for ii in range(10)]

# ind_fixs =[[k for k in range(i)] for i in range(1, num_nvars)]
# error_dict = {}

# samples = pya.generate_independent_random_samples(
#             benchmark.variable, num_samples, random_state=seeds[0])
# vals = benchmark.fun(samples)

# defaults = np.array([0])

# y_true = vals.flatten()
# for i in range(len(ind_fixs)):
#     x_default = 0.5
#     ind_fix = ind_fixs[i]
#     error_dict[str(i)] = fix_factor_genz(samples, y_true, x_default, ind_fix)
# df = pd.DataFrame.from_dict(error_dict)

##=====================TEST GAUSSIAN PROCESS===================##
def gaussian_process():
    """
    This is used to generate the total-order Sobol' effects by multiple realizations of GP.
    """
    # Generate training and validating dataset
    num_samples = 1500
    rand = np.random.randint(0, 1000)
    samples = pya.generate_independent_random_samples(
                benchmark.variable, num_samples, random_state=rand)
    nvalidation = 300
    validation_samples = samples[:, -nvalidation:]
    validation_vals = benchmark.fun(validation_samples)
    error_list = []
    [nstart, nstop, nstep] = [(10 * num_nvars), 400, 10] #(10 * num_nvars)
    nsamples = 100 # this is the sample used for Sobol' sensitivity analysis
    total_effects_dict = {}
    
    for ntrains in range(nstart, (nstop+1), nstep):
        print(ntrains)
        train_samples = samples[:, 0:ntrains]
        if ntrains == nstart:
            vals_step = benchmark.fun(train_samples)
        else:
            vals_step = benchmark.fun(train_samples[:, -nstep:])

        try:
            train_vals = np.hstack((train_vals, vals_step))
        except NameError:
            train_vals = vals_step

        approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
        approx_vals = approx(validation_samples).flatten()
        
        error = np.linalg.norm(approx_vals - validation_vals, axis=0) 
        error /= np.linalg.norm(validation_vals, axis=0)
        error_list.append(error)

        mean_sobol_indices, mean_total_effects, mean_variance, \
                std_sobol_indices, std_total_effects, std_variance, all_total_effects = \
                    sampling_based_sobol_indices_from_gaussian_process(
                        approx, benchmark.variable, interaction_terms, nsamples,
                        sampling_method='sobol', ngp_realizations=100,
                        normalize=True)
        total_effects_dict[f'nsamples_{ntrains}'] = all_total_effects    


    return total_effects_dict, error_list, samples       

def rank_parameters(sa_matrix, conf_level):
    """
    The function returns rankings of parameters using the sensitivities from gaussian_process.
    """
    
    D, num_resamples = sa_matrix.shape
    rankings = np.zeros_like(sa_matrix)
    ranking_ci = np.zeros((D, 2))
    for resample in range(num_resamples):
	    rankings[:, resample] = np.argsort(sa_matrix[:, resample]).argsort()

    ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=1)

    return ranking_ci

# call the function
import time
time_start = time.time()
total_effects_dict, error_list, samples = gaussian_process()
print(f'Use {time.time() - time_start} seconds')

conf_level = 0.95
rank_groups = {}
for key, value in total_effects_dict.items():
    try:
        ranking_ci[key] = rank_parameters(value.T, conf_level)
    except NameError:
        ranking_ci = {}; rank_groups = {}
        ranking_ci[key] = rank_parameters(value.T, conf_level)

    rank_list = partial_rank(ranking_ci[key][0], ranking_ci[key][1])
    rank_groups[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}


file_path = f'../output/genz/'
if not os.path.exists(file_path): os.mkdir(file_path)

# save parameter rankings, parameter sensitivity indices, and independent random samples
with open(f'{file_path}rankings.json', 'w') as fp: json.dump(rank_groups, fp, indent=2)

for key, value in total_effects_dict.items():
    total_effects_df = pd.DataFrame(data = value, columns=[f'x{i+1}' for i in range(num_nvars)])
    total_effects_df.to_csv(f'{file_path}{key}_st.csv')

np.savetxt(f'{file_path}error_gp.txt', error_list)
np.savetxt(f'{file_path}samples_gp.txt', samples)


## test performances of the reduced model
# Fix parameters
samples_fix = np.copy(samples)
samples_fix[-10:, :] = 0.5

# full-model
train_samples = samples[:, 0:250]
validation_samples = samples_fix[:, -300:]
train_vals = benchmark.fun(train_samples)

full_approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx

# reduced_model
train_samples = samples_fix[:, 0:250]
train_vals = benchmark.fun(train_samples)

reduced_approx = approximate(train_samples[:-9, :], train_vals, 'gaussian_process', {'nu':np.inf}).approx

# compare the errors
validation_vals = benchmark.fun(validation_samples).flatten()
full_approx_vals = full_approx(validation_samples).flatten()
        
full_error = np.linalg.norm(full_approx_vals - validation_vals, axis=0) 
full_error /= np.linalg.norm(validation_vals, axis=0)


reduced_approx_vals = reduced_approx(validation_samples[:-9, :]).flatten()
        
reduced_error = np.linalg.norm(reduced_approx_vals - validation_vals, axis=0) 
reduced_error /= np.linalg.norm(validation_vals, axis=0)

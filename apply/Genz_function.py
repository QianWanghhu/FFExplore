"""Apply experiment for Oakley function."""

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
# import settings for Sobol G-function and returns necessary elements
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank, compute_bootstrap_ranks

num_nvars = 6
cw = np.array([[0.1, 0.1, 0.2, 0.3, 0.5, 1], np.random.rand(num_nvars)])
benchmark = setup_benchmark('genz', nvars=num_nvars, test_name='oscillatory', coefficients = cw)

# Calculate the total effects of parameters in the Genz-function
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices
nvars = benchmark.variable.num_vars()
order = 2
interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
interaction_terms = interaction_terms[:, 
    np.where(interaction_terms.max(axis=0)==1)[0]]
interaction_values, total_effect_values, variance =\
    sampling_based_sobol_indices(benchmark.fun, benchmark.variable, interaction_terms, 1000, sampling_method='sobol')

###===========================###=====================================###
# Test the effects of factor fixing using the original Genz-function
def fix_factor_genz(x_subset, y_subset, x_default, ind_fix):
    """
    Function used to fix factors in Genz-function. 
    This is used for testing the effects of factor fixing.
    """
    samples_fix = np.copy(x_subset)
    samples_fix[ind_fix, :] = x_default
    y_fix = benchmark.fun(samples_fix).flatten()
    rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
    error_dict, _, y_fix = \
        group_fix(ind_fix, y_subset, y_fix, rand, {}, False)
    error_dict.append(y_fix.var())
    return error_dict

num_samples = 10000
seeds = [np.random.randint(1, 1000) for ii in range(10)]

ind_fixs =[[k for k in range(i)] for i in range(1, num_nvars)]
error_dict = {}

for i in range(len(ind_fixs)):
    # rand = np.random.randint(0, 1000)
    samples = pya.generate_independent_random_samples(
                benchmark.variable, num_samples, random_state=seeds[0])
    vals = benchmark.fun(samples)

    defaults = np.array([0])
    
    y_true = vals.flatten()
    defaults = np.array([-2.5, 0, 2.5])   
    # for i in range(len(ind_fixs)):
    #     print(i)
    x_default = 0
    ind_fix = ind_fixs[i]
    error_dict[str(i)] = fix_factor_genz(samples, y_true, x_default, ind_fix)


df = pd.DataFrame.from_dict(error_dict)
from settings import METRIC_NAME
df.index = [*METRIC_NAME, 'var']

##=====================TEST GAUSSIN PROCESS===================##
num_samples = 10000
rand = np.random.randint(0, 1000)
samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples, random_state=121)
vals = benchmark.fun(samples)
validation_samples = samples[:, 1000:]
validation_vals = vals[1000:]
error_list = []

for ntrains in range(1100, 1601, 100):
    # ntrains = 250
    train_samples = samples[:, 0:ntrains]
    train_vals = vals[0:ntrains]
    approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
    nsamples = 300
    error = compute_l2_error(approx, benchmark.fun, benchmark.variable, nsamples, rel=True)
    # approx_vals = approx(validation_samples)
    # error = np.linalg.norm(approx_vals-validation_vals, axis=0) 
    # error /= np.linalg.norm(validation_vals, axis=0)
    error_list.append(error)

error_df = pd.DataFrame(data = error_list, columns = ['nu_1.5'],
        index =[str(i) for i in range(1000, 1601,100)])
error_df.loc[:, 'nu_2.5'] = error_list
error_df.loc[:, 'nu_inf'] = error_list  
error_df.to_csv('../output/error_gp.csv')


"""Apply experiment for Oakley function."""

from toposort import toposort, toposort_flatten
import json
import os
import numpy as np

#sensitivity analysis and partial sorting 
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.approximate import approximate
from pyapprox.approximate import compute_l2_error
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices_from_gaussian_process

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import add_dummy
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank

benchmark = setup_benchmark("oakley")
num_samples = 300
train_samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples)

train_vals = benchmark.fun(train_samples)
approx = approximate(
            train_samples, train_vals, 'gaussian_process', {'nu':1.5}).approx

nsamples = 100
error = compute_l2_error(
    approx, benchmark.fun, benchmark.variable,
    nsamples, rel=True)
print(error)
nvars = benchmark.variable.num_vars()
order = 2
interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
interaction_terms = interaction_terms[:, 
    np.where(interaction_terms.max(axis=0)==1)[0]]

mean_sobol_indices, mean_total_effects, mean_variance, \
    std_sobol_indices, std_total_effects, std_variance = \
        sampling_based_sobol_indices_from_gaussian_process(
            approx, benchmark.variable, interaction_terms, nsamples,
            sampling_method='sobol', ngp_realizations=10,
            normalize=True)
"""Apply experiment for Genz Oscillatory function."""

import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.models import genz
from pyapprox.approximate import approximate
from pyapprox.approximate import compute_l2_error
from pyapprox.benchmarks.benchmarks import setup_benchmark
# import settings for Sobol G-function and returns necessary elements

def set_genz():
    """
    Set the coefficients in Genz Oscillatory function.
    """
    a = np.array([1e-7, 0.0001, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.5, 1, 2, 2.5, 2.5, 3])
    num_nvars = a.shape[0]
    u = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9]
    cw = np.array([a, u])
    benchmark = setup_benchmark('genz', nvars=num_nvars, test_name='oscillatory', coefficients = cw)
    return benchmark, num_nvars

def gp_fit(num_samples, nvalidation, train_samples, train_vals):
    approx = approximate(samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
    error = compute_l2_error(
            benchmark.fun, approx, benchmark.variable,
            nvalidation, rel=True)
    return approx, error

benchmark, num_nvars = set_genz()
nvars = benchmark.variable.num_vars()
order = 2
interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
interaction_terms = interaction_terms[:, 
    np.where(interaction_terms.max(axis=0)==1)[0]]

##=====================TEST GAUSSIAN PROCESS===================##
num_samples = 500
num_realizations = 10
rand = np.random.randint(0, num_samples, size=(num_realizations, num_samples))
samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples)
vals = benchmark.fun(samples)
nvalidation = 400

# generate multiple realizations of GP 
approx_list = []; l2_errors = np.zeros(num_realizations)
for ii in range(num_realizations):
    approx_temp, l2_errors[ii] = gp_fit(num_samples, nvalidation, 
        samples[:, rand[ii]], vals[rand[ii]])
    approx_list.append(approx_temp)

# Generate x samples to calculate the error caused by fixing parameters
x_uncond = pya.generate_independent_random_samples(
                benchmark.variable, 1000)
mu_genz = benchmark.fun(x_uncond)

num_interp = 100 # number of values to sample from the uncertain range of each return from GP.
rmae = np.zeros(num_interp*num_realizations)
k=0
for approx in approx_list:
    mu_cond, std_cond = approx(x_uncond, return_std = True)
    # Generate multiple values from the uncertain range of each return from GP.
    mu_cond_rand = np.array([np.random.normal(mu_cond[j], std_cond[j], num_interp) \
            for j in range(mu_cond.shape[0])])

    for j in range(num_interp):
        rmae[k] = (np.abs((mu_cond_rand[:, j] - mu_genz.flatten()) / mu_genz.flatten())).mean(axis=0)
        k += 1
print('l2_error:', l2_errors)
np.savetxt('rmae.txt', rmae)
import numpy as np
from pyapprox.benchmarks.benchmarks import setup_benchmark
import pyapprox as pya
from pyapprox.models import genz
from pyapprox.approximate import approximate
from pyapprox.approximate import compute_l2_error
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices_from_gaussian_process
from pyapprox.benchmarks import sensitivity_benchmarks
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.sensitivity_analysis import sampling_based_sobol_indices
import time

# define efficients a and x variables according to Sheikholeslami (2019)
# start
def set_sobol_g_func():
    """
    Set up Sobol G-function for experiment.

    Returns
    ----------
    a : np.array,
        Coefficients of Sobol G-function

    x : np.array,
        input variables

    len_params : int,
        number of parameters

    problem : dict,
        SALib problem spec (for sensitivity analysis)
    """
    a = np.zeros(21)
    x = np.zeros(21) 

    a[0:2] = 0
    a[2:4] = [0.005, 0.090]
    a[4:7] = 2
    a[7:11] = [2.10, 2.75, 3, 3.15]
    a[11:15] = [8, 13, 13.5, 16]
    a[15:] = [70, 75, 80, 85, 90, 99]

    x_names = ['x' + str(i+1) for i in range(21)]
    len_params = len(x_names)
    x_bounds = np.zeros((21, 2))
    x_bounds[:, 0] = 0
    x_bounds[:, 1] = 1

    problem = {
        'num_vars': len(x),
        'names': x_names,
        'bounds': x_bounds
    }

    return a, x, x_bounds, x_names, len_params, problem
# End

def add_dummy(bool_add, problem):
    if bool_add:
        problem['num_vars'] = problem['num_vars'] + 1
        len_params = problem['num_vars']
        problem['names'].append('dummy')
        problem['bounds'] = np.append(problem['bounds'], [[0, 1]], axis=0)
        return problem, len_params

def set_genz():
    a = np.array([1e-7, 0.0001, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.5, 1, 2, 2.5, 2.5, 3])
    # a = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1])
    # a = np.array([0, 0.0001, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 1, 1.5, 2, 2, 2.5, 3])
    num_nvars = a.shape[0]
    u = np.random.rand(num_nvars)
    cw = np.array([a, u])
    benchmark = setup_benchmark('genz', nvars=num_nvars, test_name='oscillatory', coefficients = cw)
    return benchmark, num_nvars

##=====================TEST GAUSSIAN PROCESS===================##
def gaussian_process(benchmark, interaction_terms, num_samples, 
    nvalidation, num_nvars, nsamples, **kwarg):
    """
    This is used to generate the total-order Sobol' effects by multiple realizations of GP.
    """
    # Generate training and validating dataset
    rand = np.random.randint(0, 1000)
    samples = pya.generate_independent_random_samples(
                benchmark.variable, num_samples, random_state=rand)
    validation_samples = samples[:, -nvalidation:]
    validation_vals = benchmark.fun(validation_samples)
    error_list = []
    nstart, nstop, nstep = kwarg['nstart'], kwarg['nstop'], kwarg['nstep']
    total_effects_dict = {}
    
    for ntrains in range(nstart, (nstop+1), nstep):
        print(ntrains)
        train_samples = samples[:, 0:ntrains]
        if ntrains == nstart:
            vals_step = benchmark.fun(train_samples)
        else:
            vals_step = benchmark.fun(train_samples[:, -nstep:])

        try:
            # import pdb; pdb.set_trace()
            train_vals = np.append(train_vals, vals_step, axis=0)
        except NameError:
            train_vals = vals_step

        approx = approximate(train_samples, train_vals, 'gaussian_process', {'nu':np.inf}).approx
        approx_vals = approx(validation_samples).flatten()
        
        error = np.linalg.norm(approx_vals.flatten() - validation_vals.flatten()) 
        error /= np.linalg.norm(validation_vals)
        error_list.append(error)

        mean_sobol_indices, mean_total_effects, mean_variance, \
                std_sobol_indices, std_total_effects, std_variance, all_total_effects = \
                    sampling_based_sobol_indices_from_gaussian_process(
                        approx, benchmark.variable, interaction_terms, nsamples,
                        sampling_method='sobol', ngp_realizations=100,
                        normalize=True)
        total_effects_dict[f'nsamples_{ntrains}'] = all_total_effects    

    return total_effects_dict, error_list, samples       

def l2_compute(gp, validation_samples, validation_vals):
    full_approx_vals = gp(validation_samples).flatten()
    full_error = np.linalg.norm(full_approx_vals - validation_vals, axis=0) 
    full_error /= np.linalg.norm(validation_vals, axis=0)
    return full_error

def gp_sa(benchmark, num_nvars, num_samples, nvalidation, nsamples, **kwargs):
    """
    nsamples: int, sample size used for Sobol' analysis
    """
    nvars = benchmark.variable.num_vars()
    nstart, nstop, nstep = kwargs['nstart'], kwargs['stop'], kwargs['nstep']
    order = 2
    interaction_terms = pya.compute_hyperbolic_indices(nvars, order)
    interaction_terms = interaction_terms[:, 
        np.where(interaction_terms.max(axis=0)==1)[0]]

    time_start = time.time()
    total_effects_dict, error_list, samples = gaussian_process(benchmark, interaction_terms, 
        num_samples, nvalidation, num_nvars, nsamples, 
            nstart, nstop, nstep)
    print(f'Use {time.time() - time_start} seconds')                
    return total_effects_dict, error_list, samples
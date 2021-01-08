"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
import os
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import MORRIS_DATA_DIR, METRIC_NAME, SOBOL_DATA_DIR

a, x, x_bounds, _, _, problem = set_sobol_g_func()

# calculate results with fixed parameters
x_default = 0.25
file_exist = True

# store results from fixing parameters in dict
error_dict = {}
combs_fix = [[20], [i for i in range(15, 21)], [i for i in range(12, 21)], 
    [i for i in range(11, 21)], [*[i for i in range(11, 21)], 8], [i for i in range(4, 21)], [i for i in range(21)]]
pool_res = {}; mse = {}; y_fix = np.array([])

ind_fix = combs_fix[1]
file_sample = f'{MORRIS_DATA_DIR}/metric_samples.csv'
if os.path.exists(file_sample):
    y_true_exist = True
    samples = pd.read_csv(file_sample, index_col = 'Unnamed: 0').values
    x_all = samples[:, 0:-1]
    y_true = samples[:, -1]
    nstart, nstop, nstep = 10, y_true.size, 10
    print('=======READ INPUT SAMPLES========')
else:
    y_true_exist = False
    nstart, nstop, nstep = 10, 5000, 10
    bounds = problem['bounds']
    min_bnds = [lb[0] for lb in bounds]
    max_bnds = [lb[1] for lb in bounds]
    # x_all = np.random.uniform(min_bnds, max_bnds, size=(nstop, problem['num_vars']))
    x_all = sample_latin.sample(problem, nstop, seed=101)
    print('=======GENERATE INPUT SAMPLES========')

for n in range(nstart, nstop + 1, nstep):
    # print(n)
    x_subset = x_all[:n]
    if y_true_exist == True:
        y_subset = y_true[:n]
    else:
        try:
            y_subset = np.append(y_subset, evaluate_wrap(evaluate, x_subset[-nstep:], a))
        except NameError:
            y_subset = evaluate_wrap(evaluate, x_subset, a)

    skip_calcul = results_exist(ind_fix, {})
    if skip_calcul == False:
        x_copy = np.copy(x_subset[-nstep:])
        x_copy[:, ind_fix] = [x_default]
        fix_temp = evaluate_wrap(evaluate, x_copy, a)
        y_fix = np.append(y_fix, fix_temp, axis=0)
        mse[f'{n}'] = [np.var(y_subset) / (y_subset.shape[0] ** 2)]
        y_true_ave = np.average(y_subset)
        rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
        error_dict[f'{n}'], pool_res, y_fix = group_fix(ind_fix, y_subset, y_fix, rand, {}, file_exist)
    else:
        # map index to calculated values
        error_dict[f'{n}'] = skip_calcul
        y_fix = np.copy(y_fix)
    # End for
print('=======FINISH CALCULATION========')

# convert the result into dataframe
df = pd.DataFrame.from_dict(error_dict)
df.index = METRIC_NAME
df.to_csv(f'{MORRIS_DATA_DIR}/saltelli_adaptive/fix_{len(ind_fix)}.csv')

if not y_true_exist:
    xy_df = pd.DataFrame(data = x_all, index = np.arange(nstop), columns = problem['names'])
    xy_df.loc[:, 'y'] = y_subset
    xy_df.to_csv(f'{MORRIS_DATA_DIR}/metric_samples.csv')
    mse = pd.DataFrame.from_dict(mse).T
    mse.rename(columns = {0: 'mse'})
    mse.to_csv(f'{MORRIS_DATA_DIR}/saltelli_adaptive/mse.csv')

print('=======FINISH WRITING OUTPUTS========')

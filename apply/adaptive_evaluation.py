"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
import json
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func
from utils.group_fix import group_fix
from utils.partial_sort import to_df

from settings import MORRIS_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = f'{MORRIS_DATA_DIR}morris_1010.json'
# cache_file = f'../output/morris/morris_1010.json'
with open(cache_file, 'r') as fp: partial_order = json.load(fp)

# calculate results with fixed parameters
seed = 101
# defaults_list = np.append([0, 0.1, 0.2, 0.4, 0.5], np.round(np.linspace(0.21, 0.3, 10), 2))
defaults_list = [0.25]
defaults_list.sort()
error_dict = {}; partial_key = 'result_80'
file_exists = True
mse = {}
nstart, nstop, nstep = 10, 1001, 10
x_all = sample_latin.sample(problem, nstop, seed)

for n in range(nstart, nstop, nstep):
    print(n)
    x_subset = x_all[:n]    
    try:
        y_subset = np.append(y_subset, evaluate(x_subset[-nstep:], a))
    except NameError:
        y_subset = evaluate(x_subset, a)

    mse[f'{n}'] = [np.var(y_subset) / y_subset.shape[0]]
    y_true_ave = np.average(y_subset)
    rand = np.random.randint(0, y_subset.shape[0], size=(1000, y_subset.shape[0]))
    for x_default in defaults_list:
        error_dict[f'{n}'], pool_res = group_fix(partial_order[partial_key], evaluate, 
                                        x_subset, y_subset, x_default, rand, 
                                        {}, a, file_exists)
    # End for
# End for

# convert the result into dataframe
key_outer = list(error_dict.keys())
f_names = list(error_dict[key_outer[0]].keys())
for ele in f_names:
    dict_measure = {key: error_dict[key][ele] for key in key_outer}
    df = pd.DataFrame.from_dict(dict_measure)
    df.to_csv(f'{MORRIS_DATA_DIR}/adaptive/{ele}.csv')
mse = pd.DataFrame.from_dict(mse).T
mse.rename(columns = {0: 'mse'})
mse.to_csv(f'{MORRIS_DATA_DIR}/adaptive/mse.csv')
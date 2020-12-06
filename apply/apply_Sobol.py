"""Apply experiment with Sobol."""

import pandas as pd
import numpy as np
from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import saltelli as sample_saltelli
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.partial_sort import to_df, partial_rank
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap

from settings import SOBOL_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = f'../output/sobol/sobol_main_123.json'

seed = 123; dummy = False
file_exists = os.path.exists(cache_file)
if not file_exists:
    # Loop of Morris
    partial_order = {}
    sa_main, sa_main_conf = {}, {}
    sa_total, sa_total_conf = {}, {}
    n_start, n_end, n_step = 200, 6000, 200
    x_large_size = sample_saltelli.sample(problem, n_end, calc_second_order=False, seed=seed)
    for i in range(n_start, n_end, n_step):
        # partial ordering
        x_sobol = x_large_size[:i * (len_params + 2)]
        if i == n_start:
            y_sobol = evaluate(x_sobol, a)
        else:
            y_eval = evaluate(x_sobol[-(len_params + 2) * n_step:], a)
            # y_eval = evaluate(x_morris[-(len_params + 1) * n_step:], a)
            y_sobol =  np.append(y_sobol, y_eval)
        sa_sobol = analyze_sobol.analyze(problem, y_sobol, 
                        calc_second_order=False, num_resamples=1000, conf_level=0.95, dummy=dummy)

        # use toposort to find parameter sa block
        conf_lower = sa_sobol['main_rank_ci'][0]
        conf_upper = sa_sobol['main_rank_ci'][1]
        if dummy: 
            len_dummy = len_params + 1
            abs_sort = partial_rank(len_dummy, conf_lower, conf_upper)
        else:
            abs_sort = partial_rank(len_params, conf_lower, conf_upper)

        rank_list = list(toposort(abs_sort))
        key = 'result_'+str(i)     
        partial_order[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}

        #save results returned from Morris if needed
        sa_total[key] = sa_sobol['ST']
        sa_total_conf[key] = sa_sobol['ST_conf']
        sa_main[key] = sa_sobol['S1']
        sa_main_conf[key] = sa_sobol['S1_conf']

    with open(cache_file, 'w') as fp:
        json.dump(partial_order, fp, indent=2)
else:
    with open(cache_file, 'r') as fp:
        partial_order = json.load(fp)
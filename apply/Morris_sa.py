"""Apply sensitivity analysis and rank factors with Sobol."""
import pandas as pd
import numpy as np
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import morris as sample_morris
from SALib.analyze import morris as analyze_morris
from SALib.test_functions.Sobol_G import evaluate

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func
from utils.partial_sort import partial_rank

from settings import MORRIS_DATA_DIR, SOBOL_DATA_DIR

def morris_ranking(cache_file):
    file_exists = os.path.exists(cache_file)
    if not file_exists:
        # Loop of Morris
        partial_order = {}
        mu_st, sigma_dt = {}, {}
        rank_lower_dt, rank_upper_dt = {}, {}
        n_start, n_end, n_step = 10, 130, 10
        x_large_size = sample_morris.sample(problem, n_end, num_levels=4, seed=1010)
        for i in range(n_start, n_end, n_step):
            # partial ordering
            x_morris = x_large_size[:i * (len_params + 1)]
            if i == n_start:
                y_morris = evaluate(x_morris, a)
            else:
                y_eval = evaluate(x_morris[-(len_params + 1) * n_step:], a)
                y_morris =  np.append(y_morris, y_eval)
            sa_dict = analyze_morris.analyze(problem, x_morris, y_morris, 
                num_resamples=1000, conf_level=0.95, seed=101)
            # mu_star_rank_dict = sa_dict['mu_star'].argsort().argsort()
            # use toposort to find parameter sa block
            conf_lower = sa_dict['mu_star_rank_conf'][0]
            conf_upper = sa_dict['mu_star_rank_conf'][1]
            rank_list = partial_rank(conf_lower, conf_upper)
            key = 'result_'+str(i)     
            partial_order[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}

            #save results returned from Morris if needed
            mu_st[key] = sa_dict['mu_star']
            rank_lower_dt[key] = conf_lower
            rank_upper_dt[key] = conf_upper
            sigma_dt[key] = sa_dict['sigma']

        with open(cache_file, 'w') as fp:
            json.dump(partial_order, fp, indent=2)

        return partial_order, x_large_size
    else:
        with open(cache_file, 'r') as fp:
            partial_order = json.load(fp) 
        return partial_order
    # END def morris_ranking()

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

# sensitivity analysis with Morris
outer_path = f'../output/adaptive_replicates/morris/'
cache_file = f'{outer_path}morris.json'
file_exists = os.path.exists(cache_file)
if file_exists:
    partial_order = morris_ranking(cache_file)
else:
    partial_order, x_morris = morris_ranking(cache_file)
    # np.savetxt(outer_path + 'morris_sample.txt', x_morris)

"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import morris as sample_morris
from SALib.analyze import morris as analyze_morris
from SALib.sample import saltelli as sample_saltelli
from SALib.analyze import sobol as analyze_sobol
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.test_function_setting import set_sobol_g_func, add_dummy
from utils.partial_sort import to_df, partial_rank
from utils.group_fix import evaluate_wrap

from settings import MORRIS_DATA_DIR, SOBOL_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

def morris_ranking(cache_file, seed=101):
    file_exists = os.path.exists(cache_file)
    if not file_exists:
        # Loop of Morris
        partial_order = {}
        mu_st, sigma_dt = {}, {}
        rank_lower_dt, rank_upper_dt = {}, {}
        n_start, n_end, n_step = 10, 130, 10
        x_large_size = sample_morris.sample(problem, n_end, num_levels=4, seed=684)
        for i in range(n_start, n_end, n_step):
            # partial ordering
            x_morris = x_large_size[:i * (len_params + 1)]
            if i == n_start:
                y_morris = evaluate(x_morris, a)
            else:
                y_eval = evaluate(x_morris[-(len_params + 1) * n_step:], a)
                y_morris =  np.append(y_morris, y_eval)
            sa_dict = analyze_morris.analyze(problem, x_morris, y_morris, num_resamples=1000, conf_level=0.95, seed=123)
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

def sobol_ranking(cache_file, dummy=False):
    """Apply experiment with Sobol."""
    seed = np.random.randint(0, 1000)
    file_exists = os.path.exists(cache_file)
    if not file_exists:
        # Loop of Morris
        partial_order = {}
        sa_main, sa_main_conf = {}, {}
        sa_total, sa_total_conf = {}, {}
        nstart, nstop, nstep = 200, 801, 200
        x_large_size = sample_saltelli.sample(problem, nstop, calc_second_order=False, seed=seed)
        for i in range(nstart, nstop, nstep):
            # partial ordering
            x_sobol = x_large_size[:i * (len_params + 2)]
            if i == nstart:
                y_sobol = evaluate(x_sobol, a)
            else:
                y_eval = evaluate(x_sobol[-(len_params + 2) * nstep:], a)
                # y_eval = evaluate(x_morris[-(len_params + 1) * nstep:], a)
                y_sobol =  np.append(y_sobol, y_eval)
            sa_sobol = analyze_sobol.analyze(problem, y_sobol, 
                            calc_second_order=False, num_resamples=1000, conf_level=0.95)
            # use toposort to find parameter sa block
            conf_lower = sa_sobol['total_rank_ci'][0]
            conf_upper = sa_sobol['total_rank_ci'][1]
            if dummy: 
                len_dummy = len_params + 1
                abs_sort = partial_rank(conf_lower, conf_upper)
            else:
                rank_list = partial_rank(conf_lower, conf_upper)

            key = 'result_'+str(i)     
            partial_order[key] = {j: list(rank_list[j]) for j in range(len(rank_list))}

            #save results returned from Morris if needed
            sa_total[key] = sa_sobol['ST']
            sa_total_conf[key] = sa_sobol['ST_conf']
            sa_main[key] = sa_sobol['S1']
            sa_main_conf[key] = sa_sobol['S1_conf']

        with open(cache_file, 'w') as fp:
            json.dump(partial_order, fp, indent=2)
        return partial_order, x_sobol, y_sobol

    else:
        with open(cache_file, 'r') as fp:
            partial_order = json.load(fp)

        return partial_order
    # END def sobol_ranking()

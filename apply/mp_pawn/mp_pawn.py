#%% Step 1 (import python modules)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# Import SAFE modules:
from SAFEpython import PAWN
import SAFEpython.plot_functions as pf # module to visualize the results
from SAFEpython.model_execution import model_execution # module to execute the model
from SAFEpython.sampling import AAT_sampling # module to perform the input sampling
from SAFEpython.util import aggregate_boot, KS_ranking  # function to aggregate the bootstrap results

# import other module needed
from saffix.sobol_g.Sobol_G import g_func
from toposort import toposort, toposort_flatten
import json
import os

import SALib
from SALib.sample import latin as sample_latin
from saffix.utils.group_fix import group_fix
from saffix.utils.format_convert import to_df

#%% Step 2: define the problem and fuctions for analysis

# import settings for Sobol G-function and returns necessary elements
from saffix.sobol_g.Sobol_G_setting import set_sobol_g_func


def mp_pawn(s_start, s_end, step, tuning_list, f_dir, Nboot=1000):
    """Run multiple experiments with PAWN.

    Applies PAWN for each tuning value.

    Parameters
    ----------
    s_start : int, 
        Start of segment

    s_end : int,
        End of segment (exclusive)

    step : int,
        Step size

    tuning_list : List[int],
        List of tuning values to loop over

    f_dir : str,
        Path to output directory to write results to

    Nboot : int,
        Number of bootstraps used to derive confidence bounds

    Returns
    -------
    None

    Outputs
    -------
    JSON file of results in specified output directory (`f_dir`).
    """

    a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

    distr_fun = [st.uniform] * len_params
    distr_par = [[x_bounds[i][0], x_bounds[i][1] - x_bounds[i][0]] 
                 for i in range(len_params)]

    for tuning in tuning_list:
        cache_file = f'{f_dir}pawn_{tuning}_{s_start}-{s_end}_{step}.json'
        if os.path.exists(cache_file):
            continue
        # Loop of Morris
        partial_order = {}
        pawn_mean = {}
        for N in range(s_start, s_end, step):
            samp_strat = 'lhs' # Latin Hypercube
            X = AAT_sampling(samp_strat, len_params, distr_fun, distr_par, N)

            # Run the model:
            Y = g_func(X, a)

            # Set the number of conditioning intervals:
            # option 1 (same value for all inputs):
            n = tuning

            # Choose one among multiple outputs for subsequent analysis:
            Yi = Y[:]

            # Check how the sample is split for parameter MAXBAS that takes discrete values:
            YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Yi, n)

            # Compute and plot conditional and unconditional CDFs
            YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Yi, n)

            # Add colorbar:
            YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Yi, n, cbar=True, n_col=3, labelinput=x_names)

            # Compute and plot KS statistics for each conditioning interval
            KS = PAWN.pawn_plot_ks(YF, FU, FC, xc)

            # Customize plot:
            KS = PAWN.pawn_plot_ks(YF, FU, FC, xc, X_Labels=x_names)

            # Compute PAWN sensitivity indices:
            KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Yi, n)
            
            # Compute sensitivity indices for Nboot bootstrap resamples
            KS_median, KS_mean, KS_max, KS_dummy = PAWN.pawn_indices(X, Yi, n, Nboot=Nboot, dummy=True)

            # KS_median and KS_mean and KS_max have shape (Nboot, M)
            KS_dummy = KS_dummy.reshape(KS_dummy.shape[0], 1)

            KS_median_dummy = np.append(KS_median, KS_dummy, axis=1)
            KS_mean_dummy = np.append(KS_mean, KS_dummy, axis=1)
            KS_max_dummy = np.append(KS_max, KS_dummy, axis=1)

            KS_median_m, KS_median_lb, KS_median_ub = aggregate_boot(KS_median_dummy)
            KS_mean_m, KS_mean_lb, KS_mean_ub = aggregate_boot(KS_mean_dummy)
            KS_max_m, KS_max_lb, KS_max_ub = aggregate_boot(KS_max_dummy)
    
            # return the ranking of parameters
            rank_m, rank_conf = KS_ranking(KS_median, alfa=0.05) 
            conf_low = rank_conf[0]
            conf_up = rank_conf[1]

            abs_sort = {}
            for m in range(len_params):
                list_temp = np.where(conf_low >= conf_up[m])
                
                set_temp = set()
                if len(list_temp) > 0:
                    for ele in list_temp[0]:
                        set_temp.add(ele)

                abs_sort[m] = set_temp
            # End for

            order_temp = list(toposort(abs_sort))

            partial_order['result_'+str(N)] = {j: list(order_temp[j]) for j in range(len(order_temp))}
            pawn_mean['result_'+str(N)] = KS_mean_m
        # End for

        with open(cache_file, 'w') as fp:
            json.dump(partial_order, fp, indent=2)
    # End for
# End run_pawn_mp()
            



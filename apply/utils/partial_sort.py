import numpy as np
import pandas as pd

def to_df(partial_order, fix_dict):
    """
    Convert difference between conditioned and unconditioned 
    into dataframe.

    Parameters
    ----------
    partial_order : dict, 
        partial ranking of parameters

    fix_dict : dict, 
        difference between conditioned and unconditional model results.
        (each dict result returned from group_fix / pce_group_fix)

    Returns
    ----------
    fix_df : df, 
        formatted fix_dict
    """
    keys_list = list(partial_order.keys())

    fix_df = {key:None for key in keys_list}

    for key in keys_list:
        len_each_group = []
        len_each_group = [len(value) for k, value in partial_order[key].items()]
        fix_temp = []

        for g, v in fix_dict[key].items():
            if isinstance(v, tuple):
                fix_temp.extend([v[0]])
            else:
                fix_temp.extend([v])
        fix_df[key]  = np.repeat(fix_temp, len_each_group)

    fix_df = pd.DataFrame.from_dict(fix_df)
    return fix_df

def partial_rank(len_params, conf_lower, conf_upper):
    """Perform partial ranking.

    Parameters
    ----------
    len_params : int, 
        Number of parameters

    conf_lower : numpy.ndarray, 
        Lower confidence interval values for each parameter

    conf_upper : numpy.ndarray, 
        Upper confidence interval values for each parameter

    Returns
    ----------
    rank_list: dict,
        list of partial rank
    """
    rank_conf = {j:None for j in range(len_params)}
    for j in range(len_params):
        rank_conf[j] = [conf_lower[j], conf_upper[j]]

    abs_sort= {j:None for j in range(len_params)}
    for m in range(len_params):
        list_temp = np.where(conf_lower >= conf_upper[m])
        set_temp = set()
        if len(list_temp) > 0:
            for ele in list_temp[0]:
                set_temp.add(ele)
        abs_sort[m] = set_temp

    return abs_sort

def compute_bootstrap_ranks(sobol_order_resample, conf_level):
    """Calculate confidence interval for ranking of mu_star.
    """
    D, num_resamples = sobol_order_resample.shape
    rankings = np.zeros_like(sobol_order_resample)
    ranking_ci = np.zeros((D, 2))
    for resample in range(num_resamples):
	    rankings[:, resample] = np.argsort(sobol_order_resample[:, resample]).argsort()

    ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=1)

    return ranking_ci
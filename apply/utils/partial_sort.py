import numpy as np
import pandas as pd
from toposort import toposort, toposort_flatten

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

def partial_rank(conf_lower, conf_upper):
    """Perform partial ranking.

    Parameters
    ----------
    conf_lower : numpy.ndarray, 
        Lower confidence interval values for each parameter

    conf_upper : numpy.ndarray, 
        Upper confidence interval values for each parameter

    Returns
    ----------
    rank_list: dict,
        list of partial rank
    """
    num_vars = len(conf_lower)
    rank_conf = {j:None for j in range(num_vars)}
    for j in range(num_vars):
        rank_conf[j] = [conf_lower[j], conf_upper[j]]

    abs_sort= {j:None for j in range(num_vars)}
    for m in range(num_vars):
        list_temp = np.where(conf_lower >= conf_upper[m])
        set_temp = set()
        if len(list_temp) > 0:
            for ele in list_temp[0]:
                set_temp.add(ele)
        abs_sort[m] = set_temp
    rank_list = list(toposort(abs_sort))
    return rank_list

def compute_bootstrap_ranks(sa_matrix, conf_level):
    """Calculate confidence interval for ranking of mu_star.
    """
    assert (isinstance(sa_matrix, np.ndarray)), \
        'sa_matrix should be an array of two dimensions (num_vars, num_samples)'    
    D, num_resamples = sa_matrix.shape
    rankings = np.zeros_like(sa_matrix)
    ranking_ci = np.zeros((D, 2))
    for resample in range(num_resamples):
	    rankings[:, resample] = np.argsort(sa_matrix[:, resample]).argsort()

    ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=1)

    return ranking_ci
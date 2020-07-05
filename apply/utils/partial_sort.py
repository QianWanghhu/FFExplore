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

def partial_rank(len_params, conf_low, conf_up):
    """Perform partial ranking.

    Parameters
    ----------
    len_params : int, 
        Number of parameters

    conf_low : numpy.ndarray, 
        Lower confidence interval values for each parameter

    conf_up : numpy.ndarray, 
        Upper confidence interval values for each parameter

    Returns
    ----------
    rank_list: dict,
        list of partial rank
    """
    rank_conf = {j:None for j in range(len_params)}
    for j in range(len_params):
        rank_conf[j] = [conf_low[j], conf_up[j]]  

    abs_sort= {j:None for j in range(len_params)}
    for m in range(len_params):
        list_temp = np.where(conf_low >= conf_up[m])
        set_temp = set()
        if len(list_temp) > 0:
            for ele in list_temp[0]:
                set_temp.add(ele)
        abs_sort[m] = set_temp

    return abs_sort
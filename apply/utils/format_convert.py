import numpy as np
import pandas as pd

def to_df(partial_order, fix_dict, y_true):
    """
    Help function to convert difference between 
    conditioned and unconditioned into dataframe.
    Parameters:
    ===========
    partial_order : dict, partial ranking of parameters
    fix_dict : dict, difference between conditioned and unconditional model results.
                (each dict result returned from group_fix / pce_group_fix)
    y_true : numpy.ndarray, unconditional model results

    Returns:
    ========
    fix_df : df, formatted fix_dict
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
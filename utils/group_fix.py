import numpy as np
from scipy.stats import pearsonr

def group_fix(partial_result, func, x, y_true, x_default, a=None, file_exist=False, rand=None):
    """
    Function for compare results between conditioned and unconditioned QoI.
    Fix parameters from the least influential group 
    based on results from partially sorting.
    Four types of error measures will be returned.
    Parameters:
    ===========
    partial_result: dictionary of parameter groups, results of partial sort
    func: function for analysis (analytical formula or model)
    x: numpy array of input, shape of N * D where N is sampling size and 
        D is the number of parameters
    y_true: list of func results with all x varying (the raw sampling matrix of x)
    x_default: scalar or listdefault values of x
    file_exist : Boolean for checking whether the partial results is from calculation
                or reading from the existing file.
    a: coefficients used in func
    
    Return:
    =======
    compare_mae :dict, changes in absolute mean error of the func results due to fixing parameters
    compare_var : dict, changes in variance of the func results due to fixing parameters
    pearsons : dict, changes in pearson correlation coefficients 
                of the func results due to fixing parameters
    """
    num_group = len(partial_result) - 1
    # store results from fixing parameters in dict
    compare_var = {i: None for i in range(num_group)}
    compare_mae = {i: None for i in range(num_group)}
    pearsons = {i: None for i in range(num_group)}
    # sample_fix = {i: None for i in range(num_group)}
    ind_fix = []
    y_true_ave = np.mean(y_true)
    y_true_var = np.var(y_true)
    for i in range(num_group, -1, -1):
        if file_exist:
            try:
                ind_fix.extend(partial_result[str(i)])
            except NameError:
                ind_fix = partial_result[str(i)]
        else:
            try:
                ind_fix.extend(partial_result[i])
            except NameError:
                ind_fix = partial_result[i]
        sample_copy = np.copy(x) 

        sample_copy[:, ind_fix] = [x_default]
        if  a is None:
            results_fix = func(*sample_copy.T)
        else:
            results_fix = func(sample_copy, a)
            
        # compare results with insignificant parameters fixed
        compare_var[i] = np.var(results_fix) / y_true_var
        pearsons[i] = pearsonr(results_fix, y_true)
        compare_mae[i] = np.abs((results_fix - y_true)).mean(axis=0) / y_true_ave
        # compare_mae[i] = (np.abs(results_fix - y_true) / y_true_ave).mean(axis=0)
    return compare_mae, compare_var, pearsons

def to_df(partial_order, fix_dict):
    """
    Help function to convert difference between 
    conditioned and unconditioned into dataframe.
    Parameters:
    ===========
    partial_order : dict, partial ranking of parameters
    fix_dict : dict, difference between conditioned and unconditional model results.
                (each dict result returned from group_fix / pce_group_fix)

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
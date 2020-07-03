import numpy as np
from scipy.stats import pearsonr, norm
import pandas as pd

def group_fix(partial_result, func, x, y_true, x_default, rand, a=None, file_exist=False):
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
    rand: np.ndarray of resample index in bootstrap, shape of R * N where R is the number of resamples
    file_exist : Boolean for checking whether the partial results is from calculation
                or reading from the existing file.
    a: coefficients used in func
    
    Return:
    =======
    mae : dict, changes in absolute mean error of the func results due to fixing parameters
    var : dict, changes in variance of the func results due to fixing parameters
    ppmc : dict, changes in pearson correlation coefficients 
                of the func results due to fixing parameters
    """
    num_group = len(partial_result) - 1
    # store results from fixing parameters in dict
    mae, var, ppmc = {}, {}, {}
    mae_up, var_up, prsn_up = {}, {}, {}
    mae_low, var_low, prsn_low = {}, {}, {}
    # mae = {i: None for i in range(num_group)}
    # ppmc = {i: None for i in range(num_group)}
    # var_up = {i: None for i in range(num_group)}
    # ppmc_up = {i: None for i in range(num_group)}
    # mae_up = {i: None for i in range(num_group)}
    # var_low = {i: None for i in range(num_group)}
    # ppmc_low = {i: None for i in range(num_group)}
    # mae_low = {i: None for i in range(num_group)}
    ind_fix = []
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
        x[:, ind_fix] = [x_default]
        results_fix = func(x, a)

        # compare results with insignificant parameters fixed
        Nresample = rand.shape[0]
        var_bt,  ppmc_bt,  mae_bt = np.zeros(Nresample), \
                                    np.zeros(Nresample), np.zeros(Nresample)
        for ii in range(Nresample):            
            I = rand[ii]
            y_true_resample = y_true[I]
            results_fix_resample = results_fix[I]
            y_true_ave = y_true_resample.mean()
            y_true_var = y_true_resample.var()
            var_bt[ii] = results_fix_resample.var() / y_true_var
            ppmc_bt[ii] = pearsonr(results_fix_resample, y_true_resample)[0]
            mae_bt[ii] = np.abs(results_fix_resample - y_true_resample).mean(axis=0) / y_true_ave
        
        var[i] = var_bt.mean()
        ppmc[i] = ppmc_bt.mean()
        mae[i] = mae_bt.mean()
        var_low[i], var_up[i] = np.quantile(var_bt, [0.025, 0.975])
        ppmc_low[i], ppmc_up[i] = np.quantile(ppmc_bt, [0.025, 0.975])
        mae_low[i], mae_up[i] = np.quantile(mae_bt, [0.025, 0.975])
        dict_return = {'mae': mae, 'var': var, 'ppmc': ppmc,
                        'mae_low': mae_low, 'var_low': var_low, 'ppmc_low': ppmc_low,
                        'mae_up': mae_up, 'var_up': var_up, 'ppmc_up': ppmc_up}
    
    return dict_return

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
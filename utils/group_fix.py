import numpy as np
from scipy.stats import pearsonr

def group_fix(partial_result, func, x, y_true, x_default, a=None, file_exist=False):
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
    compare_mean : dict, changes in mean of the func results due to fixing parameters
    compare_mae :dict, changes in absolute mean error of the func results due to fixing parameters
    compare_var : dict, changes in variance of the func results due to fixing parameters
    pearsons : dict, changes in pearson correlation coefficients 
                of the func results due to fixing parameters
    """
    num_group = len(partial_result) - 1
    # store results from fixing parameters in dict
    compare_mean = {i: None for i in range(num_group)}
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
        sample_copy = np.copy(x) # for pce, x is sampled by dist.sample

        sample_copy[:, ind_fix] = [x_default]
        if  a is None:
            results_fix = func(*sample_copy.T)
        else:
            results_fix = func(sample_copy, a)
            
        # compare results with insignificant parameters fixed
        # calculate the average and expected error
        compare_mean[i] = (np.square(results_fix - y_true)).mean(axis=0) / y_true_var
        compare_var[i] = np.var(results_fix) / y_true_var
        pearsons[i] = pearsonr(results_fix, y_true)
#         sample_fix[i] = results_fix
        # compare_mae[i] = np.abs((results_fix - y_true)).mean(axis=0) / y_true_ave
        compare_mae[i] = (np.abs(results_fix - y_true) / y_true_ave).mean(axis=0)
        # pearsons[i] = 1 - (((results_fix - y_true) **2).sum(axis=0) / y_true_var / len(y_true))
    return compare_mean, compare_mae, compare_var, pearsons

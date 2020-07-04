import numpy as np
from scipy.stats import pearsonr, norm
import pandas as pd

def group_fix(partial_result, func, x, y_true, x_default, 
            rand, pool_results, a=None, file_exist=False):
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
    mae_up, var_up, ppmc_low = {}, {}, {}
    mae_low, var_low, ppmc_up = {}, {}, {}
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
        ind_fix.sort()        
        # check whether results existing        
        skip_calcul = results_exist(ind_fix, pool_results)

        if skip_calcul == False:
            x_copy = np.copy(x)
            x_copy[:, ind_fix] = [x_default]
            results_fix = func(x_copy, a)
            # compare results with insignificant parameters fixed
            Nresample = rand.shape[0]
            var_bt,  ppmc_bt,  mae_bt = np.zeros(Nresample), np.zeros(Nresample), np.zeros(Nresample)

            for ii in range(Nresample):            
                I = rand[ii]
                y_true_resample = y_true[I]
                results_fix_resample = results_fix[I]
                y_true_ave = y_true_resample.mean()
                y_true_var = y_true_resample.var()
                var_bt[ii] = results_fix_resample.var() / y_true_var
                ppmc_bt[ii] = pearsonr(results_fix_resample, y_true_resample)[0]
                mae_bt[ii] = np.abs(results_fix_resample - y_true_resample).mean(axis=0) / y_true_ave
            # End for()
            
            mae[i], var[i], ppmc[i] = mae_bt.mean(), var_bt.mean(), ppmc_bt.mean()
            var_low[i], var_up[i] = np.quantile(var_bt, [0.025, 0.975])
            ppmc_low[i], ppmc_up[i] = np.quantile(ppmc_bt, [0.025, 0.975])
            mae_low[i], mae_up[i] = np.quantile(mae_bt, [0.025, 0.975])

            # update pool_results
            measure_list = [
                            mae[i], var[i], ppmc[i], mae_low[i], mae_up[i], 
                            var_low[i], var_up[i],ppmc_low[i], ppmc_up[i],
                            ]

            pool_results = pool_update(ind_fix, measure_list, pool_results)
        else:
            # map index to calculated values
            [mae[i], var[i], ppmc[i], mae_low[i], mae_up[i],
            var_low[i], var_up[i], ppmc_low[i], ppmc_up[i]] = skip_calcul

    dict_return = {'mae': mae, 'var': var, 'ppmc': ppmc,
                    'mae_low': mae_low, 'var_low': var_low, 'ppmc_low': ppmc_low,
                    'mae_up': mae_up, 'var_up': var_up, 'ppmc_up': ppmc_up}
    # End for()

    return dict_return, pool_results

def results_exist(parms_fixed, pool_results):
    """
    Help function to check whether results existed already.
    Parameters
    ===========
    parms_fixed : list, index of parameters that are to fix
    pool_results : dict, contains both index of parameters fixed and the corresponding results

    Return
    =======
    skip_cal
    """ 
    if pool_results == {}:
        skip_cal = False
    elif parms_fixed in pool_results['parms']:
        index_measure = pool_results['parms'].index(parms_fixed) 
        skip_cal = pool_results[f'measures_{index_measure}']
    else:
        skip_cal = False
    return skip_cal
    
def pool_update(parms_fixed, measure_list, pool_results):
    """
    Function used to update the pool_results.
    Parameters
    ===========
    parms_fixed : list, index of parameters that are to fix
    measure_list : list, measures newly calculated for parameters in parms_fixed
    pool_results : dict, contains both index of parameters fixed and the corresponding results

    Return:
    pool_results
    """

    try:
        pool_results['parms'].append(parms_fixed[:])
    except KeyError:
        pool_results['parms'] = [parms_fixed[:]]      
    index_measure = pool_results['parms'].index(parms_fixed)
    pool_results[f'measures_{index_measure}'] = measure_list

    return pool_results

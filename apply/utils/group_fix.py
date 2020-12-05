import numpy as np
from scipy.stats import pearsonr, norm
import pandas as pd

def _group_fix(partial_result, func, x, y_true, x_default, 
            rand, pool_results, y_subset_fix, a=None,
            file_exist=False):
    """
    Function for compare results between conditioned and unconditioned QoI.
    Fix parameters from the least influential group 
    based on results from partially sorting.

    Four error measure types will be returned.

    Parameters
    ----------
    partial_result : dict,
        dictionary of parameter groups, results of partial sort

    func : function,
        function for analysis (analytical formula or model)

    x : np.array,
        Input with shape of N * D where N is sampling size and 
        D is the number of parameters

    y_true : list,
        Function results with all x varying (the raw sampling matrix of x)

    x_default : int, float, list,
        Default values of x as a scalar or list of scalars

    rand : np.ndarray,
        Resample index in bootstrap, shape of R * N, 
        where R is the number of resamples

    pool_results : dict,
        Index of fixed parameters and the corresponding results

    a : np.array (Default: None),
        Coefficients used in `func`

    file_exist : bool (default: False), 
        If true, reads cached partial-ranking results from a file.
        Otherwise, calculates results.
    
    Returns
    ----------
    Tuple of:

    dict_return:
        mae : dict, 
            Changes in absolute mean error of the func results due to fixing 
            parameters

        var : dict, 
            Changes in variance of the func results due to fixing parameters

        ppmc : dict, 
            Changes in pearson correlation coefficients 
            of the func results due to fixing parameters

        mae_lower : dict,
            Lowest absolute MAE values

        var_lower :  dict, 
            Lowest variance

        ppmc_lower :  dict,
            Lowest PPMC

        mae_upper :  dict,
            Largest absolute MAE values

        var_upper :  dict,
            Largest variance values

        ppmc_upper :  dict,
            Largest PPMC values

    pool_results:

    
    """
    num_group = len(partial_result) - 1

    # store results from fixing parameters in dict
    mae = {i: None for i in range(num_group)}
    var, ppmc = dict(mae), dict(mae)
    mae_upper, var_upper, ppmc_upper = dict(mae), dict(mae), dict(mae)
    mae_lower, var_lower, ppmc_lower = dict(mae), dict(mae), dict(mae)
    ind_fix = []
    
    for i in range(num_group, -1, -1):
        ind_fix = index_fix(partial_result, i, file_exist, ind_fix)

        # check whether results existing        
        skip_calcul = results_exist(ind_fix, pool_results)
        print(skip_calcul)

        if skip_calcul == False:
            x_copy = np.copy(x)
            x_copy[:, ind_fix] = [x_default]
            x_temp = x_copy[y_subset_fix.shape[0]:y_true.shape[0], :]
            fix_temp = func(x_temp, a)
            results_fix = np.append(y_subset_fix, fix_temp, axis=0)
            # ff = np.copy(results_fix)

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
            # End for
            
            mae[i], var[i], ppmc[i] = mae_bt.mean(), var_bt.mean(), ppmc_bt.mean()
            var_lower[i], var_upper[i] = np.quantile(var_bt, [0.025, 0.975])
            ppmc_lower[i], ppmc_upper[i] = np.quantile(ppmc_bt, [0.025, 0.975])
            mae_lower[i], mae_upper[i] = np.quantile(mae_bt, [0.025, 0.975])

            # update pool_results
            measure_list = [
                            mae[i], var[i], ppmc[i], mae_lower[i], mae_upper[i], 
                            var_lower[i], var_upper[i],ppmc_lower[i], ppmc_upper[i],
                            ]

            pool_results = pool_update(ind_fix, measure_list, pool_results)
        else:
            # map index to calculated values
            [mae[i], var[i], ppmc[i], mae_lower[i], mae_upper[i],
            var_lower[i], var_upper[i], ppmc_lower[i], ppmc_upper[i]] = skip_calcul
            results_fix = np.copy(y_subset_fix)
        # End if
    # End for()

    dict_return = {'mae': mae, 'var': var, 'ppmc': ppmc,
                    'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                    'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}
    
    return dict_return, pool_results, results_fix


def group_fix(ind_fix, y_true, results_fix,
            rand, pool_results, file_exist=False):
    
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
    # End for
    
    mae, var, ppmc = mae_bt.mean(), var_bt.mean(), ppmc_bt.mean()
    var_lower, var_upper = np.quantile(var_bt, [0.025, 0.975])
    ppmc_lower, ppmc_upper= np.quantile(ppmc_bt, [0.025, 0.975])
    mae_lower, mae_upper = np.quantile(mae_bt, [0.025, 0.975])

    # update pool_results
    measure_list = [
                    mae, var, ppmc, mae_lower,  var_lower, 
                    ppmc_lower, mae_upper, var_upper, ppmc_upper,
                    ]

    pool_results = pool_update(ind_fix, measure_list, pool_results)
    # End if
# End for()

    dict_return = [mae, var, ppmc, mae_lower, var_lower, ppmc_lower, mae_upper, var_upper,  ppmc_upper]
    
    return dict_return, pool_results, results_fix



def index_fix(partial_result, ind, file_exist, ind_fix):
    if file_exist:
        try:
            ind_fix.extend(partial_result[str(ind)])
        except NameError:
            ind_fix = partial_result[str(ind)]
    else:
        try:
            ind_fix.extend(partial_result[ind])
        except NameError:
            ind_fix = partial_result[ind]
    ind_fix.sort()
    return ind_fix

def results_exist(parms_fixed, pool_results):
    """
    Helper function to determine whether results exist.

    Parameters
    ----------
    parms_fixed : list, 
        Index of parameters to fix

    pool_results : dict, 
        Contains both index of parameters fixed and the corresponding results

    Returns
    -------
    skip_cal : bool
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
    """Update pool_results with new values.

    Parameters
    ----------
    parms_fixed : list, 
        Index of parameters to fix

    measure_list : list, 
        Measures newly calculated for parameters in parms_fixed

    pool_results : dict, 
        Contains both index of parameters fixed and the corresponding results

    Returns
    ----------
    Updated pool_results
    """
    try:
        pool_results['parms'].append(parms_fixed[:])
    except KeyError:
        pool_results['parms'] = [parms_fixed[:]]      
    index_measure = pool_results['parms'].index(parms_fixed)
    pool_results[f'measures_{index_measure}'] = measure_list

    return pool_results

def evaluate_wrap(evaluate, x, a):
    if a is None:
        return evaluate(x)
    else:
        return evaluate(x, a)

import numpy as np
from scipy.stats import pearsonr, norm
import pandas as pd

def group_fix(ind_fix, y_true, results_fix,
            rand, pool_results, file_exist=False):
    
    # compare results with insignificant parameters fixed
    Nresample = rand.shape[0]
    var_bt,  ppmc_bt,  mae_bt = np.zeros(Nresample), np.zeros(Nresample), np.zeros(Nresample)
    assert isinstance(ind_fix, list), 'ind_fix should be a list'

    for ii in range(Nresample):            
        I = rand[ii]
        y_true_resample = y_true[I]
        results_fix_resample = results_fix[I]
        y_true_ave = y_true_resample.mean()
        y_true_var = y_true_resample.var()
        var_bt[ii] = results_fix_resample.var() / y_true_var
        ppmc_bt[ii] = pearsonr(results_fix_resample, y_true_resample)[0]
        mae_bt[ii] = np.abs((results_fix_resample - y_true_resample) / y_true_resample).mean(axis=0) # / y_true_ave
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

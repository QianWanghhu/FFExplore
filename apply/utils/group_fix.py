import numpy as np
from scipy.stats import pearsonr, norm
import pandas as pd
from scipy.stats import sem
import os

def group_fix(ind_fix, y_true, results_fix,
            rand, pool_results, file_exist=False, boot=False):
    
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
        # import pdb; pdb.set_trace()
        ppmc_bt[ii] = pearsonr(results_fix_resample, y_true_resample)[0]
        mae_bt[ii] = np.abs((results_fix_resample - y_true_resample) / y_true_resample).mean(axis=0) # / y_true_ave
    # End for
    
    mae, var, ppmc = mae_bt.mean(), var_bt.mean(), ppmc_bt.mean()
    if not boot:
        var_lower, var_upper = np.quantile(var_bt, [0.025, 0.975])
        ppmc_lower, ppmc_upper= np.quantile(ppmc_bt, [0.025, 0.975])
        mae_lower, mae_upper = np.quantile(mae_bt, [0.025, 0.975])
    else:
        var_lower, var_upper = var - np.std(var_bt), var + np.std(var_bt)
        ppmc_lower, ppmc_upper= ppmc - np.std(ppmc_bt), ppmc + np.std(ppmc_bt)
        mae_lower, mae_upper = mae - np.std(mae_bt), mae + np.std(mae_bt)

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

def evaluate_wrap(evaluate, x, **kwargs):
    if 'a' in list(kwargs.keys()): 
        a = kwargs['a']
        return evaluate(x, a)
    else:
        return evaluate(x.T).flatten()


def stderr(seq):
    """
    Calculate the standard error of mean.
    """        
    assert isinstance(seq, np.ndarray), \
        "The input matrix should be a matrix of two dimensions (N * r)"
    seq_mean = seq.mean(axis=1)
    seq_stderr = sem(seq, axis=1, ddof=1)
    return seq_mean, seq_stderr

def std_stderr(seq):
    """
    Calculate the standard error of standard deviation.
    """        
    assert isinstance(seq, np.ndarray), \
        "The input matrix should be a matrix of two dimensions (N * r)"
    r = seq.shape[1]
    seq_std = np.std(seq, axis=1)
    seq_mean, _ = stderr(seq)
    sigma = np.square((seq.transpose() - seq_mean).transpose())
    sigma_mean = sigma.mean(axis=1)
    sigma_deviation = np.sum(np.square((sigma.transpose() - sigma_mean).transpose()), axis=1)
    seq_stderr = np.sqrt(r / (r -1) ** 3 * sigma_deviation)

    return seq_std, seq_stderr

def loop_error_metrics(out_path, x_fix_set, x_default, nsubsets, r, len_params, 
    samples, evaluate, boot, file_exists, **kwargs):
# The loop of calculating error metrics 
    if boot: nboot = kwargs['nboot']
    save_file = kwargs['save_file']

    try:
        nstart = kwargs['nstart']
    except KeyError:
        nstart = 0

    for ind_fix in x_fix_set:
        print(ind_fix)
        error_dict = {}; pool_res = {}
        # loop to fix parameters and calculate the error metrics    
        for i in range(r):
            mae = {i: None for i in range(nsubsets)}
            var, ppmc = dict(mae), dict(mae)
            mae_upper, var_upper, ppmc_upper = dict(mae), dict(mae), dict(mae)
            mae_lower, var_lower, ppmc_lower = dict(mae), dict(mae), dict(mae)
            x_sample = samples[:, (i * len_params):(i + 1) * len_params]
            y_true = evaluate_wrap(evaluate, x_sample, **kwargs)
            
            # Loop of each subset 
            for n in range(nstart, nsubsets):
                y_subset = y_true[0:(n + 1)*10]
                x_copy = np.copy(x_sample[0: (n + 1) * 10, :])
                x_copy[:, ind_fix] = [x_default]
                y_fix = evaluate_wrap(evaluate, x_copy, **kwargs)
                y_true_ave = np.average(y_subset)
                if boot:
                    rand = np.random.randint(0, x_copy.shape[0], size = (nboot, x_copy.shape[0]))
                else:
                    rand = np.array([np.arange(0, x_copy.shape[0])]) 
                error_temp, pool_res, _ = group_fix(ind_fix, y_subset, \
                    y_fix, rand, pool_res, file_exists, boot)
        
                [mae[n], var[n], ppmc[n], mae_lower[n], var_lower[n], ppmc_lower[n], 
                mae_upper[n],var_upper[n], ppmc_upper[n]] = error_temp

            error_dict[f'replicate{i}'] = {'mae': mae, 'var': var, 'ppmc': ppmc,
                            'mae_lower': mae_lower, 'var_lower': var_lower, 'ppmc_lower': ppmc_lower,
                            'mae_upper': mae_upper, 'var_upper': var_upper, 'ppmc_upper': ppmc_upper}
        # import pdb; pdb.set_trace()
        if save_file:
            # convert the result into dataframe
            key_outer = list(error_dict.keys())
            f_names = list(error_dict[key_outer[0]].keys())
            len_fix = len(ind_fix)
            fpath = f'{out_path}/fix_{len_fix}/'
            if not os.path.exists(fpath): os.mkdir(fpath)
            for key in key_outer:
                # dict_measure = {key: error_dict[key][ele] for key in key_outer}
                df = pd.DataFrame.from_dict(error_dict[key], orient='columns')
                df.to_csv(f'{fpath}{key}.csv')
            # End for
        else:
            return error_dict
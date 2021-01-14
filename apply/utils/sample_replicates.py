""" Generate a number of r=10 independent Sobol' sampling sequences"""
import os
import numpy as np
import pandas as pd
from SALib.sample import sobol_sequence

from .group_fix import stderr, std_stderr

def sample_repli(N, len_params, metric_cache, split_style = 'vertical', skip_numbers = 1000, 
    num_replicates = 10):
    """
    Generate samples used for calculating the error metrics.
    split_style: str, define how to split the Sobol sequence into multiple independent sample set.
    """
    if split_style == 'horizontal': 
        base_sequence = sobol_sequence.sample(N + skip_numbers, 
            len_params * num_replicates)[skip_numbers:]
        np.savetxt(metric_cache, base_sequence)

    elif split_style == 'vertical':
        temp_sequence = sobol_sequence.sample(N * num_replicates + skip_numbers, 
            len_params)[skip_numbers:]
        base_sequence = np.hstack(np.vsplit(temp_sequence, num_replicates))
        np.savetxt(metric_cache, base_sequence)
    else:
        raise AssertionError
    return base_sequence

def replicates_process(out_path, col_names, nsubsets, r, save_file=True):
    """
    Postprocess the results using replicates.
    """
    fps = [fp for fp in os.listdir(out_path) if 'fix' in fp]
    print(fps)
    for fp in fps:
        print(fp)
        filenames = [f for f in os.listdir(f'{out_path}{fp}/') if 'repl' in f]
        std_estimation = np.zeros(shape = (nsubsets, 3 * 2))
        mean_estimation = np.zeros(shape = (nsubsets, 3 * 2)) # the number of columns is double the number of error metrics used
        for i in range(3):
            seq = np.zeros(shape=(nsubsets, r))
            for jj in range(len(filenames)):
                fn = filenames[jj]
                df_temp = pd.read_csv(f'{out_path}{fp}/{fn}', index_col='Unnamed: 0')
                seq_temp = df_temp.loc[:, df_temp.columns[i]].values
                seq[:, jj] = seq_temp
            mean_estimation[:, i], mean_estimation[:, i + 3] = stderr(seq)
            std_estimation[:, i], std_estimation[:, i + 3] = std_stderr(seq)
        df_mean = pd.DataFrame(data=mean_estimation, columns=col_names)
        df_std = pd.DataFrame(data=std_estimation, columns=col_names)
        if save_file:
            df_mean.to_csv(f'{out_path}{fp}/mean_estimation.csv')
            df_std.to_csv(f'{out_path}{fp}/std_estimation.csv')
        else:
            print("No returns or results saved.")

def boot_process(out_path, col_names, nsubsets, r, save_file=True):
    """
    Postprocess the results using replicates.
    """
    fps = [fp for fp in os.listdir(out_path) if 'fix' in fp]
    print(fps)
    for fp in fps:
        print(fp)
        filenames = [f for f in os.listdir(f'{out_path}{fp}/') if 'repl' in f]
        std_estimation = np.zeros(shape = (nsubsets, 3 * 2))
        mean_estimation = np.zeros(shape = (nsubsets, 3 * 2)) # the number of columns is double the number of error metrics used
        for i in range(3):
            seq = np.zeros(shape=(nsubsets, r))
            seq_std = np.zeros(shape=(nsubsets, r))
            for jj in range(len(filenames)):
                fn = filenames[jj]
                df_temp = pd.read_csv(f'{out_path}{fp}/{fn}', index_col='Unnamed: 0')
                seq_temp = df_temp.loc[:, df_temp.columns[i]].values
                # import pdb; pdb.set_trace()
                seq_std_temp = seq_temp - df_temp.loc[:, df_temp.columns[i + 3]].values
                seq[:, jj] = seq_temp
                seq_std[:, jj] = seq_std_temp
            mean_estimation[:, i], mean_estimation[:, i + 3] = seq.mean(axis=1), np.std(seq, axis=1)
            std_estimation[:, i], std_estimation[:, i + 3] = seq_std.mean(axis=1), np.std(seq_std, axis=1)
        df_mean = pd.DataFrame(data=mean_estimation, columns=col_names)
        df_std = pd.DataFrame(data=std_estimation, columns=col_names)
        if save_file:
            df_mean.to_csv(f'{out_path}{fp}/mean_estimation.csv')
            df_std.to_csv(f'{out_path}{fp}/std_estimation.csv')
        else:
            print("No returns or results saved.")

def return_metric_samples(metric_cache, size, len_params, split_style, skip_numbers, num_replicates):
    if os.path.exists(metric_cache): 
        samples = np.loadtxt(metric_cache)
    else:
        samples = sample_repli(800, len_params, metric_cache, split_style = 'vertical', 
            skip_numbers = 1000, num_replicates = r)
    return samples
""" Generate a number of r=10 independent Sobol' sampling sequences"""
import os
import numpy as np
from SALib.sample import sobol_sequence

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

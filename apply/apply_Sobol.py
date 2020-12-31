"""Apply experiment with Sobol."""

import pandas as pd
import numpy as np
import json
import os
#sensitivity analysis and partial sorting 
from SALib.analyze.sobol import separate_output_values
# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from sa_rankings import sobol_ranking
from settings import SOBOL_DATA_DIR

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
cache_file = f'../output/replicates/sobol/sobol_test.json'
file_exists = os.path.exists(cache_file)
if file_exists:
    partial_order = sobol_ranking(cache_file, dummy=False)
else:
    partial_order, x_sobol, y_sobol = sobol_ranking(cache_file, dummy=False)
    # Obtain A and B metrics
    nstop = int(y_sobol.shape[0] / (x.shape[0] + 2))
    index_y = np.arange(0, x_sobol.shape[0])
    index_A, index_B, _, _ = separate_output_values(index_y, 
        problem['num_vars'], nstop, False)
    index_AB = np.append(index_A, index_B)
    y = y_sobol[index_AB]
    x = x_sobol[index_AB, :]
    xy_df = pd.DataFrame(data = x, index = np.arange(0, x.shape[0]), columns = problem['names'])
    xy_df.loc[:, 'y'] = y
    xy_df.to_csv(f'../output/replicates/sobol/satelli_samples.csv')
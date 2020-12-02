"""Apply experiment with PAWN."""

import pandas as pd
import numpy as np
from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.analyze import pawn
from SALib.sample import latin as sample_latin
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa

# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank
from SALib.plotting import bar
import matplotlib.pyplot as plt

from settings import PAWN_DATA_DIR

# add dummy parameter
def wrap_evaluate(x, a):
    y = evaluate(x[:, :-1], a)
    return y

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()
problem, len_params = add_dummy(True, problem)

N = 10000
x_sample = sample_latin.sample(problem, N)
Y = wrap_evaluate(x_sample, a)
Si = pawn.analyze(problem, x_sample, Y, stat='max')


df = pd.DataFrame.from_dict(Si)
df.set_index('names', inplace = True)

df = pd.read_csv('../output/pawn/pawn_max.csv', index_col = 'names')
df.sort_values(by='PAWNi', ascending=False, inplace=True)

ax = bar.plot(df)
plt.hlines(df.iloc[-1].sum(), xmin=0, xmax=22)

plt.savefig('../output/figure/pawn_dummy_max.png', format='png', dpi=300)

# df.to_csv('../output/pawn/pawn_max.csv')


# # ## test_sobol
# from SALib.sample import saltelli
# from SALib.analyze import sobol
# N = 2000
# x_sobol = saltelli.sample(problem, N) 
# y_sobol = wrap_evaluate(x_sobol, a)
# si_sob = sobol.analyze(problem, y_sobol)s

# sob = {'S1': si_sob['S1'], 'S1_conf': si_sob['S1_conf'], 
#     'ST': si_sob['ST'], 'ST_conf': si_sob['ST_conf']}
# df_sob = pd.DataFrame.from_dict(sob)
# # df_sob.set_index('names', inplace = True)
# ax = bar.plot(df_sob)
# ax.set_xticks(ticks = ax.ticks, labels = problem['names'])
# ax.set_xlables('SA indices')
# plt.hlines(df_sob.iloc[-1].sum(), xmin=0, xmax=22)
# plt.savefig('../output/figure/sobol_dummt_zero.png', format='png', dpi=300)
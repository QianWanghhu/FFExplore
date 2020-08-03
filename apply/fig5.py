"""Create figure 5 for paper."""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import seaborn as sns

# Import global path and file variables
from settings import *

# dot plot for type II

f_dir = MORRIS_DATA_DIR + '0.25/'
rate_names = ['mae', 'var', 'ppmc', 'mae_upper', 'var_upper', 
            'ppmc_upper', 'mae_lower', 'var_lower', 'ppmc_lower'] 
df_metric = pd.DataFrame(columns=rate_names)
metric_cols = df_metric.columns
for i in range(len(rate_names)):
    f_read = pd.read_csv('{}{}{}'.format(f_dir, rate_names[i], '.csv'))
    df_metric[metric_cols[i]] = f_read[f_read.columns[8]]

# obtain relative bias    
cols = df_metric.columns
col_upper = [col for col in cols if 'up' in col]
col_lower = [col for col in cols if 'low' in col]
for ii in range(len(col_upper)):
    df_metric[col_upper[ii]] = df_metric[col_upper[ii]] - df_metric[cols[ii]]
    df_metric[col_lower[ii]] = df_metric[cols[ii]] - df_metric[col_lower[ii]]

df_metric.fillna(value=0.0, inplace=True)
df_metric['ppmc'] = df_metric['ppmc'].apply(lambda x: 1 - x)
df_metric['var'] = df_metric['var'].apply(lambda x: np.abs(1 - x))
df_metric = df_metric.drop_duplicates('mae', keep='first')
df_metric = df_metric.iloc[::-1]
yerror = [[df_metric[col_lower[ii]].values, df_metric[col_upper[ii]].values] for ii in range(len(col_upper))]

# import the analytic variance 
fvariance = np.loadtxt('data/variance_frac.txt')
# total_variance = 2.755
index_fix = np.array([[20], [15, 16, 17, 18, 19], [14, 12, 13], 
                    [11], [8], [10, 9, 7, 6, 5, 4], [2, 3, 0, 1]])
variance_frac = fvariance[[len(list(flatten(index_fix[0:i+1])))-1 for i in range(index_fix.size-1)]] / 100
variance_frac = np.append(variance_frac, fvariance[-1])

sns.set_style('whitegrid')
fig = plt.figure(figsize=(6, 5))
# form x label
x = df_metric.index
num_in_groups = []
x_ticklabels = [''] + ['{}{}{}{}'.format(i+1, ' (', (21 - x[i]), ')') for i in range(len(x))]
df_metric.index = ([i+1 for i in range(len(x))])

conf_names = [col for col in rate_names if '_conf' in col]
colors = ['orchid', 'royalblue','chocolate']
ax = df_metric[rate_names[:3]].plot(kind='line', yerr=yerror, 
                                    linestyle='', color=colors)

x = df_metric.index
ax.plot(x, df_metric[rate_names[0]], 's', color=colors[0], ms=4, alpha=0.7, label='RMAE')
ax.plot(x, df_metric[rate_names[1]], '^', color=colors[1], ms=4, label='RV')
ax.plot(x, df_metric[rate_names[2]], 'o', ms=5, markerfacecolor='none',label='PPMC',
        markeredgecolor=colors[2], markeredgewidth=1.5)
          
ax.plot(x, variance_frac, 'd', ms=3, markerfacecolor='none',label='First-order variance',
        markeredgecolor='c', markeredgewidth=1.5, alpha=0.7)            
                                                                  
ax.axhline(y=0.06, xmin=0, xmax=6, linestyle='--', linewidth=1.2, color='dimgrey')
ax.tick_params(axis='both', labelsize=12)

ax.set_xlabel('Groups (numbers) of factors fixed', fontsize=12)
ax.set_ylabel('Value of error measures', fontsize=12)
ax.set_ylim(-0.03, 0.5)
ax.set_xlim(0.85, 7.15)

ax.set_xticklabels(x_ticklabels[:-1])
ax.legend(['RMAE', '(1 - RV)', '(1 - r)', '% decrease in variance'], loc='upper left', fontsize=10)
ax.text(1, 0.08, '6% (Threshold)', fontsize=10, color='dimgrey')
plt.savefig('{}{}{}'.format(FIGURE_DIR, 'fig5_variance_fix', '.jpg'), dpi=300, format='jpg')
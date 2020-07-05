"""Create figure 5 for paper."""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# dot plot for type II
f_dir = '../../../Research/G_func_ff/output/morris/revision/test/seed123/'
rate_names = ['mae', 'var', 'pearson', 'mae_up', 'var_up', 
            'pearson_up', 'mae_low', 'var_low', 'pearson_low'] 
df_metric = pd.DataFrame(columns=rate_names)
metric_cols = df_metric.columns
for i in range(len(rate_names)):
    f_read = pd.read_csv('{}{}{}'.format(f_dir, rate_names[i], '.csv'))
    df_metric[metric_cols[i]] = f_read[f_read.columns[8]]
# obtain relative bias    
cols = df_metric.columns
col_up = [col for col in cols if 'up' in col]
col_low = [col for col in cols if 'low' in col]
for ii in range(len(col_up)):
    df_metric[col_up[ii]] = df_metric[col_up[ii]] - df_metric[cols[ii]]
    df_metric[col_low[ii]] = df_metric[cols[ii]] - df_metric[col_low[ii]]

df_metric.fillna(value=0.0, inplace=True)
df_metric['pearson'] = df_metric['pearson'].apply(lambda x: 1 - x)
df_metric['var'] = df_metric['var'].apply(lambda x: np.abs(1 - x))
df_metric = df_metric.drop_duplicates('mae', keep='first')
# df_metric.drop([0], inplace=True)
df_metric = df_metric.iloc[::-1]
yerror = [[df_metric[col_low[ii]].values, df_metric[col_up[ii]].values] for ii in range(len(col_up))]

# import the analytic variance 
fvariance = np.loadtxt(f'../../../Research/G_func_ff/output/sobol/revision/cumulative_variance_ratio.txt', usecols=[0])
# fvariance = fvariance[::-1]
total_variance = 2.755
index_fix = np.array([[20, 16, 19], [15, 17, 18], [14], [12, 13], 
                    [11], [10, 9], [8, 7, 6, 5, 4], [2], [3, 0, 1]])
for i in range(index_fix.shape[0]):
    if i == 0:
        analytic_var = np.zeros(index_fix.shape[0])
        analytic_var[i] = fvariance[index_fix[i]].sum()
    else:
        index_fix[0].extend(index_fix[i])
        analytic_var[i] = fvariance[index_fix[0]].sum()
        print()
analytic_var = analytic_var / total_variance

sns.set_style('whitegrid')
fig = plt.figure(figsize=(6, 5))
# form x label
x = df_metric.index
num_in_groups = []
x_ticklabels = [''] + ['{}{}{}{}'.format(i+1, ' (', (21 - x[i]), ')') for i in range(len(x))]
df_metric.index = ([i+1 for i in range(len(x))])

conf_names = [col for col in rate_names if '_conf' in col]
colors = ['green', 'royalblue','chocolate']
ax = df_metric[rate_names[:3]].plot(kind='line', yerr=yerror, 
                                    linestyle='', color=colors) #df_metric[conf_names].values.T

x = df_metric.index
ax.plot(x, df_metric[rate_names[0]], 's', color=colors[0], ms=5, alpha=0.7, label='RMAE')
ax.plot(x, df_metric[rate_names[1]], '^', color=colors[1], ms=5, label='RV')
ax.plot(x, df_metric[rate_names[2]], 'o', ms=5, markerfacecolor='none',label='PPMC',
                                 markeredgecolor=colors[2], markeredgewidth=1.5)
ax.plot(x, analytic_var, '*', ms=3, markerfacecolor='none',label='First-order variance',
                                 markeredgecolor='red', markeredgewidth=1.5, alpha=0.7)                                                              
ax.axhline(y=0.06, xmin=0, xmax=6, linestyle='--', linewidth=1.2, color='dimgrey')
ax.tick_params(axis='both', labelsize=12)

ax.set_xlabel('Groups (numbers) of factors fixed', fontsize=12)
ax.set_ylabel('Value of error meausres', fontsize=12)
ax.set_ylim(-0.03, 0.5)
ax.set_xlim(0.85, 7.15)

ax.set_xticklabels(x_ticklabels)
ax.legend(['RMAE', '(1-RV)', '(1-PPMC)', 'Expected decrease in variance'], loc='upper left', fontsize=10)  # bbox_to_anchor=(1.05, 0.5)
ax.text(1, 0.08, '6% (Threshold)', fontsize=12, color='dimgrey')

# plt.title('Comparion of error measures', fontsize=12)
plt.savefig('{}{}{}'.format(f_dir, 'fig5_first_order', '.jpg'), dpi=300, format='jpg')
# import packages
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# dot plot for type II
f_dir = '../../../Research/G_func_ff/output/morris/revision/test/seed123/'
rate_names = ['mae', 'var', 'pearsonr', 'mae_conf', 'var_conf', 'pearsonr_conf'] 
df_metric = pd.DataFrame(columns=rate_names)
metric_cols = df_metric.columns
for i in range(len(rate_names)):
    f_read = pd.read_csv('{}{}{}'.format(f_dir, rate_names[i], '.csv'))
    df_metric[metric_cols[i]] = f_read[f_read.columns[8]]
# obtain relative bias    
df_metric.fillna(value=0.0, inplace=True)
df_metric['pearsonr'] = df_metric['pearsonr'].apply(lambda x: 1 - x)
df_metric['var'] = df_metric['var'].apply(lambda x: np.abs(1 - x))
df_metric = df_metric.drop_duplicates('mae', keep='first')
# df_metric.drop([0], inplace=True)

sns.set_style('whitegrid')
fig = plt.figure(figsize=(6, 5))
cols = df_metric.columns
df_metric = df_metric.iloc[::-1]
# form x label
x = df_metric.index
num_in_groups = []
x_ticklabels = [''] + ['{}{}{}{}'.format(i+1, ' (', (21 - x[i]), ')') for i in range(len(x))]
df_metric.index = ([i+1 for i in range(len(x))])

conf_names = [col for col in rate_names if '_conf' in col]
colors = ['green', 'royalblue','chocolate']
ax = df_metric[rate_names[:3]].plot(kind='line', yerr=df_metric[conf_names].values.T, 
                                    linestyle='', color=colors)

x = df_metric.index
ax.plot(x, df_metric[rate_names[0]], 's', color=colors[0], ms=5, alpha=0.7, label='RMAE')
ax.plot(x, df_metric[rate_names[1]], '^', color=colors[1], ms=5, label='RV')
ax.plot(x, df_metric[rate_names[2]], 'o', ms=5, markerfacecolor='none',label='PPMC',
                                 markeredgecolor=colors[2], markeredgewidth=1.5)
ax.axhline(y=0.06, xmin=0, xmax=6, linestyle='--', linewidth=1.2, color='dimgrey')
ax.tick_params(axis='both', labelsize=12)

ax.set_xlabel('Groups (numbers) of parameters fixed', fontsize=12)
ax.set_ylabel('RMAE, (1-RV) and (1-PPMC)', fontsize=12)
ax.set_ylim(-0.03, 0.5)
ax.set_xlim(0.85, 7.15)

ax.set_xticklabels(x_ticklabels)
ax.legend(['RMAE', 'RV', 'PPMC'], loc='upper left', fontsize=10)  # bbox_to_anchor=(1.05, 0.5)
ax.text(1, 0.08, '6% (Threshold)', fontsize=12, color='dimgrey')

# plt.title('Comparion of error measures', fontsize=12)
plt.savefig('{}{}{}'.format(f_dir, 'fig5', '.jpg'), dpi=300, format='jpg')
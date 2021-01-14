import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import SALib
# from SALib.plotting import bar
import seaborn as sns
# # read Sobol' sensitivity indices as dataframe
from settings import *
# from utils.test_function_setting import set_sobol_g_func
from pandas.core.common import flatten
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

##=================Plot the adaptive evaluation==============##
from settings import *
df = {}
filename = ['fix_9'] #'fix_9', 'fix_17','fix_1', , 'fix_21', 'fix_10',
fpath = '../output/genz/genz/'

for fn in filename:
    df[fn] = pd.read_csv(f'{fpath}{fn}/mean_estimation.csv', index_col = 'Unnamed: 0').iloc[0:80]
    df[fn].index = df[fn].index.astype('int')
    df[fn].index = (df[fn].index + 1) * 10

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
fs = 12
sns.set_style('whitegrid')
plt.rcParams['legend.title_fontsize'] = 12
lgd = [ii.split('_')[1] for ii in filename]
METRIC_NAME = [METRIC_NAME[0]]
metrics = [fn for fn in METRIC_NAME if not '_' in fn ]
colors = ['lightgreen', 'orange', 'cornflowerblue']
for ii in range(len(metrics)):
    metric = metrics[ii]
    k = 1
    for fn in filename:  
        df_plot = df[fn]
        lower = df_plot.loc[:, f'{metric}_mean'] - df_plot.loc[:, f'{metric}_std']
        upper = df_plot.loc[:, f'{metric}_mean'] + df_plot.loc[:, f'{metric}_std']
        # logy = True if ii == 0 else False
        ax = df_plot.loc[:, metric + '_mean'].plot(kind='line', marker='o', linewidth=1, style='-', ms=4, ax = axes, alpha=1.0, logy=False, color=colors[k])
        ax.scatter(df_plot.index, lower, marker = 'd', color= colors[k])
        ax.scatter(df_plot.index, upper, marker = 'd', color=colors[k])
        ax.vlines(df_plot.index, lower, upper, linestyle = '--', color=colors[k])
        k += 1
        # ax.hlines(0.94, df_plot.index[0], df_plot.index[-1], linestyle = '--', colors='grey')   
    ax.set_xlabel('Sample size', fontsize=14);
    ax.set_ylabel('RMAE', fontsize = 14);
    # ax.set_ylim(0.8, 1.2)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.legend(lgd, title='Number of factors fixed', fontsize = 14, ncol=2, bbox_to_anchor=(0.98, 0.2))
# plt.savefig('../output/genz/figure/fig_mean_vertical.tif', format = 'tif', dpi = 400)

##======================Plot the comparison of error metrics=======================##
df = {}
filename = ['fix_9', 'fix_10', 'fix_11']
fpath = '../output/genz/genz/'
for fn in filename:
    df[fn] = pd.read_csv(f'{fpath}{fn}/mean_estimation.csv', index_col = 'Unnamed: 0').iloc[9]
df_metric = pd.DataFrame.from_dict(df, orient = 'index')
new_index = [ind.split('_')[1] for ind in list(df_metric.index)]
df_metric.index = new_index
# df_metric
# obtain relative bias    
cols = df_metric.columns
df_metric.fillna(value=0.0, inplace=True)
df_metric['ppmc_mean'] = df_metric['ppmc_mean'].apply(lambda x: 1 - x)
df_metric['var_mean'] = df_metric['var_mean'].apply(lambda x: np.abs(1 - x))

# drop the first row due to RMAE > 0.40
cols = df_metric.columns
yerror = [df_metric.loc[:, col].values for col in cols[-3:]]
x = df_metric.index
# df_metric.index = ([str(21 - i) for i in x])
sns.set_style('whitegrid')
fig = plt.figure(figsize=(6, 5))
# form x label
num_in_groups = []
conf_names = [col for col in METRIC_NAME if '_conf' in col]
colors = ['orchid', 'royalblue','chocolate']
ax = df_metric[cols[:3]].plot(kind='line', yerr=yerror, linestyle='', color=colors)

x = df_metric.index
x_ticklabels = ['{}{}{}{}'.format(i+1, ' (', x[i], ')') for i in range(len(x))]

ax.plot(x, df_metric[METRIC_NAME[0]+'_mean'], 's', color=colors[0], ms=4, alpha=0.7, label='RMAE')
ax.plot(x, df_metric[METRIC_NAME[1]+'_mean'], '^', color=colors[1], ms=4, label='RV')
ax.plot(x, df_metric[METRIC_NAME[2]+'_mean'], 'o', ms=5, markerfacecolor='none',label='PPMC',
        markeredgecolor=colors[2], markeredgewidth=1.5)
                                                                           
ax.axhline(y=0.05, xmin=0, xmax=6, linestyle='--', linewidth=1.2, color='dimgrey')
ax.tick_params(axis='both', labelsize=12)

ax.set_xlabel('Numbers of factors fixed', fontsize=12)
ax.set_ylabel('Value of error measures', fontsize=12)
# ax.set_ylim(-0.03, 0.5)
# ax.set_xlim(0.85, 7.15)

ax.legend(['RMAE', '(1 - RV)', '(1 - r)'], loc='upper left', fontsize=10)
ax.text(0.1, 0.08, '5% (Threshold)', fontsize=10, color='k')
# plt.savefig('{}{}{}'.format('../output/genz/figure/', 'fig_compare_metrics', '.png'), dpi=300, format='png')




# Plot the changing error of full- and reduced- GP
file_path = '../output/genz/full_reduced_compare.csv'
error_comp = pd.read_csv(file_path, index_col = 'Unnamed: 0')
# error_comp = error_comp.reset_index().rename(columns = {'index': 'Sample size'})  
sns.set_style('whitegrid')
for ii in np.arange(1050, 1401, 50):
    error_comp.loc[ii, :] = None
error_comp['reduced'] = error_comp.reduced.shift(8)
ax = error_comp.plot(logy=True, xlabel='Sample size', ylabel='relative RMSE', marker = 'o', ms=5)
plt.savefig('../output/genz/GP_compare.png', format='png', dpi=300)

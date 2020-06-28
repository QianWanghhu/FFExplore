#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import matplotlib.ticker as ticker

# read result of the same metric
# select the result on the two rows which represent the identified fixing group
path = '../../../Research/G_func_ff/output/morris/revision/test/'
f_default = np.append([0, 0.1, 0.4, 0.5], np.linspace(0.2, 0.3, 11))
f_default.sort()
f_default = [str(round(i, 2)) for i in f_default]
f_default[0] = '0.0'
names = ['mae', 'var', 'pearsonr', 'mae_conf', 'var_conf', 'pearsonr_conf']
df = {}
for fn in names:
    df[fn] = pd.DataFrame(columns=['group1', 'group2'], index=f_default)
    for val in f_default:
        f_read = pd.read_csv('{}{}{}{}{}'.format(path, val, '/', fn, '.csv'))
        df[fn].loc[val, 'group1'] = f_read.loc[15, 'result_90']
        df[fn].loc[val, 'group2'] = f_read.loc[12, 'result_90']

# transform df from dict into dataframe with multiple columns
df = pd.concat(df, axis=1)
df.index = [float(i) for i in df.index]
df=df.astype('float')

def plot_shadow(col_name, ax, ylim=None):
    up_conf =  df[col_name] + df[f'{col_name}_conf']
    low_conf =  df[col_name] - df[f'{col_name}_conf']
    df[col_name].plot(kind='line', marker='o', linewidth=1, style='--', ms=3, ax=ax)
    ax.fill_between(df.index, low_conf['group1'], up_conf['group1'],color='lightsteelblue')
    ax.fill_between(df.index, low_conf['group2'], up_conf['group2'],color='moccasin')
    ax.set_xlim(-0.01, 0.51)
    if not (ylim==None):
        ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(['fix Group1', 'fix Group1 and Group2'], fontsize=16)
# End plot_shadow()
sns.set_style('whitegrid')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 7))
plot_shadow('mae', axes[0])
plot_shadow('var', axes[1])
fig.suptitle('Morris (n=90)', fontsize=20)
axes[1].set_xlabel('Default value',  fontsize=18)
plot_shadow('pearsonr', axes[2], [0.990, 1.010])

plt.savefig(f'{path}fig7.jpg', format='jpg', dpi=300)



# define function for visualizing the effect of different default values on error measures
def default_plot(data, ax, y_label):
    data.group1.plot(color='darkorange', style='--', linewidth=2, marker='o', ms=4)
    data.group2.plot(color='steelblue', style='--',linewidth=2, marker='o', ms=4)
    ax.set_ylabel(y_label, fontsize=16)
    xticks, xticklabels = plt.xticks()
    # shift half a step to the left
    # x0 - (x1 - x0) / 2 = (3 * x0 - x1) / 2
    xmin = (3*xticks[0] - xticks[1])/2.
    # shaft half a step to the right
    xmax = (3*xticks[-1] - xticks[-2])/2.
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(['fix Group1', 'fix Group1 and Group2'], fontsize=16)

sns.set()
sns.set_style('darkgrid')
fig = plt.figure(figsize=(24, 7))
ax1 = plt.subplot(131)
default_plot(df.mae, ax1, 'RMAE')


ax2 = plt.subplot(132)
default_plot(df.loc[:, 'var'], ax2, 'RV')
ax2.title.set_text('Morris (n=90)')
ax2.title.set_fontsize(18)

ax3 = plt.subplot(133)
default_plot(df.pearsonr, ax3, 'PPMC')
ax3.set_ylim(0.99, 1.01)
ax3. yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# set xlabel at the center of the figure
fig.text(0.5, 0.01, 'Default value', ha='center', fontsize=18);
# plt.savefig('{}{}'.format(path, 'default_value_eval_dots.jpg'), format='jpg', dpi=600)
# import packages
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# plot for figure type I
f_dir = ['../../../Research/G_func_ff/output/morris/revision/test/seed123/',
         '../../../Research/G_func_ff/output/sobol/revision/']
f_names = ['mae', 'mae_low','mae_up']
def f_read(filename):
    df = pd.read_csv(filename)
    df.drop(columns = 'Unnamed: 0', inplace=True)
    df = df.reindex(index=df.index[::-1]).reset_index(drop=True)
    return df

sample_size = ['result_90', 'result_800']
f_morris = f_read(f'{f_dir[0]}{f_names[0]}.csv')
mae_comp = pd.DataFrame(index=f_morris.index)
mae_comp['Morris'] = f_morris[sample_size[0]]
mae_comp['Sobol'] = f_read(f'{f_dir[1]}{f_names[0]}.csv').loc[:, sample_size[1]]
mae_comp['Morris_low'] = f_read(f'{f_dir[0]}{f_names[1]}.csv').loc[:, sample_size[0]]
mae_comp['Morris_up'] = f_read(f'{f_dir[0]}{f_names[2]}.csv').loc[:, sample_size[0]]
mae_comp['Sobol_low'] = f_read(f'{f_dir[1]}{f_names[1]}.csv').loc[:, sample_size[1]]
mae_comp['Sobol_up'] = f_read(f'{f_dir[1]}{f_names[2]}.csv').loc[:, sample_size[1]]

cols =  mae_comp.columns

fig = plt.figure(figsize=(10, 6))
sns.set_style('white')
palette = plt.get_cmap('Set1')
col_conf = [col for col in cols if '_conf' in col]
ax = mae_comp[cols[0:2]].plot(kind='line', linewidth=1)#[cols[0:2]]
ax.fill_between(mae_comp.index, mae_comp['Morris_low'], 
                mae_comp['Morris_up'],color='lightsteelblue', label=f'95% CIs for Morris')                
ax.fill_between(mae_comp.index, mae_comp['Sobol_low'], 
                mae_comp['Sobol_up'],color='moccasin', label=f'95% CIs for Sobol')

ax.plot(mae_comp['Sobol'][8:9], 'd', color='blue', alpha=0.5, ms=7,
        label='Number of fixed factors identified')
ax.axhline(y=0.06, xmin=0, xmax=21, linestyle='--', linewidth=1.2, color='dimgrey')
# ax.text(1, 0.08, '6% (Threshold)', fontsize=12, color='dimgrey')

ax.set_xticks(range(len(mae_comp)))
ax.set_xticklabels(np.arange(1, 22));
ax.set_xlabel('The number of factors fixed', fontsize=10)
ax.set_ylabel('RMAE (%)', fontsize=10)
leg = ax.legend(['Morris (1980)', 'Sobol (18400)',
            'Number of fixed factors identified',
            'Threshold (6%)',
            f'95% CIs for Morris',
            f'95% CIs for Sobol'], 
            fontsize = 6);
leg.set_title('GSA method (sample size)', prop={'size':8})            
plt.savefig(f'{f_dir[0]}fig6.jpg', dpi=300, format='jpg')
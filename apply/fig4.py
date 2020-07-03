import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import openpyxl

from utils.plots import bg_color, show_cells 

# import data
f_dir = '../../../Research/G_func_ff/output/morris/revision/test/seed123/'
f_name = 'mae_up.csv'
df = pd.read_csv('{}{}'.format(f_dir, f_name)).set_index('Unnamed: 0')
# df.to_csv(f'{f_dir}mae_up.csv', index=True)
# df = df[df.columns[1:12]]
df = df.rename(columns={i: i.split('_')[1] for i in df.columns})
columns = df.columns
df['No. of params fixed'] = list(range(22))[1:][::-1]
df.set_index('No. of params fixed', inplace=True)

# select columns in df to show
# df = df[df.columns[::2]]
# df = df[df.columns[2:]].drop(columns=df.columns[7:12]).drop(columns=df.columns[-9:])
df = df[df.columns[:11]]
columns = df.columns
df_test = {}
df_test['Number of trajectories'] = df

# styling the dataframe
values_dict = {'first_col': [0.015, 0.39],
            'varying': [0.02, 0.39],
            'transition': [0.015, 0.058, 0.39],
            'stable': [0.015, 0.058, 0.058]}
colors_dict = {'first_col': ['cornflowerblue', 'blue', 'pink'],
            'varying': ['cornflowerblue', 'blue' , 'pink'],
            'transition': ['cornflowerblue', 'green', 'pink'],
            'stable': ['cornflowerblue','green', 'pink']}# 
styles = [
    dict(selector="caption", props=[("text-align", "center"),
                                   ('font-size', '120%'),
                                   ('color', 'black')])
]
styled_table = (df.style.format("{:.1%}").\
    set_caption("Trajectory for Morris").\
    set_table_styles(styles).\
    applymap(bg_color,values = values_dict['first_col'], 
        colors = colors_dict['first_col'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[0]]).\
    applymap(bg_color, values = values_dict['varying'],
        colors = colors_dict['varying'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[1:6]]).\
    applymap(bg_color, values = values_dict['transition'],
        colors = colors_dict['transition'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[6]]).\
    applymap(bg_color, values = values_dict['stable'],
        colors = colors_dict['stable'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[7:]]).\
    applymap(bg_color,values = values_dict['first_col'], 
        colors = colors_dict['first_col'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[0]]).\
    applymap(bg_color, values = values_dict['varying'],
        colors = colors_dict['varying'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[1:6]]).\
    applymap(bg_color, values = values_dict['transition'],
        colors = colors_dict['transition'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[6]]).\
    applymap(bg_color, values = values_dict['stable'],
        colors = colors_dict['stable'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[7:]]).\
    apply(show_cells)
)
styled_table
# export styler into excel
styled_table.to_excel('{}{}'.format(f_dir, 'table_mae.xlsx'), engine='openpyxl');
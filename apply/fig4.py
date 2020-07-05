"""Create figure 4 for paper."""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import openpyxl

from utils.plots import bg_color, show_cells 

# Import global path and file variables
from settings import *

# import data
f_dir = os.path.join(MORRIS_DATA_DIR, 'test/seed123/')
f_name = 'mae_high.csv'
df = pd.read_csv('{}{}'.format(f_dir, f_name)).set_index('Unnamed: 0')
df = df.rename(columns={i: i.split('_')[1] for i in df.columns})
columns = df.columns
df['No. of params fixed'] = list(range(22))[1:][::-1]
df.set_index('No. of params fixed', inplace=True)

# select columns in df to show
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

styled_table = (df.style.format("{:.1%}").
    set_caption("Trajectory for Morris").
    set_table_styles(styles).
    applymap(bg_color,values = values_dict['first_col'], 
        colors = colors_dict['first_col'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[0]]).
    applymap(bg_color, values = values_dict['varying'],
        colors = colors_dict['varying'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[1:6]]).
    applymap(bg_color, values = values_dict['transition'],
        colors = colors_dict['transition'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[6]]).
    applymap(bg_color, values = values_dict['stable'],
        colors = colors_dict['stable'], 
        backg_color = True,
        subset=pd.IndexSlice[:, columns[7:]]).
    applymap(bg_color,values = values_dict['first_col'], 
        colors = colors_dict['first_col'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[0]]).
    applymap(bg_color, values = values_dict['varying'],
        colors = colors_dict['varying'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[1:6]]).
    applymap(bg_color, values = values_dict['transition'],
        colors = colors_dict['transition'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[6]]).
    applymap(bg_color, values = values_dict['stable'],
        colors = colors_dict['stable'], 
        backg_color = False,
        subset=pd.IndexSlice[:, columns[7:]]).
    apply(show_cells)
)

# export styler into excel
styled_table.to_excel('{}{}'.format(f_dir, 'table_mae.xlsx'), engine='openpyxl')
"""Convenience functions for plot creation."""

import pandas as pd
import numpy as np
import seaborn as sns
import re

def bg_color(dataframe, values, colors, backg_color=True):
    """
    Set background color according to the value satisfying the criteria.
    """ 
    dataframe = np.abs(dataframe)
    if len(values) > 2:
        if dataframe <= values[0]:
            color = colors[0]
        elif dataframe < values[1]:
            color = colors[1]
        elif dataframe < values[2]:
            color = colors[2]
        else:
            color = colors[-1]
    else:
        if dataframe <= values[0]:
            color = colors[0]    
        elif dataframe < values[1]:
            color = colors[1]
        else:
            color = colors[-1]

    if backg_color == True:
        return 'background-color: %s' % color
    else:
        return 'color: %s' % color


def show_cells(dataframe):
    """
    Function for showing cell values.
    """
    index_list = []
    unique_values = dataframe.unique()
    for unique_value in unique_values:
        index_list.append(dataframe[dataframe==unique_value].index.tolist()[0])

    index_all = dataframe.index
    tf = [False] * len(index_all)

    k = 0
    for i in index_all:
        if i in index_list:
            tf[k] = True
        k += 1

    return ['color: white' if v else '' for v in tf]

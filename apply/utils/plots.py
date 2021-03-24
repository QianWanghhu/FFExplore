"""Convenience functions for plot creation."""

import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

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


def match_ci_bounds(data, metric):
    cols = data.columns
    if (metric + '_std') in cols:
        lower = data.loc[:, f'{metric}_mean'] - data.loc[:, f'{metric}_std']
        upper = data.loc[:, f'{metric}_mean'] + data.loc[:, f'{metric}_std']
    elif (metric + '_lower') in cols:
        lower = data.loc[:, f'{metric}_lower']
        upper = data.loc[:, f'{metric}_upper']
    else:
        raise AssertionError('Cannot locate the CIs of the metric')
    return lower, upper

def plot_metric_sampling(df_plot, fix_lists, metric, mean_lab, xlab, ylab, xtick_locator, 
    fs, color, lgd=None, ax=None, legd_loc=None, legd_bbox=None, alpha=None, **kwags):
    lower, upper = match_ci_bounds(df_plot, metric)
    ax = df_plot.loc[:, mean_lab].plot(**kwags, ax=ax, color= color)
    ax.scatter(df_plot.index, lower, marker = 'd', color= color, alpha=alpha)
    ax.scatter(df_plot.index, upper, marker = 'd', color=color, alpha=alpha)
    ax.vlines(df_plot.index, lower, upper, linestyle = '--', color=color, alpha=alpha)
    ax.set_xlabel(xlab, fontsize = fs);
    ax.set_ylabel(ylab, fontsize = fs);
    # ax.set_ylim(0.8, 1.2)
    ax.xaxis.set_major_locator(MultipleLocator(xtick_locator))
    plt.setp(ax.get_xticklabels(), fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)
    if not (lgd==None):
        if legd_loc==None:
            leg = ax.legend(lgd, fontsize = fs, ncol=2, bbox_to_anchor=legd_bbox)
        else:
            leg = ax.legend(lgd, fontsize = fs, ncol=2, loc=legd_loc)#loc='upper right'bbox_to_anchor=(0.65, 0.15)
        leg.set_title('Number of factors fixed',prop={'size':fs})
    return ax
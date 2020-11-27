"""Apply experiment for Oakley function."""

from toposort import toposort, toposort_flatten
import json
import os

#sensitivity analysis and partial sorting 
from SALib.sample import morris as sample_morris
from SALib.analyze import morris as analyze_morris
from SALib.sample import latin as sample_latin
from SALib.test_functions.oakley2004 import evaluate
# import settings for Sobol G-function and returns necessary elements
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.group_fix import group_fix
from utils.partial_sort import to_df, partial_rank
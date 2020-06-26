import numpy as np
from .Sobol_G import g_func

# define efficients a and x variables according to Sheikholeslami (2019)
# start
def set_sobol_g_func():
    """
    Function used to set sobol G-function.
    Parameters:
    ==========

    Returns:
    ========
    a : coefficients of Sobol G-function
    x : input variables
    len_params : lenght of parameters
    problem : Problem defined for sensitivity analysis
    """
    a = np.zeros(21)
    x = np.zeros(21) 

    # a[0:2] = 0
    # a[2:9] = [0.005, 0.020, 0.040, 0.060, 0.08, 0.090, 1]
    # a[9:16] = 2
    # a[16:24] = [2.10, 2.25, 2.75, 3, 3.10, 3.15, 3.25, 3.50]
    # a[24:30] = 9
    # a[30:44] = [8, 8.5, 9, 10, 10.5, 11, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16]
    # a[44:] = [70, 75, 80, 85, 90, 99]

    a[0:2] = 0
    a[2:4] = [0.005, 0.090]
    a[4:7] = 2
    a[7:11] = [2.10, 2.75, 3, 3.15]
    a[11:15] = [8, 13, 13.5, 16]
    a[15:] = [70, 75, 80, 85, 90, 99]

    x_names = ['x' + str(i+1) for i in range(21)]
    len_params = len(x_names)
    x_bounds = np.zeros((21, 2))
    x_bounds[:, 0] = 0
    x_bounds[:, 1] = 1

    problem = {
        'num_vars': len(x),
        'names': x_names,
        'bounds': x_bounds
        }
    return a, x, x_bounds, x_names, len_params, problem
# End
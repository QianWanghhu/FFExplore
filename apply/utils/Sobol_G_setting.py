import numpy as np

# define efficients a and x variables according to Sheikholeslami (2019)
# start
def set_sobol_g_func():
    """
    Set up Sobol G-function for experiment.

    Returns
    ----------
    a : np.array,
        Coefficients of Sobol G-function

    x : np.array,
        input variables

    len_params : int,
        number of parameters

    problem : dict,
        SALib problem spec (for sensitivity analysis)
    """
    a = np.zeros(21)
    x = np.zeros(21) 

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
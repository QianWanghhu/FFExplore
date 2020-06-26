"sobol g-function for sensitivity analysis"
import numpy as np


def g_func(values, a):
    """Sobol g-function 
    Parameters
    ==========
    values: ndarray, input variables
    a: list of coefficients.
    Return
    ======
    y: scalar of g-function result.
    """

    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
    if a is None:
        raise TypeError("The argument `a` must be given as a numpy ")

    ltz = values < 0
    gto = values > 1

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than zero")
    elif gto.any() == True:
        raise ValueError("Sobol G function called with values greater than one")

    Y = np.ones([values.shape[0]])

    len_a = len(a)
    for i, row in enumerate(values):
        for j in range(len_a):
            x = row[j]
            a_j = a[j]
            Y[i] *= (np.abs(4 * x - 2) + a_j) / (1 + a_j)

    return Y



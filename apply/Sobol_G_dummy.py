"""Use dummy parameter for screening."""

# import settings for Sobol G-function and returns necessary elements
import numpy as np
from utils.Sobol_G_setting import set_sobol_g_func, add_dummy
from utils.partial_sort import to_df, partial_rank
from utils.group_fix import group_fix, index_fix, results_exist
from utils.group_fix import evaluate_wrap
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions.Sobol_G import evaluate, total_sensitivity_index as total_sa, _total_variance
from settings import MORRIS_DATA_DIR
from SAFEpython.VBSA import vbsa_indices, vbsa_resampling
from SAFEpython.lhcube import lhcube
from scipy.stats import norm

a, x, x_bounds, x_names, len_params, problem = set_sobol_g_func()

N = 4000
# x_saltelli = saltelli.sample(problem, N)
# y_saltelli = evaluate(x_saltelli, a)
# sa = sobol.analyze(problem, y_saltelli)

# A, B, AB, BA = sobol.separate_output_values(y_saltelli, problem['num_vars'], N, True)
# AB = AB.reshape(AB.size)

X, d = lhcube(2*N, problem['num_vars'], nrep=5)
XA, XB, XAB = vbsa_resampling(X)
A, B, AB = evaluate(XA, a), evaluate(XB, a), evaluate(XAB, a)

Si, STi, Sidummy, STdummy = vbsa_indices(A, B, AB, problem['num_vars'], dummy=True, Nboot=500)
Z = norm.ppf(0.5 + 0.95 / 2)
ST_conf = Z * STdummy.std(ddof=1)
STdummy.mean(axis=0)

# calculate the analytic sensitivity indices
vd = np.array([1/3/ (i +1)**2 for i in a])
vtd = np.zeros(vd.shape[0])
total_vtd = np.prod([i + 1 for i in vd])
for i in range(vtd.shape[0]):
    vtd[i] = vd[i] * total_vtd / (vd[i] + 1) / _total_variance(a)
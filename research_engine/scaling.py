import numpy as np

EPS = 1e-12

def minmax(a):
    a = np.asarray(a, dtype=float)
    mn, mx = a.min(axis=0), a.max(axis=0)
    return (a - mn) / (mx - mn + EPS), mn, mx

def inv_scale(a, mn, mx):
    return np.asarray(a) * (mx - mn + EPS) + mn

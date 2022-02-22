import numpy as np
from scipy import signal


def get_circular_ss(x, indices):
    n = len(x)
    x_ = list(x)
    ret = np.zeros((len(indices),n))
    for i, ind in enumerate(indices):
        ret[i] = np.array(x_[n-ind:]+x_[:n-ind])
    return ret


def get_circular(x):
    n = len(x)
    x_ = list(x)
    ret = np.zeros((n,n))
    for i in range(n):
        ret[i] = np.array(x_[n-i:]+x_[:n-i])
    return ret


def get_h(f, T, **kwargs):
    if f == 'triangle':
        h = np.array([0.5,0.25]+[0]*(T-3)+[0.25])
    elif f == 'gaussian':
        g = signal.gaussian(T, std=kwargs['std'])
        h = np.concatenate((g[T//2:], g[:T//2])) / g.sum()
    elif f == 'identity':
        h = np.concatenate(([1.0], [0]*(T-1)))
    return h

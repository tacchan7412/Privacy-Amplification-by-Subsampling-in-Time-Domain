import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.special import comb


np.random.seed(seed=12345)
rng = np.random.default_rng(12345)


def dft_gaussian(x, eps, delta, max_l2sens, k=21):
    # (eps,delta)-DP extension for algorithm by Rastogi and Nath (2010)
    T = len(x)
    f = np.fft.fft(x)
    f_rr = gaussian(f.real, eps/2, delta/2, np.sqrt(T)*max_l2sens)
    f_ri = gaussian(f.imag, eps/2, delta/2, np.sqrt(T)*max_l2sens)
    f_r = f_rr+complex(0,1)*f_ri
    fp = np.concatenate((f_r[:k//2+1], [0]*(T-k), f_r[-k//2+1:]))
    assert len(fp) == T
    xp = np.fft.ifft(fp)
    return xp
    

def srank(A):
    fnorm = np.linalg.norm(A, ord='fro')  # frobenius norm
    norm = np.linalg.norm(A, ord=2)  # largest sing. value
    return (fnorm / norm)**2


def srank_circular(h):
    return sum(h**2) * len(h)


def find_alpha_log(eps, half_delta, sr, k, L):
    def func(alpha):
        gamma = k * (alpha**2) - 1
        tmp = np.log(2*sr)
        tmp += (gamma - (1+gamma)*np.log(1+gamma)) / (k*L)
        tmp += np.log(np.exp(eps/alpha) - np.exp(eps))
        return tmp - np.log(half_delta)
    root = fsolve(func, [0.99], xtol=10**(-6))[0]
    return root, func(root)


def get_subsample_ind(T, k):
    ss = 1*(rng.random(T) < 1/k)
    ss_ind = []
    for i, s in enumerate(ss):
        if s == 1: ss_ind.append(i)
    return ss_ind


def ss_interpolate(y_ss, ss_ind, T, kind='linear'):
    # use the kind for [min(ss_ind), max(ss_ind)]
    # use the nearest for other regions
    f = interpolate.interp1d(ss_ind, y_ss, kind=kind)
    x_new = np.arange(min(ss_ind), max(ss_ind)+1)
    y_new = f(x_new)
    res = np.concatenate(([y_new[0]]*min(ss_ind), y_new, 
                          [y_new[-1]]*(T-max(ss_ind)-1)))
    return res


def poisson(n,m,p):
    return comb(n, m) * (p**m) * ((1-p)**(n-m))


def f_poisson(Ip, I, k):
    ret = 0
    for i in range(Ip+1, I+1):
        ret += poisson(I, i, 1/k)
    return ret


def find_Ip(I, eps, half_delta, k):
    # naive version, complexity can be improved
    ret = I
    for i in range(1, I+1):
        deltap = f_poisson(i, I, k)
        tmp = deltap * (np.exp(np.sqrt(I/i)*eps)-np.exp(eps))
        if tmp < half_delta:
            ret = i
            break
    return ret


def ss_gaussian(x, eps, delta, I, k, interpolate_kind='linear',
                smooth='', smooth_window=20):
    T = len(x)
    Ip = find_Ip(I, eps, delta/2, k)
    sens_s = np.sqrt(Ip)
    ss_ind = get_subsample_ind(T, k)
    x_ss = x[ss_ind]
    z_ss = gaussian(x_ss, eps, delta/2, sens_s)
    ret = ss_interpolate(z_ss, ss_ind, T, kind=interpolate_kind)
    if smooth != '':
        ret = apply_smoothing(ret, smooth_window, smooth)
    return ret


def ssf_gaussian(x, A, eps, delta, max_sens, k,
                 sr=None, L=None, interpolate_kind='linear',
                 smooth='', smooth_window=20):
    # our (eps, delta)-DP mechanism
    # based on Gaussian mechanism
    T = len(x)
    if sr is None:
        sr = srank(A)
    if L is None:
        L = sum(A[0]**2)
    alpha, residual = find_alpha_log(eps, delta/2, sr, k, L)

    # in total (eps, delta/2+res_delta)-DP
    # which nearly equals to (eps, delta)-DP
    rerr = np.abs(residual)/(delta/2)
    assert rerr < 10**(-3), f'relative error of deltas is {rerr:.3g} (alpha={alpha})'
    sens_s = alpha * max_sens

    y = A@x
    ss_ind = get_subsample_ind(T, k)
    y_ss = y[ss_ind]
    z_ss = gaussian(y_ss, eps, delta/2, sens_s)
    ret = ss_interpolate(z_ss, ss_ind, T, kind=interpolate_kind)
    if smooth != '':
        ret = apply_smoothing(ret, smooth_window, smooth)
    return ret


def gaussian(x, eps, delta, sens):
    # (eps, delta)-DP mechanism
    size = x.shape
    c = np.sqrt(2*np.log(1.25/delta))
    sigma = c * sens / eps
    noise = rng.normal(0, sigma, size)
    return x + noise

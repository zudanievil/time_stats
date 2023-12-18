#! /usr/bin/env python3
import sys, os
import typing as _t

import numpy as np
from scipy import signal as s_signal

if __name__ == "__main__":
    import matplotlib.pyplot as plt
else:
    plt = "for tests, assign `matplotlib.pyplot` to this global variable"

PIx2 = np.pi * 2

def soft_slice(length, begin, end) -> slice:
    begin = 0 if begin < 0 else begin
    end = length if end > length else end
    return slice(begin, end)

_reduce_t = _t.Callable[[np.ndarray], float]

def reduce_binned(fn: _reduce_t, a: np.ndarray, bin_size: int, no_trim=True) -> np.ndarray:
    l = len(a)
    if no_trim:
        stop = l
        n = l // bin_size + bool(l % bin_size)
    else:
        n = l // bin_size
        stop = n * bin_size
    out = np.empty(n, float)
    j=0
    for i in range(0, stop, bin_size):
        out[j] = fn(a[soft_slice(l, i, i+bin_size)])
        j+=1
    return out

def reduce_periodic(fn, a: np.ndarray, period: int, no_trim=True) -> np.ndarray:
    l = len(a)
    trim_len = l % period
    trim = l - trim_len
    
    if trim_len and not no_trim:
        a = a[:trim]
    out = np.empty(period, float)
    for i in range(0, period):
        out[i] = fn(a[i::period])
    return out

def binned_avg(a: np.ndarray, bin_size: int, no_trim=True) -> np.ndarray:
    return reduce_binned(np.mean, a, bin_size, no_trim=no_trim)
    
def periodic_avg(a: np.ndarray, period: int, no_trim=True) -> np.ndarray:
    return reduce_periodic(np.mean, a, period, no_trim=no_trim)

def conv_avg(y: np.ndarray, n: int) -> np.ndarray:
    'sliding window average by convolution. n is window size'
    kern = np.full(shape=n, fill_value=1/ n)
    return np.convolve(y, kern, mode='same')

def _test_visual_binned_periodic_avg():
    _xticks = np.arange(0, 24*2 + 7, 6)
    _x = np.arange(0., 24.*2 + 6, 0.5)
    _y = np.cos((_x - 12) * PIx2/ 24)
    _y[-12:] = np.nan # check that period actually affects periodic part

    plt.scatter(_x, _y)

    _y_period = periodic_avg(_y, period=48)
    _x_period = _x[:48]
    plt.plot(_x_period, _y_period, c="cyan", ls="none", marker="s", markersize=3)

    _y_bin = binned_avg(_y, bin_size=5)
    _x_bin = binned_avg(_x, bin_size=5)

    plt.plot(_x_bin, _y_bin, ls="none", c="red", marker="o") #zorder=3)
    plt.xticks(_xticks, _xticks)
    plt.show()

def interdaily_stability(data: np.ndarray, hour_avg: np.ndarray) -> float:
    m = data.mean()
    num = np.square(hour_avg - m).sum() 
    denom = np.square(data - m).sum()
    return (num / denom) * (len(data) / 24)

IS = interdaily_stability

def __IS_test():
    """
    should be 1.0 for perfect autocorrelation, 
    0 for white noise (precision depends on sampling rate)
    """
    AVG = periodic_avg
    
    x = np.arange(0., 24*10, 1.)
    sin24 = np.cos(2*np.pi/24 * x)
    cos24 = np.sin(2*np.pi/24 * x)
    correlated = np.tile(np.random.randn(24), 10)
    white = np.random.randn(1000)
    
    IS_sin = IS(sin24, AVG(sin24, period=24))
    IS_cos = IS(cos24, AVG(cos24, period=24))
    IS_corr = IS(correlated, AVG(correlated, period=24))
    IS_white = IS(white, AVG(white, period=24))
    assert abs(IS_sin - 1.0) < 0.05, abs(IS_sin - 1.0)
    assert abs(IS_cos - 1.0) < 0.05, abs(IS_cos - 1.0) 
    assert abs(IS_corr - 1.) < 0.05, abs(IS_corr - 1.)
    assert abs(IS_white - 0.0) < 0.1, abs(IS_white - 0.0)

def intradaily_variability(x: np.ndarray) -> float:
    """
    refer to
    (Van Someren et al. 1999)
    https://doi.org/10.3109/07420529908998724
    """
    mean = np.mean(x)
    numerator = np.sum(np.diff(x)**2)/(len(x) - 1)
    denominator = np.sum((x - mean)**2)/len(x)
    return numerator/denominator

IV = intradaily_variability

def __IV_test():
    """
    should be 0.0 for sine/cosine, and 2.0 for white noise 
    (precision depends on sampling rate)
    """
    x = np.linspace(0., 10, 200)
    sin = np.sin(x)
    cos = np.cos(x)
    white = np.random.randn(1000)
    assert abs(IV(sin) - 0.0) < 0.05, abs(IV(cos) - 0.0)
    assert abs(IV(cos) - 0.0) < 0.05, abs(IV(cos) - 0.0) 
    assert abs(IV(white) - 2.0) < 0.15, abs(IV(white) - 2.0)


def _test_interdaily_stability_trends():
    def _calc() -> tuple[np.ndarray, np.ndarray]:
        _x = np.arange(0., 24.*2 + 6, 0.5)
        _y_trend = np.cos((_x - 12) * PIx2/ 24)
        n_repeats = 5
        n_pts = 50
        noise_scales = np.linspace(np.zeros(n_repeats), np.ones(n_repeats), n_pts).ravel()
        
        stability = []
        for scale in noise_scales:
            noise = scale * np.random.randn(*_x.shape)
            _y = _y_trend + noise
            _y_hour_avg = periodic_avg(binned_avg(_y, bin_size=2), period=24)
            _stab = interdaily_stability(_y, _y_hour_avg)
            stability.append(_stab)
        return noise_scales, np.array(stability)

    _ticks = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(5, 5))
    plt.scatter(*_calc())
    plt.gca().set_aspect('equal', 'box')
    plt.xticks(_ticks)
    plt.yticks(_ticks)
    plt.title("interdaily stability @ (noise_amp/trend_amp)")
    plt.grid()
    plt.show()


def intradaily_variability_subsampled(data: np.ndarray, subsampling_period: int) -> float:
    ivs = reduce_periodic(intradaily_variability, data, subsampling_period)
    return ivs.mean()

IVsub = intradaily_variability_subsampled

# def intradaily_variability(data: np.ndarray, subsampling_period: int) -> float:
#     ivs = reduce_periodic(__iv_square_diff_sum, data, subsampling_period)
#     iv = ivs.mean() * (subsampling_period / (subsampling_period-1))
#     return iv

# def __iv_square_diff_sum(bin_: np.ndarray) -> float:
#     sq_diff = np.square(np.diff(bin_)).sum()
#     mss = np.square(bin_ - bin_.mean()).sum()        
#     return sq_diff / mss

def _test_intradaily_variability_trends():
    def _calc() -> tuple[np.ndarray, dict[int, np.ndarray]]:
        _x = np.arange(0., 24.*2 + 6, 0.5)
        _y_trend = np.cos((_x - 12) * PIx2/ 24)
        n_repeats = 5
        n_pts = 50
        noise_scales = np.linspace(np.zeros(n_repeats), np.ones(n_repeats), n_pts).ravel()
        subsampling_periods = [3, 5, 7, 9]

        ivs_at_periods = {}
        for period in subsampling_periods:
            ivs = []
            for scale in noise_scales:
                noise = scale * np.random.randn(*_x.shape)
                _y = _y_trend + noise
                iv = IVsub(_y, period)
                ivs.append(iv)
            ivs_at_periods[period] = np.array(ivs)
        return noise_scales, ivs_at_periods
    
    noise_scales, ivs_at_periods = _calc()
    
    _ticks = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(5, 5))
    for period, iv in ivs_at_periods.items():
        plt.scatter(noise_scales, iv, label=f"period={period}", alpha=0.6)
    plt.xticks(_ticks)
    plt.title("intradaily variability @ (noise/trend)")
    plt.legend()
    plt.grid()


def relative_amplitude(hour_binned: np.ndarray) ->dict[str, float]:
    """
    `return {"M10": M10, "L5": L5, "RA": RA}`
    calculate maximum over 10h mean (M10),
    minimum over 5h mean (L5), 
    relative amplitude RA=(M10-L5)/(M10+L5)  
    """
    assert len(hour_binned) == 24
    cycled = np.tile(hour_binned, 3)[12:-12] # to prevent boundary effects
    M10 = np.max(np.convolve(cycled, [1/10]*10, 'valid'))
    L5 = np.min(np.convolve(cycled, [1/5]* 5, 'valid'))
    RA = (M10 - L5)/(M10 + L5)
    return {"M10": M10, "L5": L5, "RA": RA}

RelAmp = relative_amplitude


def closest(a: np.ndarray, b) -> int:
    'return argmin(abs(a - b))'
    return np.argmin(np.abs(a - b))

def PoV(a: np.ndarray, frequency: float, first_harmonic_range: tuple[float, float], n_harmonics=4) -> float:
    f, s = s_signal.periodogram(a, frequency)
    b, e = first_harmonic_range
    full_auc = a.var(ddof=1) 
    # ^^^ approximately equals the integral ` s[1:] @ np.diff(f) `
    harmonic_auc = 0
    for har in range(1, n_harmonics+1):
        hb = har*b
        he = har*e
        ib = closest(f, hb)
        ie = 1+ closest(f, he)
        si = s[ib:ie]
        fi = f[ib-1:ie]
        assert len(si)>0 and len(fi)>1, f"{si=}, {fi=} {har=}"
        harmonic_auc += si @ np.diff(fi)
    return harmonic_auc / full_auc

def conv_diff(a: np.ndarray, window_size: int) -> float:
    kern = np.full(window_size, 1/window_size)
    asmooth = np.convolve(a, kern, mode='same') # we'll truncate down below
    hw = window_size//2
    sq_diff = np.square(a[hw:-hw]-asmooth[hw:-hw])
    return sq_diff.sum()/len(sq_diff)

def petrosian_fractal_dimension(a: np.ndarray) -> float:
    """
    very weird measure. `1. < pfd < 1.072 when len(a) == 200` generally speaking, it's
    somewhere between 1 and 1.14, depending on the len(a).
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070217/
    https://sci-hub.se/https://doi.org/10.1109/CBMS.1995.465426
    """
    diff_sign = np.sign(np.diff(a))
    n_extrema = np.sum(diff_sign[1:] != diff_sign[:-1])
    n = len(a)
    logn = np.log10(n)
    pfd = logn / (logn + np.log10(n/(n+0.4*n_extrema)))
    return pfd
  
PFD = petrosian_fractal_dimension


def higuchi_fractal_dimension(X, Kmax):
    """ https://github.com/forrestbao/pyeeg/blob/master/pyeeg/fractal_dimension.py"""
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(float(1) / k), 1])
#     print(f"{L=}")
#     print(f"{x=}")

    (p, _, _, _) =np.linalg.lstsq(x, L, rcond=None)
    return p[0]

HFD = higuchi_fractal_dimension


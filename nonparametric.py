#! /usr/bin/env python3
import sys, os
import typing as _t

import numpy as np

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


def _test_binned_periodic_avg():
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

def _test_interdaily_stability_perfect_trend():
    _xticks = np.arange(0, 24*2 + 7, 6)
    _x = np.arange(0., 24.*2 + 6, 0.5)
    _y = np.cos((_x - 12) * PIx2/ 24)

    _y_hour_avg = periodic_avg(binned_avg(_y, bin_size=2), period=24)
    _x_hour_avg = _x[:48:2]

    plt.plot(_x, _y, ls="none", marker="o", markersize=6)
    plt.plot(_x_hour_avg, _y_hour_avg, ls="none", marker="o", markersize=6)

    _stab = interdaily_stability(_y, _y_hour_avg)
    plt.annotate(f"stab={_stab:.3f}", (0, 1))
    plt.show()

def _test_interdaily_stability_white_noise():
    _xticks = np.arange(0, 24*2 + 7, 6)
    _x = np.arange(0., 24.*2 + 6, 0.5)
    _y = np.random.randn(*_x.shape)

    _y_hour_avg = periodic_avg(binned_avg(_y, bin_size=2), period=24)
    _x_hour_avg = _x[:48:2]

    plt.plot(_x, _y, ls="none", marker="o", markersize=6)
    plt.plot(_x_hour_avg, _y_hour_avg, ls="none", marker="o", markersize=6)

    _stab = interdaily_stability(_y, _y_hour_avg)
    plt.annotate(f"stab={_stab:.3f}", (0, 1))
    plt.show()

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


def intradaily_variability(data: np.ndarray, subsampling_period: int) -> float:
    ivs = reduce_periodic(__iv_square_diff_sum, data, subsampling_period)
    iv = ivs.mean() * (subsampling_period / (subsampling_period-1))
    return iv

def __iv_square_diff_sum(bin_: np.ndarray) -> float:
    sq_diff = np.square(np.diff(bin_)).sum()
    mss = np.square(bin_ - bin_.mean()).sum()        
    return sq_diff / mss

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
                iv = intradaily_variability(_y, period)
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

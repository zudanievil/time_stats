"""
for detrended fluctuation analysis
"""

from typing import (
    NamedTuple as _NT, 
    Callable as _Fn, 
    Tuple as _Tup, 
    Optional as _Opt,
)
import numpy as _np
from numpy.typing import NDArray as _Arr


def root_mean_square(x: _Arr[float]) -> float:
    """:return sqrt(sum(x**2)/len(x))"""
    return _np.sqrt(_np.sum(x**2)/len(x))


def regression_poly(
        x: _Arr[float],
        y: _Arr[float],
        order: int = 1,
) -> _Tup[_Arr[float], _Arr[float]]:
    """
    compute polynomial regression.
    :param x: 1d, independent variable
    :param y: 1d, dependent variable
    :param order: polynomial order
    :returns: (coefficients, design_matrix)
    """
    if order < 1:
        raise ValueError("`order` < 1")

    D = [_np.ones_like(x), x]  # D -- design matrix
    for i in range(2, order+1):
        D.append(x**i)
    D = _np.stack(D, axis=1)
    coef = _np.linalg.lstsq(D, y, rcond=None)[0]
    # DT = D.T
    # coef = _np.linalg.inv(DT @ D) @ DT @ y  # closed form least sqares solution
    return coef, D  # design matrix is often useful for computing y_predicted, etc


def regression_poly_detrend(y: _Arr[float], x: _Opt[_Arr[float]] = None, order: int = 1) -> _Arr[float]:
    """
    compute polynomial regression for y onto x;
    substract the y_predicted from y.
    if x is None, uses numpy.arange
    """
    x = _np.arange(len(y)) if x is None else x
    coef, D = regression_poly(x, y, order=order)
    y_pred = D @ coef
    return y - y_pred


class DFA_Result(_NT):
    weight: float
    bias: float
    rms: _np.ndarray


class DFA(_NT):
    """
    Detrended Fluctuation Analysis
    #1 mean-center the data
    #2 compute cumulative sum
    #3 split data into chunks of size given by `window_sizes` (integer array, sorted)
    #4 detrend each chunk using `detrend` and `detrend_kwargs
    #5 compute rms for each chunk
    #6 average rms across chunks of the same size
    #7 regress log(rms_for_different_window_sizes) onto log(window_sizes)
    return rms values and regression coefficients
    """
    window_sizes: _np.ndarray
    detrend: _Fn[[_np.ndarray], _np.ndarray] = regression_poly_detrend
    detrend_kwargs: dict = dict(order=1)

    def __call__(self, x: _np.ndarray) -> DFA_Result:
        x = x - x.mean()  # 01 mean-center
        x = _np.cumsum(x)  # 02 cumsum
        mean_rms_arr = _np.empty(shape=(len(self.window_sizes),), dtype=float)
        # this array will accumulate our mapping results

        for ws_no, ws in enumerate(self.window_sizes):
            l = len(x) // ws
            chunks = x[:l * ws].reshape((l, ws))  # 03 split into chunks
            chunks_detrended = _np.apply_along_axis(
                self.detrend, axis=-1, arr=chunks,
            )  # 04 detrend
            rms = _np.apply_along_axis(
                root_mean_square, arr=chunks_detrended, axis=-1
            )  # 05 compute rms
            mean_rms_arr[ws_no] = rms.mean()

        coef, _ = regression_poly(x=_np.log(self.window_sizes), y=_np.log(mean_rms_arr), order=1)
        return DFA_Result(weight=coef[1], bias=coef[0], rms=mean_rms_arr)


class Test:
    # this class is simply to factor tests out of public namespace
    @staticmethod
    def test_polynomial_regression_detrend():
        import matplotlib.pyplot as plt

        x = _np.arange(10.)
        y = 0.5 * x ** 2 + 3 * _np.random.randn(10)
        coef, D = regression_poly(x, y, order=2)
        y_pred = D @ coef
        y_detr = regression_poly_detrend(y, order=2)

        plt.plot(x, y, marker="o", ls="")
        plt.plot(x, y_pred)
        plt.bar(x, y_detr, bottom=y_pred, width=0.03, color="red")
        plt.show()

    @staticmethod
    def test_dfa_white_noise():
        epsilon = 0.05
        n = 10_000  # dfa should give results close to 0.5 for large white noise series
        x = _np.random.randn(n)

        window_sizes = _np.power(2, _np.arange(2, 9.5, 0.5)).astype(int)
        my_dfa = DFA(window_sizes, regression_poly_detrend, detrend_kwargs={"order": 2})
        r = my_dfa(x)
        assert abs(r.weight) < 0.5 + epsilon

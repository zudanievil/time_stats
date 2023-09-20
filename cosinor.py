"""
for performing simple cosinor (cosine regression)
"""

import sys

assert sys.version_info > (3, 7, 0), "written for python 3.7+"

from typing import NamedTuple, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy import stats


# debug = [] # like printing, but you can inspect complex objects
# dpush = debug.append
# dpop = debug.pop
# dclear = debug.clear


_arr = np.ndarray
_triple = Tuple[float, float, float]

def _identity(x):
    'return x'
    return x

PIx2 = np.pi * 2

class Normal1D(NamedTuple):
    period: float = 24.
    ci_alpha: float = .01
    
    def do_fit(self, D: _arr, y: _arr) -> _triple:
        rr = np.linalg.lstsq(D.T, y, rcond=None)
        coef = rr[0]
        return coef
        
    model_name = "cosinor-normal"
    link = staticmethod(_identity)
    
    @staticmethod
    def log_likelihood(coef: _triple, D: _arr, y: _arr) -> float:
        n = len(y)
        d2 = len(y) - len(coef)
        RSS = np.sum(np.square(y - coef @ D))
        # we can simplify expression for ll:
        # sigma2 = RSS / d2 
        # (1) ll = 0.5* (n*np.log(PIx2* sigma2) - np.sum(np.square(y-coef@ D))/sigma2)
        # (2) ll = 0.5* (n*np.log(PIx2* RSS/d2) - RSS/(RSS/d2))
        ll = 0.5 * (n * np.log(PIx2 / d2 * RSS) - d2)
        return ll


class Poisson1D(NamedTuple):
    period: float = 24.
    ci_alpha: float = .01
    
    def do_fit(self, D: _arr, y: _arr) -> _triple:
        coef0 = np.random.normal(loc=0, scale=.2, size=D.shape[0])
        res = minimize(self._neg_log_lik, coef0, args=(D, y))
        coef = res['x']
        return coef
    
    model_name = "cosinor-poisson"
    link = staticmethod(np.exp)
    
    @staticmethod
    def log_likelihood(coef: _triple, D: _arr, y: _arr) -> float:
        log_rate = coef @ D
        rate = np.exp(log_rate)
        ll = y @ log_rate - rate.sum() 
        return ll
    
    @staticmethod
    def _neg_log_lik(coef, D, y) -> float:
        log_rate = coef @ D
        rate = np.exp(log_rate)
        nll =  rate.sum() - y @ log_rate 
        return nll
    

def fit(cosinor, x: _arr, y: _arr) -> dict:
    """
    cosinor: a trait-like object (functions + data inside fields)
    x: independent variable (time)
    y: dependent variable
    """

    D = gen_design_mat(cosinor.period, x)
    # cosinor_t = type(cosinor)
    coef = cosinor.do_fit(D, y)
    y_pred = cosinor.link(coef @ D)
    mesor, amp, acr = lin_coef_to_cyclic(coef)

    y_mean = cosinor.link(mesor)
    MSS = np.sum(np.square(y_mean - y_pred))
    RSS = np.sum(np.square(y - y_pred))
    
    d1, d2 = len(coef)-1, len(y) - len(coef)
    F = (d2 / d1) * MSS / RSS  # F(3-1, N-3)
    pval = 1 - stats.f.cdf(F, d1, d2)
    # TODO: this probably does not work for poisson
    
    sigma2 = RSS / d2
    sigma = np.sqrt(sigma2)
    iS = np.linalg.inv(D @ D.T)
    mesor_CI = stats.t.ppf(1-cosinor.ci_alpha / 2, d2) * sigma * np.sqrt(iS[0,0])
    # TODO: iS[0, 0] seems cryptic, 1/np.sqrt(len(y)) should be the same
    # mesor_CI = stats.t.ppf(1-self.ci_alpha/2, d2) * sigma / np.sqrt(len(y)) 
    
    # TODO: check if CI applies to poisson in the same way
    log_lik = cosinor.log_likelihood(coef, D, y)
    # TODO: make this into something more struct-like, literal dicts are stupid
    return {
        "model": cosinor.model_name,
        "period": cosinor.period,
        # "y_pred": y_pred,
        "amplitude": amp,
        "acrophase": acr,
        "mesor": mesor,
        "mesor_CI": mesor_CI,
        "sigma": sigma,

        "MSS": MSS,
        "RSS": RSS,

        "F": F,
        "d1": d1,
        "d2": d2,
        "pval": pval,
        "log_lik": log_lik,
    }


def gen_design_mat(period: float, x: _arr) -> _arr:
    """
    generate design matrix for cosinor regression: 
    the shape is [3, len(x)], and 
    it is composed of vectors [1, cos(x), sin(x)].
    """
    x1 = PIx2 / period * x
    D = np.stack([np.ones_like(x1), np.cos(x1), np.sin(x1)])
    return D


def lin_coef_to_cyclic(coef: _triple) -> _triple:
    'convert GLM coefficients to cosinor coefficients'
    mesor, beta, gamma = coef
    amp = np.sqrt(beta**2 + gamma ** 2)
    acr = np.arctan2(-gamma, beta)
    return mesor, amp, acr

def acrophase_to_time(period: float, acrophase: _arr, **ignored) -> _arr:
    'convert angular acrophase to time units. acr can be an array/float'
    return period * (1 -  (acrophase / PIx2))


def predict(
    x: _arr, /,
    period: float, 
    acrophase: float, 
    amplitude: float, 
    mesor: float, 
    link: callable = _identity,
    **ignored,
): 
    'predict values for time points in x'
    return link(mesor + amplitude * np.cos((PIx2 / period) * x + acrophase))


def conv_average(y: _arr, n: int) -> _arr:
    'sliding window average by convolution. n is window size'
    kern = np.full(shape=n, fill_value=1/ n)
    return np.convolve(y, kern, mode='same')


# ==============================================
# old code


# PIx2 = np.pi * 2

# def generate_design_matrix(self, x: np.ndarray[float]) -> np.ndarray[float]:
#     x1 = PIx2 / self.period * x
#     D = np.stack([np.ones_like(x1), np.cos(x1), np.sin(x1)])
#     return D

# def linear_coef_to_amp_phase(self, coef: tuple[float, float, float]) -> tuple[float, float, float]:
#     mesor, beta, gamma = coef
#     amp = np.sqrt(beta**2 + gamma ** 2)
#     acr = np.arctan2(-gamma, beta)
#     return mesor, amp, acr

# def acrophase_to_hours(self, acr: np.ndarray[float]):
#     'convert angular acrophase to hours'
#     return (self.period / PIx2) * (- acr % PIx2)

# def cosinor_params(self, coef, D, x, y, y_pred):
#     mesor, amp, acr = self.linear_coef_to_amp_phase(coef)
#     y_mean = self.link(mesor)
#     MSS = np.sum(np.square(y_mean - y_pred))
#     RSS = np.sum(np.square(y - y_pred))
#     d1, d2 = len(coef)-1, len(y) - len(coef)
#     F = (d2 / d1) * MSS / RSS  # F(3-1, N-3)
#     pval = 1 - stats.f.cdf(F, d1, d2)

#     S = D @ D.T
#     iS = np.linalg.inv(S)
#     sigma2 = RSS / d2
#     sigma = np.sqrt(sigma2)
#     mesor_CI = stats.t.ppf(1-self.ci_alpha/2, d2) * sigma * np.sqrt(iS[0,0])
# #         cov_beta_gamma = sigma2 * iS[1:,1:] 
#     return {
# #         "y_pred": y_pred,
#         "amplitude": amp,
#         "acrophase": acr,
#         "mesor": mesor,
#         "mesor_CI": mesor_CI,
#         "sigma": sigma,

#         "MSS": MSS,
#         "RSS": RSS,

#         "F": F,
#         "d1": d1,
#         "d2": d2,
#         "pval": pval,
#         "log_lik": self.log_lik(coef, D, y)
#     }


# def cosinor_predict(period, acrophase, amplitude, mesor, x): 
#     return mesor + amplitude * np.cos((PIx2 / period) * x + acrophase)


# class Cosinor1D(NamedTuple):
#     period: float = 24
#     ci_alpha: float = 0.05
#     # https://doi.org/10.1186/1742-4682-11-16
    
#     generate_design_matrix = generate_design_matrix
#     linear_coef_to_amp_phase = linear_coef_to_amp_phase
#     acrophase_to_hours = acrophase_to_hours
#     cosinor_params = cosinor_params
    
#     def fit(self, x, y):
#         D = self.generate_design_matrix(x)
#         rr = np.linalg.lstsq(D.T, y, rcond=None)
#         coef = rr[0]
#         y_pred = coef @ D
#         return self.cosinor_params(coef, D, x, y, y_pred)
    
# #     @staticmethod
# #     def log_lik(coef, D, y, sigma2):
# #         n = len(y)
# #         ll = 0.5 * (n * np.log(PIx2 * sigma2) - np.sum(y - coef @ D) / sigma2 ) 
# #         return ll
    
#     link = lambda self, x: x
    
#     @staticmethod
#     def log_lik(coef, D, y):
#         'gaussian log likelihood'
#         n = len(y)
#         d2 = len(y) - len(coef)
#         RSS = np.sum(np.square(y - coef @ D))
#         # sigma2 = RSS / d2
#         # ll = 0.5 * (n * np.log(PIx2 * sigma2) - np.sum(y - coef @ D) / sigma2 ) 
#         # ll = 0.5 * (n * np.log(PIx2 * RSS / d2) - RSS / (RSS / d2) )
#         ll = 0.5 * (n * np.log(PIx2 / d2 * RSS) - d2)
#         return ll
    
#     def predict(self, result, x):
#         return cosinor_predict(self.period, result["acrophase"], result["amplitude"], result["mesor"], x)

# class Cosinor1DPoisson(NamedTuple):
#     period: float = 24
#     ci_alpha: float = 0.05
#     # https://doi.org/10.1186/1742-4682-11-16
    
#     generate_design_matrix = generate_design_matrix
#     linear_coef_to_amp_phase = linear_coef_to_amp_phase
#     acrophase_to_hours = acrophase_to_hours
#     cosinor_params = cosinor_params
    
#     @staticmethod
#     def log_lik(coef, D, y):
#         'poisson log likelihood'
#         rate = np.exp(coef @ D)
# #         print(f"D: {D.shape}, c: {coef.shape}, lam: {rate.shape}")
#         neg_ll = y @ np.log(rate) + rate.sum() 
#         return neg_ll
    
#     link = staticmethod(np.exp)
    
#     @staticmethod
#     def neg_log_lik(coef, D, y):
#         'poisson negative log likelihood'
#         rate = np.exp(coef @ D)
# #         print(f"D: {D.shape}, c: {coef.shape}, lam: {rate.shape}")
#         neg_ll = rate.sum() - y @ np.log(rate) 
#         return neg_ll
    
#     def fit(self, x, y):
#         D = self.generate_design_matrix(x)
#         coef0 = np.random.normal(loc=0, scale=.2, size=D.shape[0])
#         res = minimize(self.neg_log_lik, coef0, args=(D, y))
#         coef = res['x']
#         y_pred = np.exp(coef @ D)
#         return self.cosinor_params(coef, D, x, y, y_pred)
#     # yet another way that oop model fails to deliver and descents into method madness
#     # todo: mb refactor into something more digestable?
#     # trait objects (function bundles)/functions as arguments? switch statements?
#     def predict(self, result, x):
#         y = cosinor_predict(self.period, result["acrophase"], result["amplitude"], result["mesor"], x)
#         return np.exp(y)
    

# def nbin_average(x, y, n: int):
#     l = len(y)
#     e = -(l%n)
#     x_ = x[:e] if e else x
#     y_ = y[:e] if e else y
    
#     x_ = x_.reshape((-1, n))[:, 0]
#     y_ = y_.reshape((-1, n)).mean(axis=-1)
#     return x_, y_

# def conv_average(x, y, n: int):
#     kern = np.full(shape=n, fill_value=1/ n)
#     y_ = np.convolve(y, kern, mode='same')
#     return x, y_ 
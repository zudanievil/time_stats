#! /usr/bin/env python3
"""
conventions: 
* standalone functions are the interface, classes are secondary
* functions are prefixed (much like in c) for using `from module import *` 
  (alternatively, there are NAMESPACE dicts to do `globals().update(module.NAMESPACE_x)`)
* default mutable arguments are all bound to global variables, 
  so that they can be modified at runtime (matplotlib does it, and so can I). 
  for function named `ns1_fun`, such variable would start with `_ns1_fun_`
"""








# stat_mannwhitenyu_2vec = stats.mannwhitneyu
# def stat_mannwhitenyu_mask(data: np.ndarray, classes: np.ndarray):

# def stat_mannwhitney_df( # actually, better start with np versions
#   df: pd.DataFrame, 
#   dependent_vars: list, 
#   independent_vars: list,
# ):
#   raise NotImplementedError


# def stat_anova_1w_df(
#   df: pd.DataFrame,
#   dependent_vars: list,
#   independent_vars: list,
# ):    
#   raise NotImplementedError

# def stat_anova_nw_df(
#   df: pd.DataFrame,
#   dependent_vars: list,
#   independent_vars: list,
# ):    
#   raise NotImplementedError

# def stat_chi2_df(
#   df: pd.DataFrame,
#   dependent_vars: list,
#   independent_vars: list,
# ):
#   raise NotImplementedError

# def stat_t_test_df(
#   df: pd.DataFrame,
#   dependent_vars: list,
#   independent_vars: list,
# ):
#   raise NotImplementedError

# def stat_spearman_r_df(
#   df: pd.DataFrame,
#   dependent_vars: list,
#   independent_vars: list,
# ):
#   raise NotImplementedError

# def stat_mannwhitney_np(
#   dependent_vars: np.ndarray,
#   independent_vars: np.ndarray,
# ):
#   raise NotImplementedError

# def plt_boxplot_df(ax, df: pd.DataFrame, dependent_vars: list):...
# def plt_polynome(ax, p: np.ndarray):...
# def plt_2x2_table(ax): ...
# def plt_2x2_table2(ax): ...
import sys
import numpy as np 
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors, cm
try:
    import statsmodels.api as sa
    import statsmodels.formula.api as sfa
    # import scikit_posthocs as posthocs
except ImportError as e:
    print('cannot import statsmodels, some functions will not work', e, sep="\n", file=sys.stderr)
    sa = 'statsmodels.api should be here'
    sfa = 'statsmodels.formula.api should be here'

from lib.sysutils import log

def stat_pearson_p_level(sample_len, quantile):
    'get pearson correlation value for a given distribution quantile (p-value)'
    df1 = sample_len/2 - 1
    dist = stats.beta(df1, df1, loc=-1, scale=2)
    return dist.isf(quantile)

def plt_cmap_name_to_hex(cmap_name='tab10') -> list[str]:
    _ = (np.array(plt.get_cmap(cmap_name).colors) * 255).astype(int)
    xs = (_[:, 0] << 16) + (_[:, 1] << 8) + _[:, 2]
    return np.array([f"#{hex(x)[2:]}" for x in xs])

def np_pts_to_scatter_histogram(xs: np.ndarray, bins: np.ndarray = 10, align_to: {'side', 'center'} = 'center'):
    """
    use this to turn 1d scatterplot into a histogram. bins may be an int.
    high bin values (30-100) recommended for big number of points, low (5-10) are recommended for multiple histograms
    returns: ys, labels, bins, counts
    ys -- float array in domain [0.; 1.], same length as xs. use `plt.scatter(xs, ys)`. contains NaNs where xs values are out of bins.
    labels -- int array, same length as xs. maps xs value to a bin number
    bins -- float array, if the input parameter `bins` was an int, length will be bins+1
    counts -- int array, same length as bins. stores number of values in each bin.
    usage:
    ```
    xs = np.random.randn(500)
    ys, _, _, counts = np_pts_to_scatter_histogram(xs, np.linspace(-3, 3, 50))
    ys2, _, _, counts = np_pts_to_scatter_histogram(xs, np.linspace(-3, 3, 50), 'side')
    plt.scatter(xs, ys, alpha=0.5)
    plt.scatter(xs, ys2+0.02, c ='red', alpha=0.5)
    ```
    """
    if type(bins) is int:
        bins = np.linspace(xs.min(), xs.max(), bins+1)
    labels = np.zeros(len(xs), dtype=int)
    counts = np.zeros(len(bins) - 1, dtype=int)
    ys = np.full(len(xs), np.nan, dtype=float)
    prev = bins[0]
    prev_mask = xs < prev
    for i in range(1, len(bins)):
        cur = bins[i]
        cur_mask = xs < cur
        mask = ~prev_mask & cur_mask
        labels[mask] = i
        count = mask.sum()
        counts[i-1] = count
        ys[mask] = np.arange(count, dtype=ys.dtype)
        # ys[mask] = np.argsort(xs[mask]).astype(ys.dtype)
        prev_mask = cur_mask
        prev = cur
    max_count = counts.max()
    if max_count < 2:
        if align_to == "side":
            fill = 0
        else:
            fill = 0.3
        return np.full(len(labels), fill), labels, bins, counts
    ys = ys / (max_count -1)
    if align_to == 'side':
        pass
    elif align_to == 'center':
        ys = (.5 - .5 / max_count * np.r_[np.nan, counts])[labels] + ys
    else: raise ValueError(align_to)
    return ys, labels, bins, counts

def test_visual_np_pts_to_scatter_histogram():
    xs = np.random.randn(500)
    ys2, _, _, counts = np_pts_to_scatter_histogram(xs, np.linspace(-3, 3, 50))
    ys, _, _, counts = np_pts_to_scatter_histogram(xs, np.linspace(-3, 3, 50), 'side')
    xs, ys, counts
    plt.scatter(xs, ys, alpha=0.5)
    plt.scatter(xs, ys2+0.02, c ='red', alpha=0.5)


def stats_anova2w_my_impl(c1, c2, dat):
    # https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
    # heavily modified, verified against statsmodels.api.anova_lm
    N = len(dat)
    c1s = np.unique(c1)
    c2s = np.unique(c2)
    df_a = len(c1s) - 1
    df_b = len(c2s) - 1
    df_axb = df_a*df_b 
    df_w = N - (len(c1s)*len(c2s))
    grand_mean = dat.mean()
    ssq_a = sum([(dat[c1 ==l].mean()-grand_mean)**2 for l in c1])
    ssq_b = sum([(dat[c2 ==l].mean()-grand_mean)**2 for l in c2])
    ssq_t = sum((dat - grand_mean)**2)

    ssq_w = 0
    for C in c1s:
        x = dat[c1==C]
        y = np.array([dat[(c2 == d) & (c1==C)].mean() for d in c2[c1==C]])
        ssq_w += np.sum((x-y)**2)
    ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w
    ms_a = ssq_a/df_a
    ms_b = ssq_b/df_b
    ms_axb = ssq_axb/df_axb
    ms_w = ssq_w/df_w
    f_a = ms_a/ms_w
    f_b = ms_b/ms_w
    f_axb = ms_axb/ms_w
    p_a = stats.f.sf(f_a, df_a, df_w)
    p_b = stats.f.sf(f_b, df_b, df_w)
    p_axb = stats.f.sf(f_axb, df_axb, df_w)
    
    results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
           'df':[df_a, df_b, df_axb, df_w],
           'F':[f_a, f_b, f_axb, np.nan],
            'pvalue':[p_a, p_b, p_axb, np.nan]}
    anova = pd.DataFrame(results, index=['c1', 'c2', 'c1*c2', 'Residual'])
    return anova
# TODO:
# def eta_squared(aov): 
#     aov['eta_sq'] = 'NaN'
#     aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
#     return aov
# def omega_squared(aov):
#     mse = aov['sum_sq'][-1]/aov['df'][-1]
#     aov['omega_sq'] = 'NaN'
#     aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
#     return aov
# eta_squared(aov_table1)
# omega_squared(aov_table1)
# print(aov_table1)
def stats_anova2w(df, c1, c2, dat):
    # 't ~ C(g) + C(d) + C(g)*C(d)'
    return sa.stats.anova_lm(sfa.ols(f"{dat} ~ C({c1}) + C({c2}) + C({c1})*C({c2})", data=df).fit())
def stats_anova1w(df, c1, dat):
    return sa.stats.anova_lm(sfa.ols(f"{dat} ~ C({c1})", data=df).fit())
    
def stats_fisher_lsd(anova, df, c1, c2, dat):
    # verivied via statistica 13 on several datasets (pvalues are same up to 7 decimal places)
    # https://www.statology.org/fishers-least-significant-difference/
    msw = anova.mean_sq.Residual
    dfw = anova.df.Residual
    c1s = df[c1].unique()
    c2s = df[c2].unique()
    
    idx = np.array(list(itertools.product(c1s, c2s)))
    pdidx =pd.MultiIndex.from_arrays(idx.T, names=[c1, c2])
    pvals = pd.DataFrame(index=pdidx, columns=pdidx)
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            g1 = df.loc[(df[c1] == idx[i, 0]) & (df[c2] == idx[i, 1]), dat]
            g2 = df.loc[(df[c1] == idx[j, 0]) & (df[c2] == idx[j, 1]), dat]
            if len(g1) == 0 or len(g2) == 0:
                p = 1.0
                log('info', f"sample {(idx[i, 0], idx[i, 1])} len={len(g1)}; "
                    f"sample {(idx[j, 0], idx[j, 1])} len={len(g2)}")
            else:
                t_abs = abs(g1.mean() - g2.mean()) / np.sqrt(msw * (1/len(g1) + 1/len(g2)))
                p = (1 - stats.t.cdf(t_abs, dfw)) * 2
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals

def stats_fisher_lsd_1d(anova, df, c1, dat):
    # verivied via statistica 13 on several datasets (pvalues are same up to 7 decimal places)
    # https://www.statology.org/fishers-least-significant-difference/
    msw = anova.mean_sq.Residual
    dfw = anova.df.Residual
    c1s = df[c1].unique()
    
    idx = np.array(c1s)
    pdidx =pd.Index(idx, name=c1)
    pvals = pd.DataFrame(index=pdidx, columns=pdidx)
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            g1 = df.loc[(df[c1] == idx[i]), dat]
            g2 = df.loc[(df[c1] == idx[j]), dat]
            t_abs = abs(g1.mean() - g2.mean()) / np.sqrt(msw * (1/len(g1) + 1/len(g2)))
            p = (1 - stats.t.cdf(t_abs, dfw)) * 2
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals

def stat_1d_pval_table_to_sparse(lsd):
    lsd = lsd.copy()
    lotri = np.tri(*lsd.shape, dtype=bool)
    lsd.values[lotri] = np.nan
    lsd = lsd.unstack().dropna()
    lsd = lsd[lsd < 0.05]
    return lsd
# ================= plots

c_mm2in = 1/25.4
c_A4_w = 210 * c_mm2in
c_A4_h = 297 * c_mm2in

class E(Exception):
    """functions from this module throw only these exceptions directly"""
    __slots__ = ("data", )
    def __init__(self, data): self.data = data
    def __repr__(self):
        cls = self.__class__
        return f"{cls.__module__}.{cls.__name__}({self.data})"

# math
def np_arange(start, step, n_elements: int) -> np.ndarray:
    """alternative form of specifying arythmetic progression. supports array inputs"""
    return np.linspace(start, step*(n_elements-1) + start, n_elements) 

def np_minmax(x):
    x = x - x.min()
    return x / x.max()

def np_shift_corr(trail: np.ndarray, lead: np.ndarray, max_shift: int = 20) -> np.ndarray:
    """
    return pearson correlation for pairs of 
    (trail, lead), ..., (trail[max_shift:], lead[:-max_shift])
    """
    assert max_shift > 0
    r = np.empty(max_shift, dtype=float)
    r[0] = stats.pearsonr(trail, lead).statistic
    for s in range(1, max_shift):
        r[s] = stats.pearsonr(trail[s:], lead[:-s]).statistic
    return r

def _test_np_shift_corr():
    shift = 20
    x = np.cos(np.linspace(-np.pi, np.pi, 40))
    y = np.cos(np.linspace(-np.pi+np.pi*0.2, np.pi+ np.pi*0.2, 40))
    rx = np.linspace(0, np.pi, shift)
    r = np_shift_corr(x, y, shift)
    plt.plot(rx, r)
    plt.plot(np.linspace(0, 2*np.pi, 40), x, c="magenta", ls="--", label="x")
    plt.plot(np.linspace(np.pi*0.2, 2*np.pi+ np.pi*0.2, 40), y, c="cyan", ls="--", label="y")
    plt.axvline(rx[r.argmax()], c="r")
    print(np.pi*0.2)
    print(rx[r.argmax()-1:r.argmax()+2])
#_test_np_shift_corr()

def np_reduce_mean_sd_count(n, mean, sd, sum_kw={}):
    """
    based on the fact that count + n first moments are additive.
    should work with pandas df due to duck typing.
    sum_kw: args to pass to x.sum() method
    """
    nm = n * mean # to additive form
    nm2 = n * (mean**2)
    ns2 = n * (sd**2)
    if sum_kw: #reduce
        n = n.sum(**sum_kw)
        nm = nm.sum(**sum_kw)
        nm2 = nm2.sum(**sum_kw)
        ns2 = ns2.sum(**sum_kw)
    else:
        n = n.sum()
        nm = nm.sum()
        nm2 = nm2.sum()
        ns2 = ns2.sum()
    r_n = n # back to moments
    r_m = nm / n
    r_s = np.sqrt((ns2 + nm2) / n - r_m**2)
    return r_n, r_m, r_s

def _test_np_reduce_mean_sd_count_2d():
    xs = np.random.uniform(size=100_000).reshape(-1, 10)
    partials = []
    for i in range(0, 10_000, 1000):
        xi = xs[i:i+1_000]
        partials.append(([len(xi)]*10, np.mean(xi, axis=0), np.std(xi, axis=0)))
    partials = np.array(partials).swapaxes(0, 1)
    r_cnt, r_m, r_sd = np_reduce_mean_sd_count(*partials, dict(axis=0))
    assert np.allclose(r_cnt, [len(xs)]*10)
    assert np.allclose(r_m, np.mean(xs, axis=0))
    assert np.allclose(r_sd, np.std(xs, axis=0))
_test_np_reduce_mean_sd_count_2d()

# wrangling
def pd_list_empty_columns(df: pd.DataFrame) -> list[str]:
    x = df.isna().sum()
    return x[x==len(df)].index.to_list()

def plt_hex_colors_to_np(colors: list[str], has_alpha=True):
    r = np.zeros((len(colors)), dtype='>u4')
    for i, color in enumerate(colors):
        if color.startswith("#"):
            color = color[1:]
        elif color.startswith("0x"):
            color = color[2:]
        short = len(color) == 6
        color = int(color, base=16)
        if has_alpha and short:
            color = (color << 8) + 0xff
        elif not (has_alpha or short):
            color = color >> 8
        r[i] = color
    r = r.view(np.uint8).reshape(-1, 4)
    if not has_alpha:
        r = r[:, 1:]
    return r
    
def _test_plt_hex_colors_to_np():
    colors = ["#abcdef", "abcdef", "abcdefff"]
    no_alpha = np.array([[171, 205, 239], [171, 205, 239], [171, 205, 239]], dtype=np.uint8)
    alpha = np.array([[171, 205, 239, 255], [171, 205, 239, 255], [171, 205, 239, 255]], dtype=np.uint8)
    r_a = plt_hex_colors_to_np(colors)
    r_na = plt_hex_colors_to_np(colors, has_alpha=False)
    assert (no_alpha == r_na).all(), repr(r_na)
    assert (alpha == r_a).all(), repr(r_a)
_test_plt_hex_colors_to_np()


# ===============================================

def plt_kde_groups(ax, gdata, colors = None, labels=None):
    from sklearn.neighbors import KernelDensity
    plot_data = []
    for i, data in enumerate(gdata):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data.reshape(-1, 1))
        xs_pred = np.linspace(data.min(), data.max()).reshape((-1, 1))
        ys_pred = np.exp(kde.score_samples(xs_pred))
        color = None if colors is None else colors[i]
        label = None if labels is None else labels[i]
        plot_data.append((data, xs_pred, ys_pred, color, label))
    _ = np.concatenate([pd[2] for pd in plot_data])
    mi, ma = _.min(), _.max()
    d = ma - mi
    for (xs, xs_pred, ys_pred, color, label) in plot_data:
        pseudo_ys = np.random.uniform(mi-0.2*d, mi-0.1*d, size=len(xs))
        l = ax.plot(xs, pseudo_ys, ls="none", marker="+", markersize=10, color=color, alpha=0.8, label=label)[0]
        if color is None:
            color = l.get_color()
            # print(color)
        ax.plot(xs_pred, ys_pred, color=color, lw=3, alpha=0.7)
        ax.set_ylim(mi-0.3*d, ma+0.3*d)


def plt_multi_boxplot(
    ax, 
    group_data: list[np.ndarray],
    labels: list[str]= None,
    colors: list = None,
    boxplot_kw = None,
    scatter_kw = None,
):
    boxplot_kw = boxplot_kw or plt_multi_boxplot.boxplot_kw
    scatter_kw = scatter_kw or plt_multi_boxplot.scatter_kw
    nc = colors is None
    if not nc: colors = iter(colors)
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    for i, data in enumerate(group_data):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(data, positions=[i, ], manage_ticks=False, **boxplot_kw)
        ax.scatter(x_pseudo, data, color=None if nc else next(colors), **scatter_kw)
    if labels:
        ax.set_xticks(np.arange(len(labels)), labels)

plt_multi_boxplot.boxplot_kw = dict(
    showcaps=False, showfliers=False,
    showmeans=True, widths=0.8,
    meanprops = dict(marker="D", markerfacecolor="k", markeredgecolor="k",),
    medianprops = dict(c="k"),
)
plt_multi_boxplot.scatter_kw = dict(alpha=0.7,)

def plt_line_and_error(
    ax, 
    x: np.ndarray, 
    y: np.ndarray, 
    yerr: np.ndarray = None,
    plot_kw = None,
    fill_between_kw = None,
):
    if yerr is None:
        yerr = y
        y = x 
        x = np.arange(len(y))
    ax.plot(x, y, **(plot_kw or plt_line_and_error.plot_kw))
    ax.fill_between(x, y-yerr, y+yerr, **(fill_between_kw or plt_line_and_error.fill_between_kw))

plt_line_and_error.plot_kw = dict(lw=2)
plt_line_and_error.fill_between_kw = dict(lw=0, alpha=0.2)


def plt_multibar(
    ax, 
    heights: np.ndarray, 
    positions: np.ndarray=None, 
    labels: list[str] = None, 
    x_tick_labels: list[str] = None, 
    colors: list = None, 
    bar_width_sum: float = 0.8, 
    bar_offset: float = 0.5, 
    bar_kw=None
) -> None: # done
    """
    heights: (m, n) array or (n, ) array or list/tuple with m (n, ) shaped arrays
    positions: (n, ) shaped array. default is np.arange(n)
    labels: m-sized list of legend labels. NB: call ax.legend() by yourself
    x_tick_labels: n-sized array for x-axis ticks, string literal 'positions' or None (ticks are not modified).
    colors: m-sized, any matplotlib-compatible color format
    bar_width_sum: how much space bars take together
    bar_offset: alignment of bars (0. -- left, .5 -- center, 1. -- right)
    bar_kwargs: dict to feed to ax.bar() or m-sized list of dicts
    ```
    # example:
    data = np.array([
        [1, 2, 3],
        [3, 2, 1],])
    plt_multibar(plt.gca(), data)
    ```
    """
    bar_kw = bar_kw or plt_multibar.bar_kw
    # coerce heights
    if not isinstance(heights, np.ndarray):
        n_sampls = len(heights)
        n_vals = len(heights[0])
    elif heights.ndims == 1:
        n_sampls = 1
        n_vals = len(heights)
        heights = [heights, ]
    elif heights.ndims != 2:
        raise _E(f"heights.ndims == {heights.ndims}, but should be 1 or 2")
    else:
        n_sampls, n_vals = heights.shape
    if positions is None:
        positions = np.arange(n_vals)
    else:
        if n_vals != len(positions):
            raise _E(f"heights.shape[-1]={n_vals} while len(positions)={len(positions)}, but they should match")
    bar_width = bar_width_sum / n_sampls
    bar_start =  (0.5 - bar_offset * n_sampls) * bar_width 
    no_color = colors is None
    no_label = labels is None
    multi_kw = not isinstance(bar_kw, dict)
    for i in range(n_sampls):
        ax.bar(
            positions+(bar_start+bar_width*i), 
            heights[i], 
            bar_width, 
            color=colors if no_color else colors[i], 
            label=labels if no_label else labels[i], 
            **(bar_kw[i] if multi_kw else bar_kw),
        )
    if x_tick_labels == 'positions':
        ax.set_xticks(positions, positions)
    elif x_tick_labels is not None:
        ax.set_xticks(positions, x_tick_labels)
plt_multibar.bar_kw = dict(alpha=0.8)


def plt_colored_bars(
    ax,
    height_vals: np.ndarray, 
    color_vals: np.ndarray,
    positions: np.ndarray = None,
    norm = colors.Normalize(),
    cmap = None,
    bar_kw = None,
    colorbar_kw = None,
):
    """
    height_vals: 1d array, heights of the bars
    color_vals: 1d array, value to be mapped to colors with norm and cmap
    positions: 1d array, positions of bars on the x-axis; np.arange(n) by default
    norm: matplotlib.colors.Normalize and subclasses
    cmap: colormap name (see list(plt.colormaps)) or matplotlib.colors.Colormap and subclasses
    bar_kwargs: kwargs for plt.bar
    colorbar_kwargs: kwargs for plt.colorbar. pass None to disable colorbar plotting
    """
    bar_kw = bar_kw or plt_colored_bars.bar_kw
    colorbar_kw = colorbar_kw or plt_colored_bars.colorbar_kw
    assert len(height_vals) == len(color_vals)
    if cmap is None or isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # Normalize just scales values between [min; max] to [0.; 1.] range, but there are other Norms 
    color_vals = cmap(norm(color_vals))
    if positions is None:
        positions = np.arange(len(height_vals))
    bars = ax.bar(positions, height_vals, color = color_vals, **bar_kw)
    if colorbar_kwargs is not None:
        plt.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, **colorbar_kw)
plt_colored_bars.bar_kw = dict()
plt_colored_bars.colorbar_kw = dict()


def plt_add_axes(fig, top, left, width, height, unit = c_mm2in, **kw):
    fh = fig.get_figheight()/unit
    fw = fig.get_figwidth()/unit
    rect = np.array([left/fw, 1-(top+height)/fh, width/fw, height/fh])
    ax = fig.add_axes(rect) 
    # fck who thought that relative units are a good idea for "PRINT-READY" FIGURES???
    return ax


class plt_Pane:
    "class for simple declarative layouts specified in physical units"
    __slots__ = ("typ", "args", "w", "h", "fn")
    
    @classmethod
    def hstack(cls, *args):
        s = cls()
        s.typ = "hstack"
        s.args = args
        s.w = s.h = s.fn = None
        return s
    @classmethod
    def wstack(cls, *args):
        s = cls()
        s.typ = "wstack"
        s.w = s.h = s.fn = None
        s.args = args
        return s
    @classmethod
    def ax(cls, fn, w, h, add_axes_kwargs={}):
        s = cls()
        s.typ = "ax"
        s.fn = fn
        s.w = w
        s.h = h
        s.args = add_axes_kwargs
        return s
        
    def draw(s, fig, top=10, left=20, unit=c_mm2in):
        if s.typ == "ax": 
            fh = fig.get_figheight()/unit 
            fw = fig.get_figwidth()/unit
            rect = np.array([left/fw, 1-(s.h+top)/fh, s.w/fw, s.h/fh])
            ax = fig.add_axes(rect) 
            # fuck who thought that relative units are a good idea for "PRINT-READY" FIGURES???
            return s.fn(ax), ax, s.h, s.w
        s.h = s.w = 0
        vertical = s.typ == "hstack"
        rets = []; axs = []
        for arg in s.args:
            if isinstance(arg, (int, float)):
                if vertical: s.h += arg # increase height offset
                else: s.w += arg # increase horizontal offset
            else:
                if vertical:
                    ret, ax, dh, dw = arg.draw(fig, top+s.h, left, unit)
                    s.h += dh
                    s.w = max(s.w, dw)
                else:
                    ret, ax, dh, dw = arg.draw(fig, top, left+s.w, unit)
                    s.h = max(s.h, dh)
                    s.w += dw
                rets.append(ret)
                axs.append(ax)
        return rets, axs, s.h, s.w


def _test_plt_Pane():
    def f(ax):
        ax.set_facecolor("#f473da")
        ax.tick_params('both', which='both', bottom=False, top=False, left=False, right=False, labelbottom = False, labeltop = False, labelleft=False, labelright=False)
    fig = plt.figure(figsize=(c_A4_w, c_A4_h), )
    p = plt_Pane
    p.hstack( 
        p.ax(f, h=50, w=50),
        1,
        p.wstack(
            p.hstack(
                p.ax(f, h=25, w=25),
                p.ax(f, h=25, w=25),
            ),
            1,
            p.ax(f, h=50, w=25),
            1,
            p.ax(f, h=50, w=25)
        ),
    ).draw(fig, top=10, left=20)
# _test_plt_Pane()


NAMESPACE_plt = {k: v for k, v in globals().items() if k.startswith("plt_")}
NAMESPACE_stat = {k: v for k, v in globals().items() if k.startswith("stat_")}
    



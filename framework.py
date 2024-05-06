import sys, os
from pathlib import Path

from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
from IPython.display import display_html, display
# from scipy.optimize import curve_fit, minimize
# from scipy import signal as s_signal
import matplotlib.pyplot as plt
from typing import NamedTuple as _NT


# <editor-fold desc="hotreload">

from lib.sysutils import *

module_dir = os.path.realpath(os.path.dirname(__file__) or ".") + "/"
from lib import xlsx_report as report
from lib import cosinor
# cosinor = import_path(module_dir+"lib/cosinor.py")
from lib import dfa
# dfa = import_path(module_dir+"lib/dfa.py")
from lib import nonparametric
# nonparametric = import_path(module_dir+"lib/nonparametric.py")
from lib.stacked_environment import *
# import_path_star(module_dir+"lib/stacked_environment")
# from lib.statplots import *
import_path_star(module_dir+"lib/statplots.py")
# </editor-fold>

# <editor-fold desc="framework_generic">

def get_opts(opts=None) -> dict:
    return ENV if (opts is None) else opts # ENV is supposed to be defined in external code

class ckpt_proto:
    def ckpt_load(self):...
    def ckpt_create(self):...
    @staticmethod
    def ckpt_postprocess(t): ... # optional

def ckpt_load_or_create(ckpt): 
    t = ckpt.ckpt_load()
    t = ckpt.ckpt_create() if (t is None) else t
    pp = getattr(ckpt, "ckpt_postprocess", None)
    t = t if (pp is None) else pp(t) 
    return t

def ckpt_dst_is_newer(dst: Path, src: Path)->bool:
    if not src.exists():
        raise FileNotFoundError(src.as_posix())
    elif not dst.exists():
        return False
    elif dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    else:
        return False

def ckpt_load_default(self) -> pd.DataFrame:
    if not ckpt_dst_is_newer(self.dst, self.src):
        return None  # type: ignore
    t = pd.read_csv(self.dst, sep="\t")
    return t

def ckpt_name(cls) -> str:
    return cls.__name__.removeprefix("ckpt_")
# </editor-fold>


# <editor-fold desc="checkpoint_impl">
cAnim = "Animal No."
cGroup = "Group"
cRer = "RER"
cDist = "DistD"
cFeed = "Feed"
cDrink = "Drink"
cTime = "Time"
cO2 = "VO2(1)"
cCO2 = "VCO2(1)"
cRelFeed = "feed%mass"
cFeedByDist = "feed%mass%dist*10^4"
# from lib.statplots import c_mm2in, c_A4_w, c_A4_h

class ckpt_activity_ts_v0(_NT):
    src: Path
    dst: Path  
        
    def ckpt_create(self):
        # from lib.statplots import np_arange
        t0 = pd.read_csv(self.src, sep="\t")

        columns = [cRer, cFeed, cDist, cO2, cCO2, cDrink, ]
        for __c in columns:
            ... 
            # print(f"number of NaNs in {__c}: {t0[__c].isna().sum()}")
        # ^^^ compare to number of animals
        # print(f"number of NaNs in {cDrink}: {t0[cDrink].isna().sum()}")
        t0[cFeed] =t0[cFeed].fillna(0)
        t0[cDrink] = t0[cDrink].fillna(0)
        t0["mass"] = t0["VO2(3)"] / t0[cO2]
        t0[cRelFeed] = t0[cFeed] / t0["mass"]
        t0[cFeedByDist] = t0[cRelFeed]*10_000/(t0[cDist]+1) # prevents NaNs
        columns += [cRelFeed, cFeedByDist]
        
        """
        WARNING: NaN values in regression will cause problems 
        (most likely, coefficients will be Nan, but other things may happen)
        """
        accum = []
        for col in columns:
            t0g = t0.groupby([cGroup, cAnim])
            
            for g, a in t0g.groups:
                df = t0g.get_group((g, a))
                y = df[col]
                time_offset = df[cTime].iloc[0] 
                x = np_arange(time_offset, step=0.5, n_elements=len(y))
                t = pd.DataFrame({
                    'BEHAV': col,
                    'GROUP': g, 
                    'ANIMAL': a,
                    'X': x, 
                    'Y': y, 
                })
                accum.append(t)
        t = pd.concat(accum, axis='index', ignore_index=True)
        t[['GROUP', 'ANIMAL']] = t[['GROUP', 'ANIMAL']].astype(int)
        
        t.to_csv(self.dst, index=False, sep="\t")
        return t
    ckpt_load = ckpt_load_default
    @staticmethod
    def ckpt_postprocess(t):
        return t.set_index(["BEHAV", "GROUP", "ANIMAL"]).sort_index()
ckpt_activity_ts_v0.name = ckpt_name(ckpt_activity_ts_v0)

class ckpt_ivf_features_v0(_NT):
    src: Path
    dst: Path
        
    def ckpt_create(self):
        t02 = pd.read_csv(self.src, sep="\t")
        t02 = t02.drop(index=[32, 33])
        t02.drop(columns=["sign", "cell"], inplace=True)
        def __date_parse(s: str):  # some weird mixture of russian and american date format
            m, d, y = map(int, s.split("/"))
            return datetime(y, m, d)
        t02['birth'] = t02['birth'].map(__date_parse)
        t02['group'] = t02['group'].astype(int)
        t02['animal_no'] = t02['animal_no'].astype(int)
        t02[cRelFeed] = t02['Feed_total'] / t02['Mass_13.05']
        t02[cFeedByDist] = t02[cRelFeed] / t02['DistD_total']
        t02["O2%feed%mass"] = (t02['O2_mouse_day'] / 10 + t02['O2_mouse_night']/14)  / t02[cRelFeed]
        t02.to_csv(self.dst, index=False, sep="\t")
        return t02
    ckpt_load = ckpt_load_default
ckpt_ivf_features_v0.name = ckpt_name(ckpt_ivf_features_v0)

class ckpt_nonparametric_ts_sum_v0(_NT):
    src: Path
    dst: Path
    animals: dict[int, list[int]]
    samples_per_hour: int
    period: int
    columns: list = [cRer, cFeed, cDist, cO2, cCO2, cDrink, ]

    def ckpt_create(self):
        # from lib import dfa, nonparametric
        t0 = pd.read_csv(self.src, sep="\t")
        window_sizes = (np.exp(np.arange(2., 4.6, 0.2)) / 2).astype(int)
        dfa1 = dfa.DFA(window_sizes)
        dfa2 = dfa.DFA(window_sizes, detrend_kwargs=dict(order=2))  
        t1 = t0.set_index(["BEHAV", "GROUP", "ANIMAL", "X"], drop=True)
        t1.sort_index(inplace=True)
        
        columns = self.columns
        res = []
        for col in columns:
            for g, animals in self.animals.items():
                for a in animals:
                    ti = t1.loc[col, g, a]
                    x = ti.index.values
                    y = ti.Y.values
                    # ti = t_animal(ts[col], a)
                    # _, _, x, y = ti_unpack(ti)
                    r = dict()
                    # SAMPLES_PER_HOUR = opts['samples_ph']
                    # PERIOD = opts['period']
                    yhour = nonparametric.binned_avg(y, self.samples_per_hour)
                    y24 = nonparametric.periodic_avg(yhour, self.period)
                    r['behav'] = col
                    r['group'] = g
                    r['animal'] = a
                    r['IS'] = nonparametric.IS(y, y24)
                    r['IV'] = nonparametric.IV(y)
                    for n in [4, 5, 6, 8, 10]:
                        r[f'IV{n}'] = nonparametric.IVsub(y, n)
                    r["DFA1"] = dfa1(y).weight
                    r["DFA2"] = dfa2(y).weight 
                    r["PoV"] = nonparametric.PoV(y, frequency = 2, first_harmonic_range=(1/25, 1/24))
                    r['conv_diff'] = nonparametric.conv_diff(y, 6)
                    r["mean"] = y.mean()
                    r['std'] = y.std()
                    s = self.period * self.samples_per_hour
                    r['AC24'] = stats.pearsonr(y[s:], y[:-s]).statistic
                    r['PFD'] = nonparametric.PFD(y)
                    for kmax in [5, 10, 15,]:
                        r[f'HFD{kmax}'] = nonparametric.HFD(y, kmax)
                    r.update(nonparametric.RelAmp(y24))
                    res.append(r)
        t03 = pd.DataFrame.from_records(res)
        t03.to_csv(self.dst, index=False, sep="\t")
        return t03
    ckpt_load = ckpt_load_default
ckpt_nonparametric_ts_sum_v0.name = ckpt_name(ckpt_nonparametric_ts_sum_v0)     

# from lib import cosinor
N24 = cosinor.Normal1D(period=24.)
P24 = cosinor.Poisson1D(period=24.)

class ckpt_ivf_ts_cosinor_v0(_NT):
    src: Path
    dst: Path
    animals: dict[int, list[int]]#= ENV['data_animals']
    columns: list[str]
    period: int
    
    def ckpt_create(self):
        t0 = pd.read_csv(self.src, sep="\t")
        t0.set_index(["BEHAV", "GROUP", "ANIMAL", "X"], drop=True, inplace=True)
        t0.sort_index(inplace=True)
        rs = []
        models = [cosinor.Normal1D(self.period), cosinor.Poisson1D(self.period)]
        model_names = [f"N{self.period}", f"P{self.period}"]
        for model, model_name in zip(models, model_names):
            for col in self.columns:
                for g, anims in self.animals.items():
                    for a in anims:
                        ti = t0.loc[col, g, a]
                        x = ti.index.values
                        y = ti.Y.values
                        r = cosinor.fit(model, x, y)
                        r['group'] = g
                        r['animal'] = a
                        r['behav'] = col
                        r['model'] = model_name
                        rs.append(r)
        rs = pd.DataFrame.from_records(rs)
        rs['acrophase_h'] = cosinor.acrophase_to_time(self.period, rs['acrophase'])
        rs.to_csv(self.dst, sep="\t", index=False)
        return rs
    ckpt_load = ckpt_load_default
    @staticmethod
    def ckpt_postprocess(t):
        return t.set_index(["model", "behav", "group", "animal"]).sort_index()
ckpt_ivf_ts_cosinor_v0.name = ckpt_name(ckpt_ivf_ts_cosinor_v0)     


def ti_unpack(ti) -> tuple[int, int, np.ndarray, np.ndarray]:
    return (
        ti.GROUP.iloc[0], ti.ANIMAL.iloc[0],
        ti.X.values, ti.Y.values)
    

class ckpt_ivf_histology_v0(_NT): # tested
    src: Path
    dst: Path
    # animals: dict[int, list[int]]
    
    def ckpt_create(self):
        # from lib.statplots import np_reduce_mean_sd_count
        rows = []
        hist_table_dir = self.src
        for p in hist_table_dir.iterdir():
            name = p.name
            if not name.endswith(".bg.csv"):
                continue
            if "_cortex_" not in name:
                continue
            p1 = hist_table_dir / name.replace(".bg.csv", ".csv")
            parts = name.removesuffix(".bg.csv").split("_")
            
            row = dict(zip(["animal", "p1", "p2"], parts[:-3]))
            row['snap_id'] = parts[-1]    # there can be 4 or 5 parts in the name
            row['region'] = parts[-2]
            row['side'] = parts[-3]
            if parts[0] in ["10r", "10null", "19.1l", "19.1r"]:
                row['group'] = 0
            else:
                row['group'] = 2
            
            bgt = pd.read_csv(p, index_col=0)
            ht = pd.read_csv(p1, index_col=0)
            def __get_channel(x): return ord(x[1]) - 48 # ascii arithmetic 
            bgt["Channel"] = bgt["Label"].map(__get_channel)
            bgt.set_index("Channel", inplace=True)
            ht["Channel"] =  ht["Label"].map(__get_channel)
            ht["Roi"] = ht["Label"].map(lambda x: x[11:])
            ht.set_index(["Channel", "Roi"], inplace=True)
            # while mean brightness is can be heavily influenced by histological artifacts, boundaries and light distribution,
            # median is influenced primarily by the amount of dye and illumination strength / filters etc.
            ht["MeanN"] =  ht["Mean"] / bgt["Median"]
            ht["StdDevN"] = ht["StdDev"] / bgt["Median"]
            r_area, r_mean, r_sd = np_reduce_mean_sd_count(
                ht["Area"].unstack("Channel"), 
                ht["MeanN"].unstack("Channel"), 
                ht["StdDevN"].unstack("Channel"),
                sum_kw=dict(axis="index"),
            )
            # print(r_area, r_mean, r_sd, sep="\n")
            # return
            row["area_mean"] = ht.loc[1, "Area"].mean()
            row["area_sd"] = ht.loc[1, "Area"].std()
            row["cell_count"] = ht.loc[1, "Area"].count()
            row["area_sum"] = r_area.loc[1]
            row.update(zip(["mean_c1", "mean_c2", "mean_c3"], r_mean))
            row.update(zip(["sd_c1", "sd_c2", "sd_c3"], r_sd))
            rows.append(row)
        t = pd.DataFrame(rows)
        t = t["snap_id	animal	group	p1	p2	region	side	area_mean	area_sum	area_sd	cell_count	mean_c1	mean_c2	mean_c3	sd_c1	sd_c2	sd_c3".split("\t")]
        t.to_csv(self.dst, sep="\t", index=False)
        return t
    ckpt_load = ckpt_load_default
ckpt_ivf_histology_v0.name = ckpt_name(ckpt_ivf_histology_v0)     

class ckpt_ivf_histology_avg_by_animal_v0(_NT): # tested
    src: Path
    dst: Path

    def ckpt_create(self):
        t = pd.read_csv(self.src, sep="\t")
        tg = t.groupby("animal")
        rows = []
        for a in tg.groups:
            gt = tg.get_group(a)
        
            row = dict()
            for i in range(1, 4):
                _ = np_reduce_mean_sd_count(gt["area_sum"], gt[f"mean_c{i}"], gt[f"sd_c{i}"])
                row[f"mean_c{i}"] = _[1]
                row[f"sd_c{i}"] = _[2]
            row["area_sum"] = _[0]
            row["animal"] = a
            row["slice_count"] = len(gt)
            row["group"] = gt["group"].iloc[0]
            row["cell_count"], row["area_mean"], row["area_sd"] = np_reduce_mean_sd_count(gt["cell_count"], gt["area_mean"], gt["area_sd"])
            rows.append(row)
        t2 = pd.DataFrame(rows)
        t2 = t2["animal group slice_count cell_count area_mean area_sd area_sum mean_c1 sd_c1 mean_c2 sd_c2 mean_c3 sd_c3".split()]
        t2.to_csv(self.dst, sep="\t", index=False)
        return t2
    ckpt_load = ckpt_load_default
ckpt_ivf_histology_avg_by_animal_v0.name = ckpt_name(ckpt_ivf_histology_avg_by_animal_v0)
        

def ckpt_table_ckpt_v0(proj: Path() = None):
    ENV = get_opts(opts=None)
    if proj is None: proj = Path()
    d = get_caller().__name__.removeprefix("ckpt_")
    outd = proj / "checkpoints" / d
    ckpts = dict()
    ckpts[ckpt_activity_ts_v0.name] = ckpt_activity_ts_v0(
        src = proj/"data_src/Phenomaster_2022_Total.csv", 
        dst = outd/f"{ckpt_activity_ts_v0.name}.csv",
    )
    ckpts[ckpt_ivf_features_v0.name] = ckpt_ivf_features_v0(
        src = proj/"data_src/ITOG2.csv",
        dst = outd/f"{ckpt_ivf_features_v0.name}.csv"
    )
    ckpts[ckpt_nonparametric_ts_sum_v0.name] = ckpt_nonparametric_ts_sum_v0(
        src = outd/f"{ckpt_activity_ts_v0.name}.csv",
        dst = outd/f"{ckpt_nonparametric_ts_sum_v0.name}.csv",
        animals = ENV['data_animals'],
        samples_per_hour = ENV['data_samples_per_hour'],
        period  = ENV['data_period'],
    )
    ckpts[ckpt_ivf_ts_cosinor_v0.name] = ckpt_ivf_ts_cosinor_v0(
        src = outd/f"{ckpt_activity_ts_v0.name}.csv",
        dst = outd/f"{ckpt_ivf_ts_cosinor_v0.name}.csv",
        animals = ENV['data_animals'],
        columns = [cRer, cFeed, cDist, cO2, cCO2, cDrink, ],
        period = ENV['data_period'],
    )
    ckpts[ckpt_ivf_histology_v0.name] = ckpt_ivf_histology_v0(
        src = proj/"ivf_hist/nuclei_measure/2MA",
        dst = outd/f"{ckpt_ivf_histology_v0.name}.csv",
    ) 
    ckpts[ckpt_ivf_histology_avg_by_animal_v0.name] = ckpt_ivf_histology_avg_by_animal_v0(
        src = outd/f"{ckpt_ivf_histology_v0.name}.csv",
        dst = outd/f"{ckpt_ivf_histology_avg_by_animal_v0.name}.csv"
    )
    return outd, ckpts

def ckpt_fn_ivf_histology_avg_by_animal_side_v1(ivf_histology_v0: ckpt_proto):
    hist = ckpt_load_or_create(ivf_histology_v0) # ckpts['ivf_histology_v0'])
    tg = hist.groupby(["animal", "side"])
    rows = []
    for gr in tg.groups:
        gt = tg.get_group(gr)
        a, s = gr # animal, side
        row = dict()
        for i in range(1, 4):
            _ = np_reduce_mean_sd_count(gt["area_sum"], gt[f"mean_c{i}"], gt[f"sd_c{i}"])
            row[f"mean_c{i}"] = _[1]
            row[f"sd_c{i}"] = _[2]
        row["area_sum"] = _[0]
        row["animal"] = a
        row["side"] = s
        row["slice_count"] = len(gt)
        row["group"] = gt["group"].iloc[0]
        row["cell_count"], row["area_mean"], row["area_sd"] = np_reduce_mean_sd_count(gt["cell_count"], gt["area_mean"], gt["area_sd"])
        rows.append(row)
    t2 = pd.DataFrame(rows)
    t2 = t2["animal group side slice_count cell_count area_mean area_sd area_sum mean_c1 sd_c1 mean_c2 sd_c2 mean_c3 sd_c3".split()]
    return t2

# </editor-fold> # checkpoint_impl

# <editor-fold desc="plotter_util"> 

LETTERS_NUM = "123456789"
LETTERS_EN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_RU = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"
LANG_RU = "RU"
LANG_EN = "EN"
class plt_add_letters:
    "add letters/numbers onto Axes (as in publications)"
    def __init__(
        self, 
        letters = LETTERS_EN, 
        style = dict(fontweight="bold", fontsize=15),
    ):
        self.letters = letters; self.style = style
        
    def __call__(self, axs, pos):
        axs = np.asarray(axs).ravel()
        for ax, l in zip(axs, self.letters):
            ax.annotate(l, pos, xycoords="axes fraction", **self.style)
                
def plt_figsave_svg_png_300dpi(path, fig):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(path)

def plt_figsave_svg_png_300dpi_no_close(path, fig):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(path)

def plt_figsave_phony(path, fig):
    print("phony save", path)

def plt_figsave_svg(path, fig):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)

def plt_figsave_png_300dpi(path, fig):
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(path)

def plt_figsave_svg_png_600dpi(path, fig):
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(path)

def plt_delete_ticks(axs, axis: {'x', 'y', 'both'} = 'x'):
    for ax in axs:
        if axis == 'x':
            ax.tick_params(axis=axis,          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,
            )
        elif axis == 'y':
            ax.tick_params(axis=axis, which='both', left=False, right=False,)
        elif axis == 'both':
            ax.tick_params(axis=axis, which='both', left=False, right=False, top=False, bottom=False)

def cmp_to_stars_2_3gr(
    data: list[np.ndarray], test_fn: callable, do_trend=False,
) -> list[str]:
    def pval_to_star(pval, sym) -> str:
        if pval > 0.1: return ''
        elif 0.05 < pval <= 0.1: return (sym + "t") if do_trend else '' 
        elif 0.01 < pval <= 0.05: return sym
        elif 0.001 < pval <= 0.05: return sym*2
        else: return sym*3
    if len(data) == 2:
        return [
            pval_to_star(test_fn(data[0], data[1]), "*")
        ]
    elif len(data) == 3:
        return [
            pval_to_star(test_fn(data[0], data[1]), "*"),
            pval_to_star(test_fn(data[0], data[2]), "*"),
            pval_to_star(test_fn(data[1], data[2]), "#"),
        ]
    else: raise ValueError(f"{len(data)=}")
        
def cmp_to_words_2_gr(
    data: list[np.ndarray], test_fn: callable, do_trend=False,
) -> list[str]:
    if len(data) != 2: raise ValueError(f"{len(data)=}")
    pval = test_fn(data[0], data[1])
    if (0.001<pval<=0.05) or (do_trend and 0.05<pval<=0.1):
        return [ f"p={pval:0.3f}", ]
    if pval <= 0.001:
        return [ "p<0.001", ]
    else:
        return ["", ]

_multi_boxplot_pairw_cmp_2_3gr_default_style = dict(
    fontsize=15, font='DejaVu Sans Mono',
)
def multi_boxplot_pairw_cmp_2_3gr(
    ax, stars: list[str], 
    style=None,
):
    style = style or _multi_boxplot_pairw_cmp_2_3gr_default_style
    if len(stars) == 1:
        p01 = stars[0]
        if p01:
            ax.annotate(p01, (5/8, 0.9), xycoords="axes fraction", **style)
    elif len(stars) == 3:
        p01, p02, p12 = stars
    #     _s = dict( fontfamily='DejaVu Sans') # 'ᵀ' sub/superscripts unfortunately require careful font selection
        if p01:
            ax.annotate(p01, (5/12, 0.9), xycoords="axes fraction", **style)
        if p02:
            ax.annotate(p02, (9/12, 0.9), xycoords="axes fraction", **style)
        if p12:
            ax.annotate(p12, (9/12, 0.85), xycoords="axes fraction", **style)
    else: raise ValueError(f"{len(stars)=}")



def test_t_ind(a, b, axis=None) -> float:
    return stats.ttest_ind(a, b, axis=axis, equal_var=False).pvalue
def test_mwu(a, b, axis=None) -> float:
    return stats.mannwhitneyu(a, b, axis=axis).pvalue
    
def plotname(): 
    return get_caller(1).__name__.removeprefix('plot_')

def boxplot_main(ax, gdata, opts):
    # from lib.statplots import plt_multi_boxplot
    opts = get_opts(opts)
    'macro-like, it just reduces number of lines dramatically'
    plt_multi_boxplot(ax, gdata, opts['group_labels'], opts['group_colors'])
    stars = opts['cmp2stars'](gdata, opts['test2'], opts['cmp_do_trends'])
    opts['boxplot_stars'](ax, stars)
# </editor-fold> # plotter_util

# <editor-fold desc="plotters_behav">

def plot_behav_24h_2gr_3gr_v1(activity_ts_v0, opts=None):
    activity_t = activity_ts_v0
    ENV = get_opts(opts)
    SAVE_P = ENV['outd'] / f"{plotname()}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", "дыхательный коэффициент", "потребление O2, Л/кг/ч"]
    elif ENV['lang']==LANG_EN:
        title = ["distance travelled, m/h", 
                 "food consumed, g/h", "respiratory exchange coefficient", "O2 consumed, L/kg/h"]
    colors = ENV['group_colors']
    labels = ENV['group_labels']
    if len(groups) == 3:
        linestyles, markers = ["-", "--", ":"], ["o", "^", "s"]
    elif len(groups) == 2:
        linestyles, markers = [None]*3, [None]*3
    else: raise ValueError(f"{len(groups)=}")
    # panel_h = 210 * c_mm2in
    # hspace = 20*c_mm2in /(panel_h/4)
    
    plot_cols = [cDist, cFeed, cRer, cO2]    
    X_TICKS = np_arange(0, 6, 5).astype(int)
    x = np.arange(0, 24) + 1
    
    fig, axs = plt.subplots(4, 1, figsize=(c_A4_w-1, c_A4_h-2), layout="compressed")#gridspec_kw=dict(hspace=0.27))
    axs = axs.ravel()
    
    for ax, col in zip(axs, plot_cols):
        # aggregate data
        accum1 = []
        # t = ts[col]
        for group in groups:
            accum2 = []
            for animal in data_animals[group]:
                ti = activity_t.loc[col, group, animal]
                y = ti.Y.values
                # compensate for 2 samples / h
                # RER is not compensated because it's a "flow" measure, while food & distance are "stock" measures
                if col == cDist:
                    y = y / 100 * 2
                elif col == cO2:
                    y = y / 1000 * 2
                elif col == cFeed:
                    y = y * 2
                y_p = nonparametric.periodic_avg(y, 48)
                assert len(y_p) == 48
                y_h = nonparametric.binned_avg(y_p, 2)
                assert len(y_h) == 24
                roll = 16
                y_r = np.roll(y_h, roll)
                assert y_r[roll] == y_h[0], np.array(list(zip(range(25), y_h, y_r)))
                accum2.append(y_r)
            accum1.append(accum2)
        # ---------
        gdata = [np.stack(g, axis=-1) for g in accum1]
        del accum1, accum2
        
        
        ms = np.stack([g.mean(axis=-1,) for g in gdata])
        stes = np.stack([g.std(axis=-1)/ np.sqrt(g.shape[-1]) for g in gdata])
        
#         mwus = stats.mannwhitneyu(grps[0], grps[1], axis=-1).pvalue
#         plot_data[col] = [ms, stes, mwus]
        for m, ste, c, ls, mark, lbl in zip(ms, stes, colors, linestyles, markers, labels):
            ax.plot(x, m, c=c, ls=ls, marker=mark, markersize=3, label=lbl)
            ax.fill_between(x, m-ste, m+ste, color=c, alpha=0.2)
            ax.set_xticks([1, 6, 12, 18, 24], [1, 6, 12, 18, 24])
            ax.set_xticks(np.arange(24), minor=True)
            ax.set_xlim(1, 24)
        
        ax.axvspan(0.0+1, 2.5+1, color="k", alpha=0.08)
        ax.axvspan(15.5+1, 23.0+1, color="k", alpha=0.08)
        
        # plot pvalues
        top = (ms+stes).max(axis=0)
        span = top.max() - (ms-stes).min()
        ann1 = top + 0.12*span # line on which pvalues rest
        ann2 = ann1 + 0.12*span
        ann3 = ann2 + 0.12*span
        if len(groups) == 2:
            p = test_t_ind(gdata[0], gdata[1], axis=-1)
            l1 = (p<=0.05) 
            l2 = (p<=0.01)
            l3 = p<=0.001
            for l, ann in [(l1, ann1), (l2, ann2), (l3, ann3)]:
                ax.plot(x[l], ann[l], c="k", marker="*", ls="none")
        elif len(groups) == 3:
            for dat, c, ann in zip(gdata[1:], colors[1:], [ann1, ann2]):
                p = test_t_ind(gdata[0], dat, axis=-1)
                l1 = (p<=0.05) & (p>0.01)
                l2 = (p<=0.01) & (p>0.001)
                l3 = p<=0.001
    #             ax.plot(x, ann, lw=0.3, c="k", alpha=0.1)
                ax.plot(x[l1], ann[l1], c=c, marker="$1$", ls="none")
                ax.plot(x[l2], ann[l2], c=c, marker="$2$", ls="none")
                ax.plot(x[l3], ann[l3], c=c, marker="$3$", ls="none")
        
        # pvalue
        y_pval = np.max(np.stack(ms, axis=-1), axis=-1)
        y_scale = (y_pval.max() - y_pval.min())
#         for p, coef in ((0.05, .2), (0.01, .3), (0.001, .4)):  
#             ax.plot(x[mwus<p], coef*y_scale+y_pval[mwus<p], ls="none", color="k", marker="*", markersize=5)

    for i in range(4):
        axs[i].set_title(title[i])#, y=0.9)
    axs[2].set_ylim((0.85, 1.05))
    axs[0].legend(loc="upper left", facecolor="none", edgecolor="none", ncols=3)
    
    letters(axs, (0.01, 1.03))
    figsave(SAVE_P, fig)

# def plot_behav_24h_3gr_v1(activity_ts_v0, opts=None):
#     activity_t = activity_ts_v0
#     ENV = get_opts(opts)
#     SAVE_P = ENV['outd'] / f"{plotname()}.svg"
#     groups = ENV['groups']
#     data_animals = ENV['data_animals']
#     figsave = ENV['figsave']
#     letters = ENV['letters']
#     if ENV['lang'] == LANG_RU:
#         title = ["пройденный путь, м/ч", "съеденный корм, г/ч", "дыхательный коэффициент"]
#     elif ENV['lang']==LANG_EN:
#         title = ["distance travelled, m/h", 
#                  "food consumed, g/h", "respiratory exchange coefficient", "O2 consumed, L/kg/h"]
#     colors = ENV['group_colors']
    
#     # panel_h = 210 * c_mm2in
#     # hspace = 20*c_mm2in /(panel_h/4)
    
#     plot_cols = [cDist, cFeed, cRer, cO2]    
#     X_TICKS = np_arange(0, 6, 5).astype(int)
#     x = np.arange(0, 24) + 1
    
#     fig, axs = plt.subplots(4, 1, figsize=(c_A4_w-1, c_A4_h-2), layout="compressed")#gridspec_kw=dict(hspace=0.27))
#     axs = axs.ravel()
    
#     for ax, col in zip(axs, plot_cols):
#         # aggregate data
#         accum1 = []
#         # t = ts[col]
#         for group in groups:
#             accum2 = []
#             for animal in data_animals[group]:
#                 ti = activity_t.loc[col, group, animal]
#                 y = ti.Y.values
#                 # compensate for 2 samples / h
#                 # RER is not compensated because it's a "flow" measure, while food & distance are "stock" measures
#                 if col == cDist:
#                     y = y / 100 * 2
#                 elif col == cO2:
#                     y = y / 1000 * 2
#                 elif col == cFeed:
#                     y = y * 2
#                 y_p = nonparametric.periodic_avg(y, 48)
#                 assert len(y_p) == 48
#                 y_h = nonparametric.binned_avg(y_p, 2)
#                 assert len(y_h) == 24
#                 roll = 16
#                 y_r = np.roll(y_h, roll)
#                 assert y_r[roll] == y_h[0], np.array(list(zip(range(25), y_h, y_r)))
#                 accum2.append(y_r)
#             accum1.append(accum2)
#         # ---------
#         gdata = [np.stack(g, axis=-1) for g in accum1]
#         del accum1, accum2
        
        
#         ms = np.stack([g.mean(axis=-1,) for g in gdata])
#         stes = np.stack([g.std(axis=-1)/ np.sqrt(g.shape[-1]) for g in gdata])
        
# #         mwus = stats.mannwhitneyu(grps[0], grps[1], axis=-1).pvalue
# #         plot_data[col] = [ms, stes, mwus]
#         for m, ste, c, ls, mark, lbl in zip(ms, stes, colors, ["-", "--", ":"], ["o", "^", "s"], ["K", "35", "37"]):
#             ax.plot(x, m, c=c, ls=ls, marker=mark, markersize=3, label=lbl)
#             ax.fill_between(x, m-ste, m+ste, color=c, alpha=0.2)
#             ax.set_xticks([1, 6, 12, 18, 24], [1, 6, 12, 18, 24])
#             ax.set_xticks(np.arange(24), minor=True)
#             ax.set_xlim(1, 24)
        
#         ax.axvspan(0.0+1, 2.5+1, color="k", alpha=0.08)
#         ax.axvspan(15.5+1, 23.0+1, color="k", alpha=0.08)
        
#         # plot pvalues
#         top = (ms+stes).max(axis=0)
#         span = top.max() - (ms-stes).min()
#         ann1 = top + 0.12*span # line on which pvalues rest
#         ann2 = ann1 + 0.12*span 
#         for dat, c, ann in zip(gdata[1:], colors[1:], [ann1, ann2]):
#             p = test_t_ind(gdata[0], dat, axis=-1)
#             l1 = (p<=0.05) & (p>0.01)
#             l2 = (p<=0.01) & (p>0.001)
#             l3 = p<=0.001
# #             ax.plot(x, ann, lw=0.3, c="k", alpha=0.1)
#             ax.plot(x[l1], ann[l1], c=c, marker="$1$", ls="none")
#             ax.plot(x[l2], ann[l2], c=c, marker="$2$", ls="none")
#             ax.plot(x[l3], ann[l3], c=c, marker="$3$", ls="none")
        
#         # pvalue
#         y_pval = np.max(np.stack(ms, axis=-1), axis=-1)
#         y_scale = (y_pval.max() - y_pval.min())
# #         for p, coef in ((0.05, .2), (0.01, .3), (0.001, .4)):  
# #             ax.plot(x[mwus<p], coef*y_scale+y_pval[mwus<p], ls="none", color="k", marker="*", markersize=5)

#     for i in range(4):
#         axs[i].set_title(title[i])#, y=0.9)
#     axs[2].set_ylim((0.85, 1.05))
#     axs[0].legend(loc="upper left", facecolor="none", edgecolor="none", ncols=3)
    
#     letters(axs, (0.01, 1.03))
#     figsave(SAVE_P, fig)

def plot_mass_glucose_v0(ivf_features_v0: pd.DataFrame, opts=None):
    ENV = get_opts(opts)
    save_to = ENV['outd'] / f"{plotname()}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    if  ENV['lang'] == LANG_RU:
        title1 = "масса, г"
        title2 = "AUC глюкозы, ммоль/л×120 мин"
    else:
        title1 = "mass, g"
        title2 = "glucose AUC, mmol/L×120 m"
    # collect data
    gdata_mom = []
    for group in groups:
        animals = data_animals[group]
        t02 = ivf_features_v0 #tbls['ivf_features_v0']
        d = t02[t02['animal_no'].apply(lambda x: x in animals)]
        gdata_mom.append(d)
    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w, 3), gridspec_kw=dict(wspace=0.3))
    axs = axs.ravel()
    ax = axs[0]
    gdata = [d['Mass_13.05'] for d in gdata_mom]
    boxplot_main(ax, gdata, ENV)
    ax.set(ylim=(30, 50), title=title1, yticks=[30, 35, 40, 45, 50])
    ax = axs[1]
    gdata = [d['AUG'] for d in gdata_mom]
    boxplot_main(ax, gdata, ENV)
    ax.set(title=title2)
    ENV['letters'](axs, (0.02, 0.90))
    ENV['figsave'](save_to, fig)

def plot_behav_24h_3gr_v0(activity_ts_v0: pd.DataFrame, opts=None):
    activity_t = activity_ts_v0 # tbls['activity_ts_v0']
    ENV = get_opts(opts)
    SAVE_P = ENV['outd'] / f"{plotname()}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", "дыхательный коэффициент"]
    elif ENV['lang']==LANG_EN:
        title = ["distance travelled, m/h", 
                 "food consumed, g/h", "respiratory exchange coefficient"]
    colors = ENV['group_colors']
    
    panel_h = 210 * c_mm2in
    hspace = 20*c_mm2in /(panel_h/3)
    
    plot_cols = [cDist, cFeed, cRer]    
    X_TICKS = np_arange(0, 6, 5).astype(int)
    x = np.arange(0, 24) + 1
    
    fig, axs = plt.subplots(3, 1, figsize=(c_A4_w, panel_h), gridspec_kw=dict(hspace=0.27))
    axs = axs.ravel()
    
    for ax, col in zip(axs, plot_cols):
        # aggregate data
        accum1 = []
        # t = ts[col]
        for group in groups:
            accum2 = []
            for animal in data_animals[group]:
                ti = activity_t.loc[col, group, animal]
                y = ti.Y.values
                # y = ti_unpack(t_animal(t, animal))[-1]
                if col == cDist:
                    y = y / 100
                y_p = nonparametric.periodic_avg(y, 48)
                assert len(y_p) == 48
                y_h = nonparametric.binned_avg(y_p, 2)
                assert len(y_h) == 24
                roll = 16
                y_r = np.roll(y_h, roll)
                assert y_r[roll] == y_h[0], np.array(list(zip(range(25), y_h, y_r)))
                accum2.append(y_r)
            accum1.append(accum2)
        # ---------
        gdata = [np.stack(g, axis=-1) for g in accum1]
        del accum1, accum2
        
        if col in [cDist, cFeed]:
            gdata = [g *2 for g in gdata] # compensate for 2 samples / h
            # RER is not compensated because it's a "flow" measure, while food & distance are "stock" measures
        ms = np.stack([g.mean(axis=-1,) for g in gdata])
        stes = np.stack([g.std(axis=-1)/ np.sqrt(g.shape[-1]) for g in gdata])
        
#         mwus = stats.mannwhitneyu(grps[0], grps[1], axis=-1).pvalue
#         plot_data[col] = [ms, stes, mwus]
        for m, ste, c, ls, mark, lbl in zip(ms, stes, colors, ["-", "--", ":"], ["o", "^", "s"], ["K", "35", "37"]):
            ax.plot(x, m, c=c, ls=ls, marker=mark, markersize=3, label=lbl)
            ax.fill_between(x, m-ste, m+ste, color=c, alpha=0.2)
            ax.set_xticks([1, 6, 12, 18, 24], [1, 6, 12, 18, 24])
            ax.set_xticks(np.arange(24), minor=True)
            ax.set_xlim(1, 24)
        
        ax.axvspan(0.0+1, 2.5+1, color="k", alpha=0.08)
        ax.axvspan(15.5+1, 23.0+1, color="k", alpha=0.08)
        
        # plot pvalues
        top = (ms+stes).max(axis=0)
        span = top.max() - (ms-stes).min()
        ann1 = top + 0.12*span # line on which pvalues rest
        ann2 = ann1 + 0.12*span 
        for dat, c, ann in zip(gdata[1:], colors[1:], [ann1, ann2]):
            p = test_t_ind(gdata[0], dat, axis=-1)
            l1 = (p<=0.05) & (p>0.01)
            l2 = (p<=0.01) & (p>0.001)
            l3 = p<=0.001
#             ax.plot(x, ann, lw=0.3, c="k", alpha=0.1)
            ax.plot(x[l1], ann[l1], c=c, marker="$1$", ls="none")
            ax.plot(x[l2], ann[l2], c=c, marker="$2$", ls="none")
            ax.plot(x[l3], ann[l3], c=c, marker="$3$", ls="none")
        
        # pvalue
        y_pval = np.max(np.stack(ms, axis=-1), axis=-1)
        y_scale = (y_pval.max() - y_pval.min())
#         for p, coef in ((0.05, .2), (0.01, .3), (0.001, .4)):  
#             ax.plot(x[mwus<p], coef*y_scale+y_pval[mwus<p], ls="none", color="k", marker="*", markersize=5)
    
    axs[0].set_title(title[0])#, y=0.9)
    axs[1].set_title(title[1])#, y=0.9)
    axs[2].set_title(title[2])#, y=0.9)
    axs[0].set_ylim((0, 220))
    axs[1].set_ylim((0,0.7))
#     axs[2].set_ylim((0.85, 1.05))
    axs[0].legend(loc="upper left", facecolor="none", edgecolor="none", ncols=3)
    
    letters(axs, (0.01, 1.03))
    figsave(SAVE_P, fig)


def plot_hippocampus_startle_v0(ivf_features_v0: pd.DataFrame, opts=None):
    ENV = get_opts(opts)
    save_to = ENV['outd'] / f"{plotname()}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_EN:
        title = ["max startle reflex amplitude, a.u.", "hippocampus volume, mkl"]
    elif ENV['lang'] == LANG_RU:
        title = ["макс. амплитуда стартл-рефлекса, у.е.", "объем гиппокампа, мкл"]
    # else: raise Exception(ENV['lang'])
    t02 = ivf_features_v0 #tbls['ivf_features_v0']
    animals = t02['animal_no']
    
    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w, 3), gridspec_kw=dict(wspace=0.25))
    axs = axs.ravel()
    
    selectors = [animals.apply(lambda x: x in data_animals[g]).values for g in groups]
    # startle
    ax = axs[0]
    gdata = [t02.loc[s, 'Startle_max'] for s in selectors]
    boxplot_main(ax, gdata, opts)
    ax.set(ylim=(2., 3.6))
    # hippocampus
    ax = axs[1]
    gdata = [
        np.concatenate((
            t02.loc[s, "Hippoc_L"].values,
            t02.loc[s, "Hippoc_R"].values,
        )) for s in selectors
    ]
    boxplot_main(ax, gdata, opts)
    ax.set(ylim=(22, 36))
    axs[0].set_title(title[0])
    axs[1].set_title(title[1])
    letters(axs, (0.02, 0.90))
    figsave(save_to, fig)

     
def _plot_ts_night(
    ax, 
    night_begin = 16.0,
    night_len = 10.0,
    period = 24.0,
    ndays = 5,
):
    beg = np_arange(night_begin, period, ndays)
    end = beg + night_len
    for b, e in zip(beg, end): 
        ax.axvspan(b,e, alpha=0.08, color="k")
def _plot_ts_individ(ax, ti, col, opts=None):
    ENV = get_opts(opts)
    win = ENV['ts_avg_window']
    xticks = ENV['ts_x_ticks']
    colors = ENV['group_colors']
    hw = win // 2
    g, a, x, y = ti_unpack(ti)
    c = colors[g]
    y_avg = nonparametric.conv_avg(y, win)
    ax.plot(x, y, c=c, lw=0.5)
    ax.plot(x[hw:-hw], y_avg[hw:-hw], c=c, lw=1.5)    
    ax.set_xticks(xticks, xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    _ts_ylim = {
        cDist: (0, 12000),
        cFeed: (0., .75),
        cRer: (0.7, 1.2),
        cO2: (2000, 5000),
        cCO2: (2000, 5000),
        cRelFeed: (0, 25),
        cFeedByDist: (0, 40),
    }
    ylim = _ts_ylim.get(col)
    ax.set_ylim(*ylim) if ylim else None
# _x_ticks = np.arange(12, 12 + 6*19, 6)
def _plot_cosinor_fit(ax, _x_fit, model, ri):
    y_fit = cosinor.predict(_x_fit, link=model.link, **ri)
    ax.plot(_x_fit, y_fit, c='r')
    if type(model) == cosinor.Normal1D:
        ax.fill_between(_x_fit, y_fit - ri['mesor_CI'], y_fit + ri['mesor_CI'], color='red', alpha=0.3)
    ax.axvline(ri['acrophase_h'], c='r', lw=0.5)
    ax.axhline(model.link(ri['mesor']), c="red", lw=0.5)
def plot_cosinor_individ(ax, ti, ri, model, col, opts=None):
    'plot on single axis (we can build upon this part)'
    ENV= get_opts(opts)
    colors = ENV['group_colors']
    g, a, x, y = ti_unpack(ti)
    color = colors[g]
    _x_fit = np.arange(15., 15. + 204 * 0.5, 0.5)
#     y_smooth = npmetric.conv_avg(y, 6)
    f = plot_cosinor_individ
    f.plot_ts(ax, ti, col, ENV)    
    f.plot_night(ax)
    f.plot_model(ax, _x_fit, model, ri)
f = plot_cosinor_individ
f.plot_night = _plot_ts_night
f.plot_ts = _plot_ts_individ
f.plot_model = _plot_cosinor_fit
del f

def plot_behav_averages_v4(activity_ts_v0, ivf_features_v0, axs=None, opts=None):
    activity_t = activity_ts_v0 # tbls['activity_ts_v0']
    feat_t = ivf_features_v0 # tbls['ivf_features_v0']
    
    ENV = get_opts(opts)
    save_to = ENV['outd'] / f"{plotname()}.svg"
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    
    if ENV['lang'] == LANG_RU:
        title = "!!!!!!!!!"
    elif ENV['lang'] == LANG_EN:
        title = [
            "food consumed, g/h",
            "food consumed, g /(h × kg)",
            "distance travelled, m/h",
            "distance/food, m/g",
            "respiratory exchange ratio",
            "O$_2$ consumed, L /(h × kg)",
        ]
    opts = [
        dict(ylim = (0.15, 0.40), yticks = np_arange(0.15, 0.05, 5)), # feed
        dict(ylim = (5.3, 7.3)), # feed/mass
        dict(ylim=(48, 140), yticks = np_arange(50, 20, 5)), # dist
        dict(), # dist/feed
        dict(ylim=(0.93, 1.08), yticks = np_arange(0.93, 0.03, 5)), # rer
        dict() # o2
    ]
    if len(ENV['groups']) == 3:
        opts[1]['ylim']=(5, 8)
    ln_m_to_ln_cm = np.log(100)
    m_to_cm = 100
    assert np.isclose(np.log(5),np.log(5*m_to_cm) - ln_m_to_ln_cm, )
    log10_to_ln = np.log(10)
    assert np.isclose(np.log10(5)*log10_to_ln, np.log(5))
    
    no_fig = axs is None
    if no_fig:
        fig, axs = plt.subplots(3, 2, figsize=(c_A4_w, 3*3), sharex=True, gridspec_kw=dict(wspace=0.2, hspace=0.2))
        axs = axs.ravel()
    
    def summarize(col, animals):
            values = []
            for animal in animals:
                if col in ENV['data_columns']:
                    y = activity_t.loc[col, :, animal].Y.values
                    # y = t_animal(ts[col], animal).Y.values
                else: y = None

                if col == cDist:
                    ys = y.mean() /100 * 2
                elif col == cFeed:
                    ys = y.mean() * 2
                elif col == cRer:
                    ys = y.mean()
                elif col == cO2:
                    ys = y.mean()/1000
                elif col == "feed/mass": 
                    y = activity_t.loc[cFeed, :, animal].Y.values
                    # y = t_animal(ts[cFeed], animal).Y.values
                    mass = feat_t.loc[feat_t.animal_no == animal, 'Mass_13.05']
                    # mass = t02['Mass_13.05'][t02.animal_no == animal]
                    ys = y.mean() * 2 * (1000 / mass)
                # elif col == "dist/feed": # special case
                values.append(ys)
            return np.stack(values)
        
    gdatas = dict()
    for col in [cFeed, "feed/mass", cDist, cRer, cO2]:
        gdata = [summarize(col, data_animals[g]) for g in groups]
        gdatas[col] = gdata
    gdatas["dist/feed"] = [d/f for d, f in zip(gdatas[cDist], gdatas[cFeed])]
    for icol, col in enumerate([cFeed, "feed/mass", cDist, "dist/feed", cRer, cO2]):
        gdata = gdatas[col]
        ax = axs[icol]
        boxplot_main(ax, gdata, ENV)

    for i in range(6):
        axs[i].set(title=title[i], **opts[i])

    plt_delete_ticks(axs[:4])
    if no_fig:
        ENV['letters'](axs, (0.02, 0.90))
        ENV['figsave'](save_to, fig)
     
def plot_cosinor_example_v0(ivf_ts_cosinor_v0, activity_ts_v0, axs = None, opts=None):
    ENV = get_opts(opts)
    cosinor_t = ivf_ts_cosinor_v0 #tbls['ivf_ts_cosinor_v0']
    activity_t = activity_ts_v0 #tbls['activity_ts_v0']
    animal = 1
    SAVE_P = ENV['outd'] / f"{plotname()} {animal=}.svg"
    xticks = ENV['ts_x_ticks']
    AVG_WINDOW = ENV['ts_avg_window']
    if ENV['lang'] == LANG_RU:
        ylabel = [
            "пройденный путь, м",
            "съеденный корм, г",
            "дыхательный коэф.",
        ]
    elif ENV['lang'] == LANG_EN:
        ylabel = [
            "distance travelled, m",
            "food consumed, g",
            "respiratory exchange ratio",
        ]
    cosinor_x = np_arange(15, .5, 204)
    color = "#7d2eaa"
#     color = ENV_colors[-1]
    plot_cols = [cDist, cFeed, cRer]
    
    no_fig = axs is None
    if no_fig:
        fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw=dict(hspace=0.1),)
        axs = axs.ravel()
    _x_fit = np.arange(15., 15. + 204 * 0.5, 0.5)
    HW = AVG_WINDOW // 2
    for ax, col in zip(axs, plot_cols):
        ti = activity_t.loc[col, :, animal]
        x = ti.X.values
        y = ti.Y.values
        if col in [cRer, ]:
            model= N24
            ri = cosinor_t.loc['N24', col, :, 1].iloc[0]
        else: 
            model = P24
            ri = cosinor_t.loc['P24', col, :, 1].iloc[0]
        plot_cosinor_individ.plot_night(ax)
        y_fit = cosinor.predict(cosinor_x, link=model.link, **ri)
        mesor = model.link(ri['mesor'])
        if col == cDist:
            y = y / 100
            y_fit = y_fit / 100
            mesor = mesor / 100
        y_avg = nonparametric.conv_avg(y, AVG_WINDOW)
        ax.plot(x, y, c=color, lw=0.5)
        ax.plot(x[HW:-HW], y_avg[HW:-HW], c=color, lw=1.5,)
        ax.plot(_x_fit, y_fit, c='r', alpha=0.7)
        ax.axvline(ri['acrophase_h'], c='r', lw=0.5)
        ax.axhline(mesor, c="red", lw=0.5)
        ax.set_xticks(xticks, xticks)
        ax.set_xlim(xticks[0], xticks[-1])
    axs[0].set(
        ylabel = ylabel[0],
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
    )
    axs[1].set(
        ylabel=ylabel[1],
        ylim = (0, 0.5),
    )
    axs[2].set(
        ylabel=ylabel[2],
        ylim=(0.85, 1.05),
        yticks=np_arange(0.85, .05, 4),
    )
    fig.align_ylabels(axs)
    plt_delete_ticks(axs[:2], 'x')
    ENV['letters'](axs, (0.004, 0.85))
    if no_fig:
        ENV['figsave'](SAVE_P, fig)

def plot_cosinor_params_v3(ivf_ts_cosinor_v0, opts=None):
    ENV = get_opts(opts)
    cosinor_t = ivf_ts_cosinor_v0 #tbls["ivf_ts_cosinor_v0"]
    
    def mb2(
        ax, 
        group_data: list[np.ndarray],
        labels: list[str]= None,
        colors: list = None,
        style_kw = None
    ):
        colors = iter(colors)
        style_kw = plt_multi_boxplot.boxplot_kw if style_kw is None else style_kw
        scatter_kw = plt_multi_boxplot.scatter_kw
        for i, data in enumerate(group_data):
            x_pseudo = i + np.random.uniform(-0.2, 0.2, len(data))
            ax.boxplot(data, positions=[i, ], manage_ticks=False, vert=False, **style_kw)
            ax.scatter(data, x_pseudo, color=next(colors), **scatter_kw)
        if labels:
            ax.set_yticks(np.arange(len(labels)), labels, verticalalignment="center") # rotation=90,

    def cmp(
        ax, stars: list[str], 
        style=None,
    ):
        style = style or _multi_boxplot_pairw_cmp_2_3gr_default_style
        style = dict(rotation=90, **style)
        if len(stars) == 1:
            p01 = stars[0]
            if p01:
                ax.annotate(p01, (0.9, 1-5/8), xycoords="axes fraction", **style)
        elif len(stars) == 3:
            p01, p02, p12 = stars
        #     _s = dict( fontfamily='DejaVu Sans') # 'ᵀ' sub/superscripts unfortunately require careful font selection
            if p01:
                ax.annotate(p01, (0.9, 6/12), xycoords="axes fraction", **style)
            if p02:
                ax.annotate(p02, (0.9, 2/12), xycoords="axes fraction", **style)
            if p12:
                ax.annotate(p12, (0.80, 2/12), xycoords="axes fraction", **style)
        else: raise ValueError(f"{len(stars)=}")
            
    def bm2(ax, gdata, **kw):
        'macro-like, it just reduces number of lines drammatically'
        mb2(ax, gdata[::-1], ENV['group_labels'][::-1], ENV['group_colors'][::-1])
        stars = ENV['cmp2stars'](gdata, ENV['test2'], ENV['cmp_do_trends'])
        cmp(ax, stars)

    
    save_to = ENV['outd'] / f"{plotname()}.svg"
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    if ENV['lang']==LANG_RU:
        letters = LETTERS_RU
        title = ["пройденный путь", "съеденный корм", "дыхательный коэф."]
        ylabel = [
            ["амплитуда, log10(м)", "амплитуда, log10(г)", "амплитуда, у.е."],
            ["мезор, log10(m)", "мезор, log10(г)", "мезор, у.е."],
            "акрофаза, ч",
        ]
    elif ENV['lang']==LANG_EN:
        letters = LETTERS_EN
        title = ["distance travelled", "food consumed", "respiratory exchange ratio"]
        ylabel = [
            ["amplitude, log10(m)", "amplitude, log10(g)", "amplitude, a.u."],
            ["mesor, log10(m)", "mesor, log10(g)", "mesor, a.u."],
            "acrophase, h",
        ]
    letters = plt_add_letters(ENV['letters'])
    # conversion coefs
    ln_m_to_ln_cm = np.log(100)
    m_to_cm = 100
    assert np.isclose(
        np.log(5),
        np.log(5*m_to_cm) - ln_m_to_ln_cm, 
    )
    log10_to_ln = np.log(10)
    assert np.isclose(np.log10(5)*log10_to_ln, np.log(5))

    fig, axs = plt.subplots(4, 3, figsize=(c_A4_w, c_A4_h), sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.3))
    
    for icol, col in enumerate([cDist, cFeed, cRer, cO2]):
        if col in [cRer, cO2]:
            model= N24
            modeln = 'N24'
        else: 
            model = P24
            modeln = 'P24'
        # gdatas = [cos[cos['animal'].apply(
        #                 lambda x: x in data_animals[g]
        #             )] for g in groups]
        gdatas = [cosinor_t.loc[modeln, col, g, data_animals[g]] for g in groups]
            
        for irow, row in enumerate(['amplitude', 'mesor', 'acrophase_h']):
            gdata = [d[row].values for d in gdatas]
#             group_data = {k: v[row].values for k, v in gb.items()}
            if row in ['amplitude', 'mesor']: # adjust y-values, but not x-values (acrophase)
                if col == cDist: # convert to meters, convert to log10
                    gdata = [(d - ln_m_to_ln_cm) / log10_to_ln for d in gdata]
                if col == cFeed: # convert to log10
                    gdata = [d / log10_to_ln for d in gdata]
                if col == cO2:
                    gdata = [d/1000 for d in gdata] # to liters
#             mwu = stats.mannwhitneyu(group_data["WT"], group_data["IVF"])
            ax = axs[icol, irow, ]
            bm2(ax, gdata)
            ax.set_ylim(-0.5, 1.8 if len(ENV['groups']) == 2 else 2.8)
            if irow == 0:
                ax.set_ylabel(col)
    plt_delete_ticks(axs[:, 1:].ravel(), 'y')
    axs[0, 0].set_title("amplitude")
    axs[0, 1].set_title("mesor")
    axs[0, 2].set_title("acrophase")
    
    axs[0, 0].set_ylabel("distance travelled")
    axs[0, 0].set_xlabel("amplitude, log10(m)")
    axs[0, 1].set_xlabel("mesor, log10(m)")
    axs[0, 2].set_xlabel("acrophase, h")
    
    axs[1, 0].set_ylabel("food consumed")
    axs[1, 0].set_xlabel("amplitude, log10(g)")
    axs[1, 1].set_xlabel("mesor, log10(g)")
    axs[1, 2].set_xlabel("acrophase, h")
    
    axs[2, 0].set_ylabel("respiratory exchange ratio")
    axs[2, 0].set_xlabel("amplitude, a.u.")
    axs[2, 0].set_xlim(0.01, 0.08)
    axs[2, 1].set_xlabel("mesor, a.u.")
    # axs[2, 1].set_xlim(0.92, 1.08)
    axs[2, 2].set_xlabel("acrophase, h")
    
    axs[3, 0].set_ylabel("oxygen consumed")
    axs[3, 0].set_xlabel("amplitude, L/kg")
    axs[3, 0].set_xlim(0.3, 0.9)
    axs[3, 1].set_xlabel("mesor, L/kg")
    axs[3, 1].set_xlim(2.2, 3.8)
    axs[3, 2].set_xlabel("acrophase, h")
    ENV['letters'](axs, (0.02, 0.90))
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)

def plot_behav_day_night_v3(activity_ts_v0, opts=None):
    ENV = get_opts(opts)
    activity_t = activity_ts_v0 #tbls["activity_ts_v0"]
    plot_cols = [cDist, cFeed, cRer, cO2]
    save_to = ENV['outd'] / f"{plotname()} {'_'.join(plot_cols)}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", "дыхательный коэф."]
        ylabel = ["темная фаза", "светлая фаза"]
    elif ENV['lang'] == LANG_EN:
        title = ["distance travelled, m/h", "food consumed, g/h", "respiratory exchange ratio"]
        ylabel = ["dark phase", "light phase"]
    fig, axs = plt.subplots(4, 2, 
        figsize=(c_A4_w-20*c_mm2in, 290 * c_mm2in), sharex=True,
        gridspec_kw=dict(hspace=0.1, wspace=0.2),
    )
    for ax_pair, col in zip(axs, plot_cols):
        accum1 = []
        for group in groups:
            accum2 = []
            for animal in data_animals[group]:
                ti = activity_t.loc[col, group, animal]
                y = ti.Y.values           
                if col == cDist:
                    y = y / 100
                if col == cO2:
                    y = y / 1000
                y_p = nonparametric.periodic_avg(y, 48)
                assert len(y_p) == 48
                y_h = nonparametric.binned_avg(y_p, 2)
                assert len(y_h) == 24
                roll = 16
                y_r = np.roll(y_h, roll)
                assert y_r[roll] == y_h[0], np.array(list(zip(range(25), y_h, y_r)))
                accum2.append(y_r)
            accum1.append(accum2)
        gdata = [np.stack(g, axis=-1) for g in accum1]
        del accum1, accum2
        # ---------
        # average across dark/light periods
        dark = [(d[:3].sum(axis=0) + d[17:].sum(axis=0))/10 for d in gdata]
        light = [d[3:17].mean(axis=0) for d in gdata]
        # plot dark
        ax = ax_pair[0]
        boxplot_main(ax, dark, ENV)
        ax.axvspan(-1, 4, color="k", alpha=0.08)
        ax.set_xlim((-0.5, len(groups)-0.5))
        # plot light
        ax = ax_pair[1]
        boxplot_main(ax, light, ENV)
    plt_delete_ticks(axs[:-1].ravel(), 'x')
    # dist
    axs[0, 0].set(
        title="dark phase",
        
        ylim=(25, 85),
    )
    axs[0, 0].set_ylabel("distance travelled, m/h", fontsize=12)
    axs[0, 1].set(
        title = "light phase",
        ylim=(15, 65),
    )
    # feed
    axs[1, 0].set(
        yticks=np_arange(0.10, 0.05, 4),
        ylim=(0.07, 0.27),
    )
    axs[1, 0].set_ylabel("food consumed, g/h", fontsize=12)
    axs[1, 1].set(
        yticks = np_arange(0.05, 0.05, 3),
        ylim=(0.03, 0.19),
    )
    # rer
    axs[2, 0].set(
        yticks=[0.90, 0.95, 1.0, 1.05],
        ylim=(0.90, 1.05),
    )
    axs[2, 0].set_ylabel("respiratory exchange ratio", fontsize=12)
    axs[2, 1].set(
        yticks=[0.90, 0.95, 1.0, 1.05],
        ylim=(0.90, 1.05),
    )
    axs[3, 0].set_ylabel("O2 consumption, L/kg/h", fontsize=12)
    # axs[3, 1].set(
    #     yticks=[0.90, 0.95, 1.0, 1.05],
    #     ylim=(0.90, 1.05),
    # )
    # axs[0, 0].set_ylabel(ylabel[0], fontsize=12)
    # axs[0, 1].set_ylabel(ylabel[1], fontsize=12)    
    fig.align_ylabels(axs[:, 0])
    # print(axs.shape)
    letters(axs, (0.02, 0.90))
    figsave(save_to, fig)

# </editor-fold> # plotter implementations

# <editor-fold desc="plotters_sleep">

def ckpt_fn_ivf_sleep_total_v0(src_dir):
    p = Path(src_dir)
    p = p / "Sleep_total.csv" if p.is_dir() else p
    st = pd.read_csv(p, sep="\t")
    def timeparse(s: str):
        d, t = s.split()
        dd, mm, yy = map(int, d.split("."))
        hh, mi, ss = map(int, t.split(":"))
        return datetime(yy, mm, dd, hh, mi, ss)
    st['bt'] = st['Start Time'].map(timeparse)
    st['et'] = st['End Time'].map(timeparse)
    
    st['dt'] = (st['et'] - st['bt']).map(lambda x: x.total_seconds())
    assert not (st['Duration']-st['dt']).any()
    st['ld'] = st['Light/Dark'].map(lambda x: 1 if x.lower()=="light" else 0)
    stg = st.groupby("Box")
    for a in range(1, 33):
        assert (stg.get_group(a)['bt'].diff().map(lambda x: x.total_seconds())[1:] > 0).all(), f"{a}"
        # assert sorted
    
    
    for a in range(1, 33):
        g = stg.get_group(a)
        _ = g['bt']
        _0 = _.iloc[0]
        dbt = ( (_ - _0).map(lambda x: x.total_seconds()) + (_0.hour*3600 + _0.minute*60+_0.second)) 
        st.loc[g.index, 'dbt'] = dbt / 3600
        st.loc[g.index, 'dbt_s'] = dbt.astype(int)
        _ = g['et']
        _0 = _.iloc[0]
        det = ((_ - _0).map(lambda x: x.total_seconds()) + (_0.hour*3600 + _0.minute*60+_0.second)) 
        st.loc[g.index, 'det_s'] = det
        st.loc[g.index, 'det'] = det / 3600
        nosleep = dbt.values - np.roll(det.values, 1)
        nosleep[0] = 0 # i dislike IEEE nans
        st.loc[g.index, 'nosleep_dt']=nosleep
    
    st['dbt_s'] = st['dbt_s'].astype(int)
    st['det_s'] = st['det_s'].astype(int)
    return st


def ckpt_fn_ivf_sleep_agg(sleep_total, opts=None):
    ENV = get_opts(opts)
    stg = sleep_total.groupby("Box")
    sleep_agg = []
    sleep_ts = {}
    for g in ENV['groups']:
        for a in ENV['data_animals'][g]:
            
            grp = stg.get_group(a)
            # plt.scatter(grp['dbt'], grp['dt'])
            total_time_h = np.diff(grp['dbt'].iloc[[0, -1]].values)[0]
            # print(total_time_h)
            sleep_agg.append({
                "anim": a,
                "group": g,
                "mean": grp['dt'].mean(),
                "med": grp['dt'].median(),
                "std": grp['dt'].std(),
                "q1": grp['dt'].quantile(0.25),
                "q3": grp['dt'].quantile(0.75),
                "day_avg": grp['dt'].sum() / total_time_h,
                "ns_mean": grp["nosleep_dt"].mean(),
                "ns_day_avg": grp["nosleep_dt"].sum() / total_time_h,
            })
            dbtv = grp["dbt"].values; dtv = grp["dt"].values
            xs = np.arange(15, 118, 0.5)
            ys = np.zeros_like(xs)
            for i, h in enumerate(np.arange(15, 118, 0.5)):
                ys[i] = dtv[(h < dbtv) & (dbtv< h+0.5)].sum() / 1800
            sleep_ts[a] = ys
            
    sleep_agg = pd.DataFrame(sleep_agg)
    sleep_ts = pd.DataFrame.from_dict(sleep_ts, orient="index", columns=xs).T
    return sleep_agg, sleep_ts
    
def plot_sleep_agg_dark_light_v0(sleep_ts, opts=None):
    ENV = get_opts(opts)
    _ = sleep_ts.index % 24
    is_dark = (_ > 16) | (_ <= 2)
    dark_s = sleep_ts.loc[is_dark].mean(axis="index") * 100
    light_s = sleep_ts.loc[~is_dark].mean(axis="index") * 100
    # print(ENV['groups'])
    wtg = ENV['data_animals'][0]
    ivfg = ENV['data_animals'][2]
    gdata_dark = [dark_s[ENV['data_animals'][g]] for g in ENV['groups']]
    gdata_light = [light_s[ENV['data_animals'][g]] for g in ENV['groups']]
    
    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w, 3),)# layout='compressed')
    axs = axs.ravel()
    ax = axs[0]
    boxplot_main(ax, gdata_dark, ENV)
    ax.set_yticks([15, 20, 25, 30, 35, 40])
    ax.set_title("sleep time in dark phase, %")
    ax = axs[1]
    boxplot_main(ax, gdata_light, ENV)
    ax.set_yticks([30, 40, 50, 60])
    ax.set_title("sleep time in light phase, %")
    ENV['figsave'](ENV['outd']/ f"{plotname()}.svg", fig)

def plot_sleep_agg_lengths_v0(sleep_agg, opts = None):
    ENV = get_opts(opts)
    gb = sleep_agg.groupby('group')
    gdata_mom = [gb.get_group(g) for g in ENV['groups']] 
    fig, axs = plt.subplots(1, 3, figsize=(c_A4_w, 3), layout='compressed')
    axs = axs.ravel()
    ax = axs[0]
    boxplot_main(ax, [gd['mean'] for gd in gdata_mom], ENV)
    ax.set_title("sleep episode length, s")
    ax.set_yticks([90, 120, 150, 180])
    
    ax = axs[1]
    boxplot_main(ax, [gd['ns_mean'] for gd in gdata_mom], ENV) # [wt["ns_mean"], ivf["ns_mean"]])
    ax.set_title("awakeness episode length, s")
    ax.set_yticks([120, 180, 240, 300])
    
    ax = axs[2]
    boxplot_main(ax, [gd["day_avg"] / 36 for gd in gdata_mom], ENV) #/3600*100 # [wt["day_avg"] /(3600/100), ivf["day_avg"]/(3600/100)])
    ax.set_title("sleep per day, %")
    ax.set_yticks([30, 40, 50, 60])
    ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)

def plot_awakeness_agg_v0(sleep_agg, opts=None):
    ENV = get_opts(opts)
    gb = sleep_agg.groupby('group')
    gdata_mom = [gb.get_group(g) for g in ENV['groups']] 
    fig, axs = plt.subplots(1, 3, figsize=(c_A4_w, 3), layout='compressed')
    axs = axs.ravel()
    axs[2].remove()
    ax = axs[0]
    boxplot_main(ax, [gd["ns_mean"] for gd in gdata_mom], ENV)#[wt["ns_mean"], ivf["ns_mean"]])
    ax.set_title("average awakeness time, s")
    ax = axs[1]
    boxplot_main(ax, [gd["ns_day_avg"]/36 for gd in gdata_mom], ENV)# [wt["ns_day_avg"] /(3600/100), ivf["ns_day_avg"]/(3600/100)])
    # [wt["day_avg"] /(3600/100), ivf["day_avg"]/(3600/100)]
    ax.set_title("total time awake per day, %")
    # ax.set_ylim(12, 17)
    ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)

def plot_sleep_agg_dark_light_v1(sleep_agg, sleep_ts, opts=None):
    ENV = get_opts(opts)
    _ = sleep_ts.index % 24
    is_dark = (_ > 16) | (_ <= 2)
    dark_s = sleep_ts.loc[is_dark].mean(axis="index") * 100
    light_s = sleep_ts.loc[~is_dark].mean(axis="index") * 100
    # print(ENV['groups'])
    wtg = ENV['data_animals'][0]
    ivfg = ENV['data_animals'][2]
    gdata_dark = [dark_s[ENV['data_animals'][g]] for g in ENV['groups']]
    gdata_light = [light_s[ENV['data_animals'][g]] for g in ENV['groups']]
    gb = sleep_agg.groupby('group')
    gdata_mom = [gb.get_group(g) for g in ENV['groups']] 
    
    fig, axs = plt.subplots(3, 2, figsize=(c_A4_w, c_A4_h),)# layout='compressed')
    axs = axs.ravel()
    ax = axs[0]
    boxplot_main(ax, [gd["day_avg"] / 36 for gd in gdata_mom], ENV)
    ax.set_title("total sleep time, % of day")
    ax.set_yticks([30, 35, 40, 45, 50, 55])
    ax = axs[1]
    boxplot_main(ax, [gd["ns_day_avg"]/36 for gd in gdata_mom], ENV)
    ax.set_title("total time awake, % of day")
    ax.set_yticks([45, 55, 65, 75])
    ax = axs[2]
    boxplot_main(ax, [gd['mean']/60 for gd in gdata_mom], ENV)
    ax.set_title("average sleep episode, min")
    # ax.set_yticks([90, 120, 150, 180])
    ax.set_yticks([1.5, 2.0, 2.5, 3.0])
    ax = axs[3]
    boxplot_main(ax, [gd['ns_mean']/60 for gd in gdata_mom], ENV)
    ax.set_title("average awakeness episode, min")
    # ax.set_yticks([120, 180, 240, 300])
    ax.set_yticks([2, 3, 4, 5])
    ax = axs[4]
    boxplot_main(ax, gdata_dark, ENV)
    ax.set_yticks([15, 20, 25, 30, 35, 40])
    ax.set_title("sleep time in dark phase, %")
    ax = axs[5]
    boxplot_main(ax, gdata_light, ENV)
    ax.set_yticks([30, 40, 50, 60])
    ax.set_title("sleep time in light phase, %")
    
    ENV['letters'](axs, (0.01, 0.92))
    ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)

# </editor-fold> # sleep


def plot_Aryana_thesis_no3(nonparametric_ts_sum_v0, activity_ts_v0, ivf_features_v0, opts =None):
    nps = nonparametric_ts_sum_v0
    act = activity_ts_v0
    feat = ivf_features_v0
    
    _raii0 = dct_scope(plt_multi_boxplot.scatter_kw, dict(alpha=0.8))
    # def aryanas_plot()
    opts = dct_diff_update({
        'groups': [0, 2],
        'group_colors': ['#3a3faf', '#f28574'],
        'group_labels': ['К', 'ЭКО'],
        'figsave': plt_figsave_svg_png_300dpi,
        'outd': Path('../workbooks/misc'),
        'test2': test_mwu,
    }, get_opts(opts))
    opts['outd'].mkdir(exist_ok=True)
    # nps.loc[nps['group']==0, 'mean']
    nps = nps.set_index(['behav', 'group', 'animal'])
    rers = [nps.loc[('RER', g, slice(None)), 'mean'].values for g in opts['groups']]
    O2_ras = [nps.loc[(cO2, g, slice(None)), 'RA'].values for g in opts['groups']]
    O2_iv = [nps.loc[(cO2, g, slice(None)), 'IV'].values for g in opts['groups']]
    
    fig, axs = plt.subplots(3, 3, figsize=(c_A4_w-2, 6), layout='compressed')
    axs = axs.ravel()
    feed = []
    dist_feed = []
    dist = []
    dist_day = []
    dist_night = []
    mass = []
    for g in opts['groups']:
        dof = []
        fs = []
        ms = []
        ds = []
        dds =[]
        dns = []
        for a in opts['data_animals'][g]:
            f = act.loc[cFeed, :, a].Y.values.mean()
            d = act.loc[cDist, :, a].Y.values.mean()
            m = feat.loc[feat.animal_no == a, 'Mass_13.05']
    
            y = act.loc[cDist, g, a].Y.values /100  * 2
            y_p = nonparametric.periodic_avg(y, 48)
            assert len(y_p) == 48
            y_h = nonparametric.binned_avg(y_p, 2)
            assert len(y_h) == 24
            roll = 16
            y_r = np.roll(y_h, roll)
            assert y_r[roll] == y_h[0], np.array(list(zip(range(25), y_h, y_r)))
    
            dns.append((y_r[:3].sum(axis=0) + y_r[17:].sum(axis=0))/10)
            dds.append(y_r[3:17].mean(axis=0))
            # mass = feat.loc[feat.animal_no == a, 'Mass_13.05']
            # yi = y.mean() * 2 * (1000 / mass)
            dof.append(d / f /100)
            fs.append(f  *2)
            ds.append(d / 100 *2)
            ms.append(m)
        dist_feed.append(np.array(dof))
        feed.append(np.array(fs))
        mass.append(np.array(ms))
        dist.append(np.array(ds))
        dist_day.append(np.array(dds))
        dist_night.append(np.array(dns))
    kw = dict(fontstyle='italic')
    ax = axs[0]
    boxplot_main(ax, mass, opts)
    ax.set_ylim(33, 50)
    ax.set_title('a. масса тела, г', **kw)
    ax = axs[1]
    boxplot_main(ax, feed, opts)
    ax.set_title('б. потреб. корма, г/ч', **kw)
    ax.set_ylim(0.20, 0.30)
    ax = axs[2]
    boxplot_main(ax, rers, opts) 
    ax.set_title('в. средний ДК', **kw)
    ax = axs[3]
    boxplot_main(ax, dist_feed, opts) 
    ax.set_title('г. путь/корм, м/г', **kw)
    ax = axs[4]
    boxplot_main(ax, O2_ras, opts)
    ax.set_ylim(0.1, 0.25)
    ax.set_title('д. потреб. О2, ОА', **kw)
    ax = axs[5]
    boxplot_main(ax, O2_iv, opts)
    ax.set_ylim(0.4, 1.5)
    ax.set_title('е. потреб. O2, ВИ', **kw)
    ax = axs[6]
    boxplot_main(ax, dist_night, opts)
    ax.set_ylim(60, 160)
    ax.set_title('ж. пробег ночью, м/ч', **kw)
    ax = axs[7]
    boxplot_main(ax, dist_day, opts)
    ax.set_title('з. пробег днем, м/ч', **kw)
    ax.set_ylim(20, 120)
    ax = axs[-1]
    ax.scatter([], [], c=opts['group_colors'][0], label='контроль')
    ax.scatter([], [], c=opts['group_colors'][1], label='ЭКО')
    ax.legend(edgecolor='none', fontsize=12)
    ax.axis('off')
    for ax in axs[:-4]:
        ax.set_xticks([0, 1], ['', ''])
    plt_delete_ticks(axs[:-4], 'x')
    opts['figsave'](Path('../workbooks/misc/Aryana_thesis_no3.svg'), fig)

    fig, axs = plt.subplots(2, 4, figsize=(c_A4_w, 4), layout='compressed', sharex=True)
    axs = axs.ravel()
    ax = axs[0]
    boxplot_main(ax, mass, opts)
    ax.set_ylim(33, 50)
    ax.set_title('а. масса тела, г', **kw)
    ax = axs[1]
    boxplot_main(ax, feed, opts)
    ax.set_title('б. потреб. корма, г/ч', **kw)
    ax.set_ylim(0.20, 0.30)
    ax = axs[2]
    boxplot_main(ax, rers, opts) 
    ax.set_title('в. средний ДК', **kw)
    ax = axs[3]
    boxplot_main(ax, dist_feed, opts) 
    ax.set_title('г. путь/корм, м/г', **kw)
    ax = axs[4]
    boxplot_main(ax, O2_ras, opts)
    ax.set_ylim(0.1, 0.25)
    ax.set_title('д. потреб. О2, ОА', **kw)
    ax = axs[5]
    boxplot_main(ax, O2_iv, opts)
    ax.set_ylim(0.4, 1.5)
    ax.set_title('е. потреб. O2, ВИ', **kw)
    ax = axs[6]
    boxplot_main(ax, dist_night, opts)
    ax.set_ylim(60, 160)
    ax.set_title('ж. пробег ночью, м/ч', **kw)
    ax = axs[7]
    boxplot_main(ax, dist_day, opts)
    ax.set_title('з. пробег днем, м/ч', **kw)
    ax.set_ylim(20, 120)
    opts['figsave'](opts['outd']/f'{plotname()}_alt1.svg', fig)


# ====================== pls

def ckpt_fn_pls_source_joint_v0(src_dir):
    p = Path(src_dir)
    p = p  / "MMoshkin_sources_wb4.csv" if p.is_dir() else p
    t = pd.read_csv(p, sep="\t", comment="#")
    return t
    
def plot_pls_yscores_bxp_v0(pls_source_joint_v0, opts=None):
    ENV = get_opts(opts)
    sb = report.splitby(pls_source_joint_v0, "Group", ["YSCORE1", "YSCORE2"])
    yscore1 = [sb[i].YSCORE1.values for i in range(3)]
    yscore2 = [sb[i].YSCORE2.values for i in range(3)]
    
    # plt_multi_boxplot(ax, yscore1, colors=ENV['group_colors'], labels=ENV['group_labels'], boxplot_kw=bxp_kw)
    
    def _boxplot_like(ax, gdata, colors, labels):
        bxp_kw = {
         'showcaps': False,
         'showfliers': False,
         'showmeans': True,
         'widths': 0.8,
         'meanprops': {'marker': 's', 'markerfacecolor': 'k', 'markeredgecolor': 'k'},
         'medianprops': {'alpha': 0}
        }
        source_data = gdata
        gdata = []
        for y in source_data:
            m = y.mean()
            sd = y.std()
            se = sd / np.sqrt(len(y))
            gdata.append({
                "med": np.nan,
                "mean": m,
                "q1": m-se,
                "q3": m+se,
                "whislo": m-sd,
                "whishi": m+sd,
            })
        ax.bxp(gdata, positions=np.arange(len(gdata)), manage_ticks=False, **bxp_kw)
        w = 0.8*bxp_kw.get("widths", 0.5)
        for i, y in enumerate(source_data):
            x = i + w*(np_pts_to_scatter_histogram(y, 10)[0] - 0.5)
            ax.scatter(x, y, c=colors[i])
        ax.plot([], [], ls='none', c="k", marker="s", label="Mean")
        ax.plot([], [], ls='none', c="k", marker='s', markerfacecolor='none', label="M ± SE")
        ax.plot([], [], ls='none', c="k", marker="|", markersize=13, label="M ± SD")
        ax.set_xticks(np.arange(len(labels)), labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 4), layout="compressed")
    _boxplot_like(ax, yscore1, ENV['group_colors'], ENV['group_labels']) 
    ax.set_ylim(-4, 4)
    # ax.set_ylabel("YSCORE1")
    ax.legend(loc="lower right", frameon=False, facecolor="#FFFFFF00")
    ENV['figsave'](ENV['outd'] / f"{plotname()}-y1.svg", fig)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 4), layout="compressed")
    _boxplot_like(ax, yscore2, ENV['group_colors'], ENV['group_labels']) 
    # ax.set_ylabel("YSCORE2")
    ax.set_ylim(-4, 4)
    ax.legend(loc="lower right", frameon=False, facecolor="#FFFFFF00")
    ENV['figsave'](ENV['outd'] / f"{plotname()}-y2.svg", fig)

def plot_pls_behav_corr_w_yscores(pls_source_joint_v0, opts=None):
    from matplotlib import patches as mpatches
    ENV = get_opts(opts)
    t = pls_source_joint_v0
    figsize = (c_A4_w - 0.5)/2, 4


    def barplot_like(ax, values, labels, colors, bar_width= 0.8):
        w = bar_width
        hw = w/2
        N = len(values)
        for i in range(N):
            ax.add_patch(mpatches.Rectangle((0, i-hw), values[i], w, color=colors[i]))
        ax.set_ylim(-0.5, i+0.5)
        ax.set_yticks(np.arange(N), labels)
    
    
    y1 = t.YSCORE1
    y2 = t.YSCORE2
    y1r = pd.DataFrame(index=t.columns[:24], columns=['stat', 'pval'])
    y2r = pd.DataFrame(index=t.columns[:24], columns=['stat', 'pval'])
    for name, vals in t.iloc[:, :24].items():
        y1r.loc[name] = stats.pearsonr(y1, vals)
        y2r.loc[name] = stats.pearsonr(y2, vals)
    
    y1r['R2'] = y1r.stat ** 2
    y2r['R2'] = y2r.stat ** 2
        
    
    def _collect_data(yr):
        indices = {
            'Mass_2208': "Масса тела",
            'Dist_dark': 'Активность день',
            'Dist_Light': 'Активность ночь',  
            "Sleep_light": "Сон день",
            "Sleep_dark": "Сон ночь", 
            'RQ_light': "ДК день",
            'RQ_dark': "ДК ночь",
            'O2_light': 'VO2 день',
            'O2_dark': 'VO2 ночь',
            'drink_light': 'Вода день',
            'drink_dark': 'Вода ночь',
        }
        values, labels, colors = [], [], []
        for i, (idx, lbl) in enumerate(reversed(indices.items())):
            values.append(yr.loc[idx, 'R2'])
            labels.append(lbl)
            if lbl.endswith("день"):   colors.append("#cccc7c")
            elif lbl.endswith("ночь"): colors.append("#2c7fb8")
            else:                      colors.append("#99c9be")
        return values, labels, colors

    def __mk_pval_line(ax, N, y_pos):
        line = stat_pearson_p_level(N, 0.025)**2
        ax.axvline(line, c='red', lw=2, alpha=0.5)
        ax.annotate("p=0.05", (line+0.01, y_pos), fontsize=12)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize) 
    values, labels, colors = _collect_data(y1r)
    barplot_like(ax, values, labels, colors)
    __mk_pval_line(ax, N=len(y1), y_pos=1.8)
    ax.set_xticks(np.arange(0, 0.9, 0.1))
    ax.set_xlim(0, 0.8)
    ENV['figsave'](ENV['outd'] / f"{plotname()}-y1.svg", fig)

    fig, ax = plt.subplots(1, 1, figsize=figsize) 
    values, labels, colors = _collect_data(y2r)
    barplot_like(ax, values, labels, colors)
    __mk_pval_line(ax, N=len(y2), y_pos=3.8)
    ax.set_xticks(np.arange(0, 0.9, 0.1))
    ax.set_xlim(0, 0.8)
    ENV['figsave'](ENV['outd'] / f"{plotname()}-y2.svg", fig)


def plotwb_behav_ts_3gr_v6(reset_dir=False):
    # TODO: FIX COSINOR PLOT (units for vo2, fix y axis)
    # TODO: check units everywhere
    fn = get_caller()
    d = fn.__name__.removeprefix("plotwb_")
    outd = Path(f"../workbooks/{d}").resolve()
    if reset_dir: shutil.rmtree(outd)
    outd.mkdir(exist_ok=True, parents=True)
    opts = {
        'group_colors': ['#40c080', '#f0a040', '#a040f0'],
        'group_labels': ['C', '35', '37'],
        'groups': [0, 1, 2],
        'lang': LANG_EN,
        'letters': plt_add_letters(LETTERS_EN),
        'outd': outd,
        'figsave': plt_figsave_svg_png_300dpi,
    }
    opts = dct_diff_update(opts, ENV)
    
    plot_mass_glucose_v0(tbls['ivf_features_v0'], opts=opts)
    plot_behav_averages_v4(tbls['activity_ts_v0'], tbls['ivf_features_v0'], opts=opts)
    ## plot_behav_24h_3gr_v0(opts=opts)
    plot_behav_24h_2gr_3gr_v1(tbls['activity_ts_v0'], opts=opts)
    plot_hippocampus_startle_v0(tbls['ivf_features_v0'],opts=opts)
    plot_cosinor_example_v0(tbls['ivf_ts_cosinor_v0'], tbls['activity_ts_v0'], opts=opts)
    plot_cosinor_params_v3(tbls['ivf_ts_cosinor_v0'], opts=opts)
    plot_behav_day_night_v3(tbls['activity_ts_v0'], opts=opts)
    plot_sleep_agg_dark_light_v1(sleep_agg, sleep_ts, opts=opts)

def plotwb_behav_ts_2gr_v6(reset_dir=False):
    # TODO: FIX COSINOR PLOT (units for vo2, fix y axis)
    # TODO: check units everywhere
    fn = get_caller()
    d = fn.__name__.removeprefix("plotwb_")
    outd = Path(f"../workbooks/{d}").resolve()
    if reset_dir: shutil.rmtree(outd)
    outd.mkdir(exist_ok=True, parents=True)
    opts = {
        'group_colors': ['#a040f0', '#f0a040'],
        'group_labels': ['WT', 'IVF'],
        'groups': [0, 2],
        'lang': LANG_EN,
        'letters': plt_add_letters(LETTERS_EN),
        'outd': outd,
        'figsave': plt_figsave_svg_png_300dpi,
    }
    opts = dct_diff_update(opts, ENV)
    
    plot_mass_glucose_v0(tbls['ivf_features_v0'], opts=opts)
    plot_behav_averages_v4(tbls['activity_ts_v0'], tbls['ivf_features_v0'], opts=opts)
    ## plot_behav_24h_3gr_v0(opts=opts)
    plot_behav_24h_2gr_3gr_v1(tbls['activity_ts_v0'], opts=opts)
    plot_hippocampus_startle_v0(tbls['ivf_features_v0'],opts=opts)
    plot_cosinor_example_v0(tbls['ivf_ts_cosinor_v0'], tbls['activity_ts_v0'], opts=opts)
    plot_cosinor_params_v3(tbls['ivf_ts_cosinor_v0'], opts=opts)
    plot_behav_day_night_v3(tbls['activity_ts_v0'], opts=opts)
    plot_sleep_agg_dark_light_v1(sleep_agg, sleep_ts, opts=opts)

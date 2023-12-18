import numpy as np
import matplotlib.pyplot as plt


glob_colors = plt.get_cmap("tab10").colors
"""global color cycle (by default, a tab10 colormap)"""

def cycle(xs: list):
	"""create a cycled iterator from indexable sequence"""
    n = len(xs)-1
    i = 0
    while True:
        i = 0 if i>=n else (i+1)
        yield xs[i]


_multi_boxplot_style = style = dict(
    showcaps=False, showfliers=False,
    showmeans=True,
    meanprops = dict(
        marker="D", markerfacecolor="k", markeredgecolor="k",),
    medianprops = dict(c="k"),
)

def multi_boxplot(
    ax, 
    group_data: dict['group type', np.ndarray],
    labels = None,
    colors = None,
    style_kw = None
):
	colors = cycle(glob_colors if colors is None else colors)
	style_kw = _multi_boxplot_style if style_kw is None else style_kw
	groups = list(group_data.keys())
    labels = groups if labels is None else labels

    for i, (group, data) in enumerate(group_data.items()):
        x_pseudo = i + np.random.uniform(-0.2, 0.2, len(data))
        ax.boxplot(data, positions=[i, ], widths=0.8, manage_ticks=False, **style_kw)
        ax.scatter(x_pseudo, data, color=next(colors), alpha=0.7)
    ax.set_xticks(np.arange(i), labels)

# this file is basically a jupyter notebook cell with function definitions.
# it should be directly executed


def ckpt_fn_pregnancy_offspring_v0(src_dir):
    p = Path(src_dir)
    p = p / "pregnancy.csv" if p.is_dir() else p
    return pd.read_csv(p, sep="\t")


def ckpt_fn_blastulation_times_areas_v0(src_dir):
    p = Path(src_dir)
    p = p / "Эмбр_деления вся дата(микрометры) (1).csv" if p.is_dir() else p
    t = pd.read_csv(p, sep="\t")
    t.rename(
        inplace=True,
        columns={
            "Group(0-invivo,1-35C,2-37C,3-39C)": "Group",
            "No_embryo(new)": "No_embryo",
            "2 cells N embr_well": "Embryos_p_well",
            "T_2_cells": "C2_time",
            "2cells_size_1bl(area-micrometer2)": "C2_size0",
            "2cells_size_2bl(area-micrometer2)": "C2_size1",
            "2cells_size_1bl": "C2_size0",
            "2cells_size_2bl": "C2_size1",
            # 'Means_2Cells_bl(area-micrometer2)': "C2_mean",
            # 'SD_2Cells_bl(area-micrometer2)': "C2_sd",
            # 'CV_2Cells': "C2_cv",
            "T_4_cells(2 diwision)": "C4_time",
            "4cells_size_1bl(area-micrometer2)": "C4_size0",
            "4cells_size_2bl(area-micrometer2)": "C4_size1",
            "4cells_size_3bl(area-micrometer2)": "C4_size2",
            "4cells_size_4bl(area-micrometer2)": "C4_size3",
            "4cells_size_1bl": "C4_size0",
            "4cells_size_2bl": "C4_size1",
            "4cells_size_3bl": "C4_size2",
            "4cells_size_4bl": "C4_size3",
            # 'Means_4Cell_bl(area-micrometer2)': "C4_mean",
            # 'SD_4Cell_bl(area-micrometer2)': "C4_sd",
            # 'CV_4Cells': "C4_cv",
            "T_8_cells": "C8_time",
            "0-live$ 1-death": "Death",
        },
    )
    t = t[
        [
            "Group",
            "Plate",
            "Well",
            "No_embryo",
            "Embryos_p_well",
            "C2_time",
            "C2_size0",
            "C2_size1",  # "C2_mean", "C2_sd", "C2_cv",
            "C4_time",
            "C4_size0",
            "C4_size1",
            "C4_size2",
            "C4_size3",  # "C4_mean", "C4_sd", "C4_cv",
            "C8_time",
            "Death",
        ]
    ]
    xs = t.Embryos_p_well.values  # take view
    mem = None
    for i in range(len(xs)):
        xi = xs[i]
        if xi == xi:
            mem = xi
        else:
            xs[i] = mem
    del (
        xs,
        mem,
        i,
    )
    t.Embryos_p_well = t.Embryos_p_well.astype(int)
    t.set_index(["Group", "Plate", "Well", "No_embryo"], inplace=True)  # unique index
    t.sort_index(inplace=True)
    return t


def ckpt_fn_pronucleus_fusion_v0(src_dir):
    p = Path(src_dir)
    p = p / "Слияние пронуклеусов (добавлено АЛИЯ) +CV_wb1_t1.csv" if p.is_dir() else p
    t2 = pd.read_csv(p, sep="\t")
    t2.rename(
        inplace=True,
        columns={
            "Group(1-35C,2-37C,3-39C)": "Group",
            "T_1_pronuclei": "C1_time",
            "T_2_cells": "C2_time",
        },
    )
    t2 = t2[["Group", "Plate", "Well", "No_embryo", "C1_time", "C2_time"]]
    t2.set_index(["Group", "Plate", "Well", "No_embryo"], inplace=True)  # unique index
    t2.sort_index(inplace=True)
    return t2


def ckpt_fn_ivf_embryo_methylation_v0():
    t = pd.read_csv("../data_src/5mC-метилирование_М_wb1_t1.csv", sep="\t")
    # filter records where N_cells != 2**n
    nc = t["N_cells"]
    filt = (nc == 2) | (nc == 4) | (nc == 8)  # no nc==1
    t = t[filt].rename(
        columns={
            "Group(0-in_vivo_fert,1-35C,2-37C,3-39C)": "Group",
            "5mC_intensity_nucl": "I_5mC",
            "PI_intensity_nucl": "I_PI",
        }
    )
    t.set_index(["Group", "N_cells", "No_batch", "NoEmb"], inplace=True)
    t.sort_index(inplace=True)
    t = t[["I_5mC", "I_PI"]]
    t["Ratio"] = t.I_5mC / t.I_PI
    t["Log2Ratio"] = np.log2(t.Ratio)
    del nc, filt
    return t


def ckpt_fn_embryo_cell_mass_v0(src_dir):
    p = Path(src_dir)
    p = p / "Дифференциальная окраска (ICM иTE)_wb1_t1.csv" if p.is_dir() else p
    t = pd.read_csv(p, sep="\t")
    t.rename(
        inplace=True,
        columns={
            "Group(0-in-vivo, 1-35C, 2-37C,3-39C)": "Group",
        },
    )
    t = t[["Group", "No", "ICM", "TE", "Total_cells"]]
    assert ((t.ICM + t.TE) == t.Total_cells).all()
    return t


# ========================== plots


def plot_FIG9_cell_mass_distribution_v0(embryo_cell_mass_v0, opts=None):
    "boxplots of ICM / TE data"
    ENV = get_opts(opts)
    t = embryo_cell_mass_v0
    gb = t.groupby(["Group"])
    gdata = [gb.get_group((g,)).ICM.values for g in range(4)]
    gdata2 = [gb.get_group((g,)).TE.values for g in range(4)]

    colors = ENV["group_colors"]
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w - 1, 4), layout="compressed")
    axs = axs.ravel()
    ax = axs[0]
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    ax.set_ylim(0, 55)
    ax.set_xticks(np.arange(4), ENV["group_labels"])
    ax = axs[1]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    ax.set_ylim(0, 55)
    ax.set_xticks(np.arange(4), ENV["group_labels"])
    if ENV.get("extra_letters"):
        ax = axs[1]
        kw = dict(
            fontsize=12, horizontalalignment="center", backgroundcolor="#ffffff50"
        )
        ax.annotate("A", (0.05, 41), **kw)
        ax.annotate("B", (1.05, 33), **kw)
        ax.annotate("A", (2.05, 34), **kw)
        ax.annotate("A, B", (3.05, 36), **kw)
    if ENV.get("do_axlabels"):
        axs[0].set_ylabel("number of inner cellular mass cells", fontsize=12)
        axs[1].set_ylabel("number of trophoectoderm cells", fontsize=12)
        ENV["letters"](axs, (0.02, 0.94))
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG11_implantation_offspring_rates_v0(
    pregnancy_offspring_v0: pd.DataFrame, opts=None
):
    t = pregnancy_offspring_v0
    opts = get_opts(opts)

    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w - 2, 3), layout="compressed")
    axs = axs.ravel()

    def _barplot(ax, t, ok, tot, ylim, ann_height):
        ax.bar(
            t.index,
            t[ok] / t[tot] * 100,
            color=opts.get("group_colors")[1:],
            alpha=0.9,
            width=0.6,
        )
        for i, ann in t.apply(lambda r: f"{r[ok]}/{r[tot]}", axis=1).items():
            ax.annotate(ann, (i, ann_height), horizontalalignment="center")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)

    def _cmpplot(ax, ann, x0, x1, y):
        ax.plot([x0, x1], [y, y], c="k", lw=1.5)
        ax.annotate(
            ann,
            ((x0 + x1) / 2, y),
            horizontalalignment="center",
            weight="bold",
            fontsize=15,
        )

    ax = axs[0]
    _barplot(
        ax, t, ok="PregnancySuccess", tot="PregnancyTotal", ylim=(0, 80), ann_height=10
    )
    ax.set_xticks(t.index, opts.get("group_labels")[1:])
    _cmpplot(ax, "*", -0.2, 2.2, 67)
    ax = axs[1]
    _barplot(
        ax, t, ok="OffspringAlive", tot="OffspringTotal", ylim=(0, 40), ann_height=5
    )
    ax.set_xticks(t.index, opts.get("group_labels")[1:])
    _cmpplot(ax, "***", -0.2, 1.2, 28)
    _cmpplot(ax, "***", 0.8, 2.2, 20)
    if opts.get("do_axlabels"):
        opts["letters"](axs, (0.02, 1.03))
        axs[0].set_title("successful pregnancies, %")
        axs[1].set_title("survived offspring, %")
    opts["figsave"](opts["outd"] / f"{plotname()}.svg", fig)


def plot_FIG1_embryos_survival_v0(blastulation_times_areas_v0: pd.DataFrame, opts=None):
    opts = get_opts(opts)
    gb = blastulation_times_areas_v0["Death"].reset_index().groupby("Group")["Death"]
    total = gb.count()
    t = pd.DataFrame({"Live": total - gb.sum(), "Total": total})
    # display(t)
    t = t.drop(index=0).reset_index()  # don't use group 0 in the plot
    t.Group = t.Group.map({1: 35, 2: 37, 3: 39})

    # display(t)
    def _barplot(ax, t, ok, tot, ylim, ann_height):
        ax.bar(
            t.index,
            t[ok] / t[tot] * 100,
            color=opts.get("group_colors")[1:],
            alpha=0.9,
            width=0.6,
        )
        for i, ann in t.apply(lambda r: f"{r[ok]}/{r[tot]}", axis=1).items():
            ax.annotate(ann, (i, ann_height), horizontalalignment="center")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)

    def _cmpplot(ax, ann, x0, x1, y):
        ax.plot([x0, x1], [y, y], c="k", lw=1.5)
        ax.annotate(
            ann,
            ((x0 + x1) / 2, y),
            horizontalalignment="center",
            weight="bold",
            fontsize=15,
        )

    fig, ax = plt.subplots(1, 1, figsize=((c_A4_w - 2) / 2, 3), layout="compressed")
    _barplot(ax, t, ok="Live", tot="Total", ylim=(0, 110), ann_height=100 / 8)
    ax.set_xticks(t.index, opts.get("group_labels")[1:])
    if opts.get("do_axlabels"):
        ax.set_title("embryos surving\nfirst 3 cleavages, %")
    _cmpplot(ax, "**", -0.2, 1.2, 82)
    _cmpplot(ax, "**", -0.2, 2.2, 95)
    opts["figsave"](opts["outd"] / f"{plotname()}.svg", fig)


def plot_FIG9_cell_mass_distribution_v1(embryo_cell_mass_v0, opts=None):
    "unlike v0, uses %"
    ENV = get_opts(opts)
    t = embryo_cell_mass_v0
    gb = t.groupby(["Group"])
    gdata_mom = [gb.get_group((g,)) for g in range(4)]
    gdata = [100 * (d.ICM / d.Total_cells).values for d in gdata_mom]
    gdata2 = [(d.Total_cells).values for d in gdata_mom]

    for_anova = []
    for group in range(0, 4):
        for_anova.append(
            pd.DataFrame(
                {
                    "Group": group,
                    "ICM_percent": gdata[group],
                    "Total_cells": gdata2[group],
                }
            )
        )
    for_anova = pd.concat(for_anova)
    anova = stats_anova1w(for_anova, c1="Group", dat="ICM_percent")
    lsd = stats_fisher_lsd_1d(anova, for_anova, c1="Group", dat="ICM_percent")
    # ...

    colors = ENV["group_colors"]
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w - 1, 4), layout="compressed")
    axs = axs.ravel()
    ax = axs[0]
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    ax.set_xticks(np.arange(4), ENV["group_labels"])

    ax = axs[1]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    #     ax.set_ylim(0, 40)
    ax.set_xticks(np.arange(4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        axs[0].set_ylabel("inner cellular mass, % of total", fontsize=12)
        ax.set_ylabel("total number of cells", fontsize=12)
        ENV["letters"](axs, (0.02, 0.94))
    cmpplot(axs[0], "*", -0.2, 1.2, 80)  # 0 vs 1 *
    cmpplot(axs[0], "**", 0.9, 2.2, 85)  # 1 vs 2 **

    cmpplot(axs[1], "***", -0.2, 1.2, 65)  # 0 vs 1 ***
    cmpplot(axs[1], "*", -0.2, 2.2, 60)  # 0 vs 2 *

    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG9_cell_mass_distribution_v1b(embryo_cell_mass_v0, opts=None):
    "unlike v0, uses %"
    ENV = get_opts(opts)
    t = embryo_cell_mass_v0
    gb = t.groupby(["Group"])
    gdata_mom = [gb.get_group((g,)) for g in range(4)]
    gdata = [(d.ICM / d.TE).values for d in gdata_mom]
    gdata2 = [(d.Total_cells).values for d in gdata_mom]

    global for_anova
    for_anova = []
    for group in range(0, 4):
        for_anova.append(
            pd.DataFrame(
                {
                    "Group": group,
                    "ICM_TE_ratio": gdata[group],
                    "Total_cells": gdata2[group],
                }
            )
        )
    for_anova = pd.concat(for_anova)

    colors = ENV["group_colors"]
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(1, 2, figsize=(c_A4_w - 1, 4), layout="compressed")
    axs = axs.ravel()
    ax = axs[0]
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    ax.set_xticks(np.arange(4), ENV["group_labels"])
    ax.set_yscale("log", base=2)
    ax.set_yticks(
        [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"]
    )
    ax.set_ylim(1 / 8, 8)

    ax = axs[1]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], **scatter_kw)
    #     ax.set_ylim(0, 40)
    ax.set_xticks(np.arange(4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        axs[0].set_ylabel("inner cellular mass, % of total", fontsize=12)
        ax.set_ylabel("total number of cells", fontsize=12)
        ENV["letters"](axs, (0.02, 0.94))

    cmpplot(axs[0], "*", -0.2, 1.2, 4)  # 0 vs 1 *
    cmpplot(axs[0], "**", 0.9, 2.2, 6)  # 1 vs 2 **

    cmpplot(axs[1], "***", -0.2, 1.2, 65)  # 0 vs 1 ***
    cmpplot(axs[1], "*", -0.2, 2.2, 60)  # 0 vs 2 *
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG2_embryo_division_times_v0(
    blastulation_times_areas_v0, pronucleus_fusion_v0, opts=None
):
    ENV = get_opts(opts)
    t = blastulation_times_areas_v0
    t2 = pronucleus_fusion_v0

    empty = np.array([np.nan])
    no_c = np.zeros(3)
    gdata = []
    gdata2 = []
    for_anova = []
    for_anova2 = []
    for g in range(0, 4):
        tg = t.loc[g]
        T2, T4, T8 = tg.C2_time, tg.C4_time, tg.C8_time
        D4 = (T4 - T2).dropna().values
        D8 = (T8 - T4).dropna().values
        for_anova.extend(
            [
                pd.DataFrame({"g": g, "d": 4, "t": D4}),
                pd.DataFrame({"g": g, "d": 8, "t": D4}),
            ]
        )
        gdata.extend([D4, D8, empty])

        if g == 0:
            continue
        tg2 = t2.loc[g]
        gdata2.extend(
            [
                (tg2.C1_time + tg2.C2_time).values,
            ]
        )
    del for_anova
    # for_anova= pd.concat(for_anova, ignore_index=True)
    # anova = stats_anova2w(for_anova, "d", "g", "t")
    # anova.to_csv(sys.stdout, sep="\t")
    # lsd = stats_fisher_lsd(anova, for_anova, "d", "g", "t")
    # lsd.to_csv(sys.stdout, sep="\t")

    # from plt_multi_boxplot
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["^", "s", "*"] * (len(gdata) // n)

    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(c_A4_w, 4),
        layout="compressed",
        gridspec_kw=dict(width_ratios=(1, 3)),
    )
    axs = axs.ravel()
    ax = axs[1]
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw)
        # mean, sd = np.mean(data), np.std(data)
        # ax.plot([i, i], [mean-sd, mean+sd], color="orange", lw=3)
    ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 60)

    ax.scatter([], [], color=cc, marker="^", label="2nd cleavage")
    ax.scatter([], [], color=cc, marker="s", label="3rd cleavage")
    ax.legend(loc="upper right")

    ax = axs[0]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(x_pseudo, data, color=colors[(i + 1) * 3], **scatter_kw)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(3), ENV["group_labels"][1:])
    if ENV["do_axlabels"]:
        ax.set_ylabel("time, h")
        fig.suptitle("time between 1st (A), 2nd and 3rd (B) cleavages")
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        ax = axs[0]
        kw = ENV.get("extra_letters_kw")
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 0, 38)
        an("B", 1, 32)
        an("C", 2, 28)
        ax = axs[1]
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        cmpplot(ax, "***", -0.2, 1.2, 40)
        an("B", 0, 35)
        an("b", 1, 27)
        cmpplot(ax, "**", 2.8, 4.2, 55)
        an("A", 3, 50)
        an("a", 4, 35)
        cmpplot(ax, "***", 5.8, 7.2, 43)
        an("B", 6, 38)
        an("b", 7, 26)
        cmpplot(ax, "*", 8.8, 10.2, 42)
        an("B", 9, 35)
        an("a", 10, 37)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG3_embryo_dead_alive_div_time_v0(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
    t = blastulation_times_areas_v0[["C4_time", "Death"]]
    empty = np.array([np.nan])
    gdata = []
    for_anova = []
    for group in [1, 2, 3]:
        for death in [0, 1]:
            vals = t.loc[t.Death == death].loc[group, "C4_time"].dropna().values
            gdata.append(vals)
            for_anova.append(pd.DataFrame({"group": group, "death": death, "D4": vals}))
        gdata.append(empty)
    for_anova = pd.concat(for_anova, ignore_index=True)
    # for_anova.to_csv(sys.stdout, sep="\t", index=False)
    anova = stats_anova2w(for_anova, c1="group", c2="death", dat="D4")
    # anova.to_csv(sys.stdout, sep="\t")
    lsd = stats_fisher_lsd(anova, for_anova, c1="group", c2="death", dat="D4")
    # lsd.to_csv(sys.stdout, sep="\t")

    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    cc, c35, c37, c39 = ENV["group_colors"]
    colors = [c35] * 3 + [c37] * 3 + [c39] * 3
    # colors = ENV['group_colors'] * 3

    fig, axs = plt.subplots(1, 1, figsize=(c_A4_w / 2, 4), layout="compressed")
    # axs = axs.ravel()
    ax = axs
    for i, data in enumerate(gdata):
        marker = "o" if i % 3 == 0 else "P"
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=marker, **scatter_kw
        )  # marker=markers[i], **scatter_kw)
    ax.set_ylim(15, 90)
    ax.set_xticks(np_arange(0.5, 3, 3), ["35°C", "37°C", "39°C"])

    ax.scatter([], [], marker="o", label="alive", color="#999999")
    ax.scatter([], [], marker="P", label="dead", color="#999999")
    ax.legend()

    def _cmpplot(ax, ann, x0, x1, y):
        ax.plot([x0, x1], [y, y], c="k", lw=1.5)
        ax.annotate(
            ann,
            ((x0 + x1) / 2, y),
            horizontalalignment="center",
            weight="bold",
            fontsize=15,
        )

    _cmpplot(ax, "***", -0.2, 1.2, 80)
    _cmpplot(ax, "***", 2.8, 4.2, 63)
    _cmpplot(ax, "*", 5.8, 7.2, 40)
    if ENV.get("do_axlabels"):
        ax.set_title("time between 2nd and 3rd cleavages")
        ax.set_ylabel("time, h")
    if ENV.get("extra_letters"):
        ax = axs
        kw = ENV.get("extra_letters_kw")
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 0, 40)
        an("a", 1, 76)
        an("B", 3, 33)
        an("a", 4, 60)
        an("B", 6, 36)
        an("b", 7, 36)
    # ENV['letters'](axs, (0.01, 0.95))
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG4_blastomere_areas_v0(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
    empty = np.array([np.nan])

    t2 = blastulation_times_areas_v0[["C2_size0", "C2_size1"]]
    t4 = blastulation_times_areas_v0[["C4_size0", "C4_size1", "C4_size2", "C4_size3"]]
    m2 = pd.DataFrame({"m": t2.mean(axis=1), "sd": t2.std(axis=1)})
    assert (m2.index == t2.index).all()
    m4 = pd.DataFrame({"m": t4.mean(axis=1), "sd": t4.std(axis=1)})
    assert (m4.index == t4.index).all()
    m2.dropna(axis=0, inplace=True)
    m4.dropna(axis=0, inplace=True)
    m2["cv"] = m2["sd"] / m2["m"]
    m4["cv"] = m4["sd"] / m4["m"]

    gdata = []
    for_anova = []
    for group in range(4):
        gdata.append(m2.loc[group, "m"].values)
        gdata.append(m4.loc[group, "m"].values)
        gdata.append(empty)
        for_anova.append(
            pd.DataFrame({"group": group, "div": 2, "means": m2.loc[group, "m"].values})
        )
        for_anova.append(
            pd.DataFrame({"group": group, "div": 4, "means": m4.loc[group, "m"].values})
        )
    for_anova = pd.concat(for_anova)
    # for_anova.to_csv(sys.stdout, sep="\t", index=False)
    anova = stats_anova2w(for_anova, c1="group", c2="div", dat="means")
    # print("----")
    # anova.to_csv(sys.stdout, sep="\t")
    # print("----")
    lsd = stats_fisher_lsd(anova, for_anova, c1="group", c2="div", dat="means")
    # lsd.to_csv(sys.stdout, sep="\t")
    # print("----")

    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["^", "s", "*"] * (len(gdata) // n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    gdata = [d / 1000 for d in gdata]
    fig, axs = plt.subplots(1, 1, figsize=(c_A4_w * 2 / 3, 4), layout="compressed")
    # axs = axs.ravel()
    ax = axs
    for i, data in enumerate(gdata):
        marker = "^" if i % 3 == 0 else "s"
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=marker, **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=cc, marker="^", label="2nd cleavage")
    ax.scatter([], [], color=cc, marker="s", label="3rd cleavage")
    ax.legend(loc="upper right")

    # ax.set_ylim(0, 90)
    # ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    # empty = np.array([np.nan])
    # gdata = []
    # for_anova = []
    # for group in [1, 2, 3]:
    #     for death in [0, 1]:
    #         vals = t.loc[t.Death==death].loc[group, "C4_time"].dropna().values
    #         gdata.append(vals)
    #         for_anova.append(pd.DataFrame({"group": group, "death": death, "D4": vals}))
    #     gdata.append(empty)
    ax.set_ylim(2, 7.5)
    ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    if ENV.get("do_axlabels"):
        ax.set_ylabel("μm $^2$ / 1000")
        ax.set_title("average blastomere area")
        # ENV['letters'](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        ax = axs
        kw = ENV.get("extra_letters_kw")
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        cmpplot(ax, "***", -0.2, 1.2, 6.5)
        an("A", 0, 6.25)
        an("A", 1, 4)
        cmpplot(ax, "***", 2.8, 4.2, 6)
        an("B", 3, 5.7)
        an("B", 4, 3.7)
        cmpplot(ax, "***", 5.8, 7.2, 6.8)
        an("A", 6, 6.5)
        an("A,C", 7, 3.8)
        cmpplot(ax, "***", 8.8, 10.2, 6.45)
        an("A", 9, 6.15)
        an("C", 10, 3.6)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG5_blastomere_areas_v0(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
    empty = np.array([np.nan])

    t2 = blastulation_times_areas_v0[["C2_size0", "C2_size1"]]
    t4 = blastulation_times_areas_v0[["C4_size0", "C4_size1", "C4_size2", "C4_size3"]]
    m2 = pd.DataFrame({"m": t2.mean(axis=1), "sd": t2.std(axis=1)})
    assert (m2.index == t2.index).all()
    m4 = pd.DataFrame({"m": t4.mean(axis=1), "sd": t4.std(axis=1)})
    assert (m4.index == t4.index).all()
    m2["death"] = blastulation_times_areas_v0["Death"]
    m4["death"] = blastulation_times_areas_v0["Death"]
    m2.dropna(axis=0, inplace=True)
    m4.dropna(axis=0, inplace=True)
    m2["cv"] = m2["sd"] / m2["m"]
    m4["cv"] = m4["sd"] / m4["m"]
    # display(m2)
    gdata2 = []
    gdata4 = []
    for_anova2 = []
    for_anova4 = []
    for group in range(0, 4):
        for death in range(2):
            # display(m2[m2['death']==death])
            if group == 0 and death:
                gdata2.append(empty)
                gdata4.append(empty)
                continue
            v2 = m2[m2["death"] == death].loc[group, "m"].values
            v4 = m4[m4["death"] == death].loc[group, "m"].values
            gdata2.append(v2)
            gdata4.append(v4)
        gdata2.append(empty)
        gdata4.append(empty)
    #         for_anova2.append(pd.DataFrame({"group": group, "death": death, "means": m2.loc[group, 'm'].values}))
    #         for_anova4.append(pd.DataFrame({"group": group, "death": death, "means": m4.loc[group, 'm'].values}))
    # for_anova2 = pd.concat(for_anova2)
    # for_anova4 = pd.concat(for_anova4)
    # anova = stats_anova2w(for_anova, c1="group", c2="div", dat="means")
    # lsd = stats_fisher_lsd(anova, for_anova, c1="group", c2="div", dat="means")
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["o", "P", "*"] * (len(gdata2) // n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(
        2, 1, figsize=(c_A4_w * 2 / 3, 8), sharex=True, layout="compressed"
    )
    axs = axs.ravel()

    gdata2 = [d / 1000 for d in gdata2]
    ax = axs[0]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=cc, marker="o", label="alive")
    ax.scatter([], [], color=cc, marker="P", label="dead")
    ax.legend(loc="upper right")
    ax.set_ylim(4, 7)
    plt_delete_ticks(
        [
            ax,
        ],
        "x",
    )
    gdata4 = [d / 1000 for d in gdata4]
    ax = axs[1]
    for i, data in enumerate(gdata4):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)
    ax.scatter([], [], color=cc, marker="o", label="alive")
    ax.scatter([], [], color=cc, marker="P", label="dead")
    ax.legend(loc="upper right")
    ax.set_yticks(np_arange(1.5, 0.5, 5))

    ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    if ENV.get("do_axlabels"):
        fig.suptitle("blastomere areas and survival, μm$^2$/100")
        ENV["letters"](axs, (0.01, 0.95))

    if ENV.get("extra_letters"):
        kw = ENV.get("extra_letters_kw")
        ax = axs[0]
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 0, 6.25)
        an("B", 3, 5.7)
        cmpplot(ax, "**", 5.8, 7.2, 6.6)
        an("A", 6, 6.45)
        an("A", 9, 5.85)
        ax = axs[1]
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 0, 3.85)
        an("A,B", 3, 3.63)
        an("A,C", 6, 3.73)
        cmpplot(ax, "**", 5.8, 7.2, 3.85)
        an("A", 9, 3.55)
    ax.set_ylim(2, 4)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG6_blastomere_areas_v0(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
    empty = np.array([np.nan])

    t2 = blastulation_times_areas_v0[["C2_size0", "C2_size1"]]
    t4 = blastulation_times_areas_v0[["C4_size0", "C4_size1", "C4_size2", "C4_size3"]]
    m2 = pd.DataFrame({"m": t2.mean(axis=1), "sd": t2.std(axis=1)})
    assert (m2.index == t2.index).all()
    m4 = pd.DataFrame({"m": t4.mean(axis=1), "sd": t4.std(axis=1)})
    assert (m4.index == t4.index).all()
    m2.dropna(axis=0, inplace=True)
    m4.dropna(axis=0, inplace=True)
    m2["cv"] = m2["sd"] / m2["m"]
    m4["cv"] = m4["sd"] / m4["m"]

    gdata = []
    for_anova = []
    for group in range(4):
        gdata.append(m2.loc[group, "cv"].values)
        gdata.append(m4.loc[group, "cv"].values)
        gdata.append(empty)
        for_anova.append(
            pd.DataFrame({"group": group, "div": 2, "means": m2.loc[group, "m"].values})
        )
        for_anova.append(
            pd.DataFrame({"group": group, "div": 4, "means": m4.loc[group, "m"].values})
        )
    for_anova = pd.concat(for_anova)
    anova = stats_anova2w(for_anova, c1="group", c2="div", dat="means")
    lsd = stats_fisher_lsd(anova, for_anova, c1="group", c2="div", dat="means")

    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["^", "s", "*"] * (len(gdata) // n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    gdata = [d * 100 for d in gdata]
    fig, axs = plt.subplots(1, 1, figsize=(c_A4_w * 2 / 3, 4), layout="compressed")
    # axs = axs.ravel()
    ax = axs
    for i, data in enumerate(gdata):
        marker = "^" if i % 3 == 0 else "s"
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=marker, **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=cc, marker="^", label="2nd cleavage")
    ax.scatter([], [], color=cc, marker="s", label="3rd cleavage")
    ax.legend(loc="upper right")

    ax.set_ylim(0, 50)
    ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    if ENV.get("do_axlabels"):
        ax.set_ylabel("CV, %")
        ax.set_title("blastomere area average CV")

    if ENV.get("extra_letters"):
        kw = ENV.get("extra_letters_kw")
        ax = axs
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 0, 27)
        an("A,B", 3, 32)
        cmpplot(ax, "***", 5.8, 7.2, 37)
        an("A,C", 6, 25)
        cmpplot(ax, "***", 8.8, 10.2, 35)
        an("A,C", 9, 22)

    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG7_blastomere_areas_v0(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
    empty = np.array([np.nan])

    t2 = blastulation_times_areas_v0[["C2_size0", "C2_size1"]]
    t4 = blastulation_times_areas_v0[["C4_size0", "C4_size1", "C4_size2", "C4_size3"]]
    m2 = pd.DataFrame({"m": t2.mean(axis=1), "sd": t2.std(axis=1)})
    assert (m2.index == t2.index).all()
    m4 = pd.DataFrame({"m": t4.mean(axis=1), "sd": t4.std(axis=1)})
    assert (m4.index == t4.index).all()
    m2["death"] = blastulation_times_areas_v0["Death"]
    m4["death"] = blastulation_times_areas_v0["Death"]
    m2.dropna(axis=0, inplace=True)
    m4.dropna(axis=0, inplace=True)
    m2["cv"] = m2["sd"] / m2["m"]
    m4["cv"] = m4["sd"] / m4["m"]
    # display(m2)
    gdata2 = []
    gdata4 = []
    for_anova2 = []
    for_anova4 = []
    for group in range(0, 4):
        for death in range(2):
            # display(m2[m2['death']==death])
            if group == 0 and death:
                gdata2.append(empty)
                gdata4.append(empty)
                continue
            v2 = m2[m2["death"] == death].loc[group, "cv"].values
            v4 = m4[m4["death"] == death].loc[group, "cv"].values
            gdata2.append(v2)
            gdata4.append(v4)
        gdata2.append(empty)
        gdata4.append(empty)
    #         for_anova2.append(pd.DataFrame({"group": group, "death": death, "means": m2.loc[group, 'm'].values}))
    #         for_anova4.append(pd.DataFrame({"group": group, "death": death, "means": m4.loc[group, 'm'].values}))
    # for_anova2 = pd.concat(for_anova2)
    # for_anova4 = pd.concat(for_anova4)
    # anova = stats_anova2w(for_anova, c1="group", c2="div", dat="means")
    # lsd = stats_fisher_lsd(anova, for_anova, c1="group", c2="div", dat="means")
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["o", "P", "*"] * (len(gdata2) // n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(
        2, 1, figsize=(c_A4_w * 2 / 3, 8), sharex=True, layout="compressed"
    )
    axs = axs.ravel()

    gdata2 = [d * 100 for d in gdata2]
    ax = axs[0]
    for i, data in enumerate(gdata2):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=cc, marker="o", label="alive")
    ax.scatter([], [], color=cc, marker="P", label="dead")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 40)
    plt_delete_ticks(
        [
            ax,
        ],
        "x",
    )
    gdata4 = [d * 100 for d in gdata4]
    ax = axs[1]
    for i, data in enumerate(gdata4):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=cc, marker="o", label="alive")
    ax.scatter([], [], color=cc, marker="P", label="dead")
    ax.legend(loc="upper right")
    # ax.set_yticks(np_arange(1.5, 0.5, 5))
    ax.set_xticks(np_arange(0.5, 3, 4), ["in-vivo", "35°C", "37°C", "39°C"])
    ax.set_ylim(0, 45)

    if ENV.get("do_axlabels"):
        fig.suptitle("blastomere areas and survival, CV, %")
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        kw = ENV.get("extra_letters_kw")
        ax = axs[0]
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 4, 32)
        an("A,B", 7, 18)
        an("B", 10, 21)
        ax = axs[1]
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("A", 4, 27)
        an("B", 7, 35)
        cmpplot(ax, "***", 5.8, 7.2, 38)
        an("A", 10, 30)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG10_5mC_PI_ratio_v0(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    empty = np.array([np.nan])
    gdata = []
    for group in range(4):
        for ncells in [2, 4, 8]:
            gdata.append(t.loc[group, ncells].Ratio.values)
        gdata.append(empty)

    cc, c35, c37, c39 = ENV["group_colors"]
    n = 4
    colors = ([c35, c37, c39, None]) * n
    markers = ["o", "^", "s", None] * n
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(
        1, 1, figsize=(c_A4_w * 2 / 3, 4), sharex=True, layout="compressed"
    )
    # axs = axs.ravel()

    # ax = axs[0]
    ax = axs
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=c35, marker="o", label="2 cells")
    ax.scatter([], [], color=c37, marker="^", label="4 cells")
    ax.scatter([], [], color=c39, marker="s", label="8 cells")
    ax.legend(loc="upper right")
    ax.axhline(1, c="#000000", alpha=0.5, lw=0.5)
    ax.set_xticks(np_arange(1, 4, 4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        kw = dict(horizontalalignment="center", weight="bold", fontsize=20)
        ax = axs
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("*", 4, 3.4)
        an("*", 6, 3)
        an("*", 8, 2.8)
        an("*", 10.1, 6)

    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG10b_5mC_PI_ratio_CV_v0(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    gb = t.reset_index().groupby(["Group", "N_cells", "No_batch", "NoEmb"])
    t = gb.std() / gb.mean()
    empty = np.array([np.nan])
    gdata = []
    for group in range(4):
        for ncells in [2, 4, 8]:
            gdata.append(t.loc[group, ncells].Ratio.values)
        gdata.append(empty)

    cc, c35, c37, c39 = ENV["group_colors"]
    n = 4
    colors = ([c35, c37, c39, None]) * n
    markers = ["o", "^", "s", None] * n
    # markers = ["o", "P", "*"]*(len(gdata)//n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    gdata = [d * 100 for d in gdata]
    fig, axs = plt.subplots(
        1, 1, figsize=(c_A4_w * 2 / 3, 4), sharex=True, layout="compressed"
    )
    # axs = axs.ravel()

    # ax = axs[0]
    ax = axs
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=colors[0], marker="o", label="2 cells")
    ax.scatter([], [], color=colors[1], marker="^", label="4 cells")
    ax.scatter([], [], color=colors[2], marker="s", label="8 cells")
    ax.legend(loc="upper right")
    ax.axhline(5, c="#000000", alpha=0.5, lw=0.5)
    ax.set_xticks(np_arange(1, 4, 4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        kw = dict(horizontalalignment="center", weight="bold", fontsize=20)
        ax = axs
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), **kw)
        an("*", 4, 76)
        an("*", 5, 62)
        an("*", 8, 43)
        an("*", 9, 45)
        an("*", 10.1, 85)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)

# ===================
# below section is blindly copy-pasted from notebook,
# i didn't make sure that it keeps working

def plot_FIG9_cell_mass_distribution_v2(embryo_cell_mass_v0, opts=None):
    "boxplots of ICM / TE data"
    ENV = get_opts(opts)
    t = embryo_cell_mass_v0
    gb = t.groupby(["Group"])
    
    gdata_mom = [gb.get_group((g,)) for g in range(4)]
    gdata = [d.ICM.values for d in gdata_mom]
    gdata2 = [d.TE.values for d in gdata_mom]
    gdata3 = [(d.ICM / d.TE).values for d in gdata_mom]
    gdata4 = [(d.Total_cells).values for d in gdata_mom]

    global hook # dict for output
    for_anova = []
    for group in range(0, 4):
        for_anova.append(pd.DataFrame({
            "Group": group,
            "ICM": gdata[group],
            "TE": gdata2[group],
            "ICM_TE_ratio": gdata3[group],
            "Total_cells": gdata4[group],
        }))
    for_anova = pd.concat(for_anova)
    hook['fig9_for_anova'] = for_anova
    
    def __bxp(ax, gdata):
        colors = ENV["group_colors"]
        boxplot_kw = plt_multi_boxplot.boxplot_kw
        hw = boxplot_kw.get("widths", 0.5) * 0.45
        for i, data in enumerate(gdata):
            x_pseudo = i + np.random.uniform(-hw, hw, len(data))
            ax.boxplot(data, positions=[i,], manage_ticks=False, **boxplot_kw)
            ax.scatter(x_pseudo, data, color=colors[i], **plt_multi_boxplot.scatter_kw)
        ax.set_xticks(np.arange(4), ENV["group_labels"])
    

    fig, axs = plt.subplots(2, 2, figsize=(c_A4_w - 1, 4*2), layout="compressed")
    axs = axs.ravel()
    ax = axs[0]
    __bxp(ax, gdata)
    ax.set_ylim(0, 55)
    
    ax = axs[1]
    __bxp(ax, gdata2)
    ax.set_ylim(0, 55)

    ax = axs[2]
    __bxp(ax, gdata3)
    ax.set_yscale('log', base=2)
    ax.set_yticks(
        [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"]
    )
    ax.set_ylim(1 / 8, 8)

    ax = axs[3]
    __bxp(ax, gdata4)
    
    if ENV.get("extra_letters"):
        ax = axs[1]
        # kw = dict(
        #     fontsize=12, horizontalalignment="center", backgroundcolor="#ffffff50"
        # )
        # ax.annotate("A", (0.05, 41), **kw)
        # ax.annotate("B", (1.05, 33), **kw)
        # ax.annotate("A", (2.05, 34), **kw)
        # ax.annotate("A, B", (3.05, 36), **kw)
        cmpplot(ax, "***", 0-0.2, 1+0.2, 43)
        cmpplot(ax, "*", 1-0.2, 2+0.2, 47)
        
        ax = axs[2]
        cmpplot(ax, "*", 0-0.2, 1+0.2, 3.9)
        cmpplot(ax, "**", 1-0.2, 2+0.2, 4.9)
        ax = axs[3]
        cmpplot(ax, "***", 0-0.2, 1+0.2, 60)
        cmpplot(ax, "*",  0-0.2, 2+0.2, 65)
        
    if ENV.get("do_axlabels"):
        axs[0].set_ylabel("number of inner cellular mass cells", fontsize=12)
        axs[1].set_ylabel("number of trophoectoderm cells", fontsize=12)
        axs[2].set_ylabel("inner cellular mass / trophoectoderm ratio", fontsize=12)
        axs[3].set_ylabel("total number of cells", fontsize=12)
        ENV["letters"](axs, (0.02, 0.94))
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)

def plot_FIG10_5mC_PI_ratio_v1(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    empty = np.array([np.nan])
    global hook # dict for extra output
    for_anova = []
    gdata = []
    for group in range(4):
        for ncells in [2, 4, 8]:
            d = t.loc[group, ncells].Ratio.values
            # d = t.loc[group, ncells].Log2Ratio.values
            gdata.append(d)
            for_anova.append(pd.DataFrame({"Group": group, "Ncells": ncells, "ratio": d}))
        gdata.append(empty)
    for_anova = pd.concat(for_anova)
    hook['fig10_for_anova'] = for_anova
    
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 4
    colors = ([c35, c37, c39, None]) * n
    markers = ["o", "^", "s", None] * n
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    fig, axs = plt.subplots(
        1, 1, figsize=(c_A4_w -1, 4), sharex=True, layout="compressed"
    )
    # axs = axs.ravel()

    # ax = axs[0]
    ax = axs
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(
            data,
            positions=[
                i,
            ],
            manage_ticks=False,
            **boxplot_kw,
        )
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=c35, marker="o", label="2 cells")
    ax.scatter([], [], color=c37, marker="^", label="4 cells")
    ax.scatter([], [], color=c39, marker="s", label="8 cells")
    ax.legend(loc="upper right")
    ax.axhline(1, c="#000000", alpha=0.5, lw=0.5)
    ax.set_xticks(np_arange(1, 4, 4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        ax = axs
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), horizontalalignment="center", weight="bold", fontsize=20)
        an2 = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), horizontalalignment="center", weight="bold", fontsize=12)
        cmpplot(ax, "***", 7.8, 10.2, 5.7) #         2  8   2   2   0.0000  ***
        cmpplot(ax, "***", 8.8, 10.2, 5) #         2    8   2   4   0.0000  ***
        an2("a", 0, 2.3), an2("a", 4, 3.5) #   1    2   0   2   0.0239  *
        an2("bb,c", 2, 2.3), an2("bb", 6, 3.5) # 1  8   0   8   0.0074  **
        #an2("c", 2, 3), 
        an2("c", 14, 3.5) #    3    8   0   8   0.0218  *
 

    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG10b_5mC_PI_ratio_CV_v1(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    gb = t.reset_index().groupby(["Group", "N_cells", "No_batch", "NoEmb"])
    t = gb.std() / gb.mean()
    empty = np.array([np.nan])
    gdata = []
    global hook # dict for extra output
    for_anova = []
    for group in range(4):
        for ncells in [2, 4, 8]:
            d = t.loc[group, ncells].Ratio.values
            gdata.append(d)
            for_anova.append(pd.DataFrame({"Group": group, "Ncells": ncells, "cv": d}))
        gdata.append(empty)
    for_anova = pd.concat(for_anova)
    hook['fig10b_for_anova'] = for_anova
    # mi, ma = for_anova.cv.quantile(0.03), for_anova.cv.quantile(0.96)
    # spa = ma-mi
    # mi, ma = mi-10*spa, ma+10*spa
    
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 4
    colors = ([c35, c37, c39, None]) * n
    markers = ["o", "^", "s", None] * n
    # markers = ["o", "P", "*"]*(len(gdata)//n)
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    gdata = [d * 100 for d in gdata]
    fig, axs = plt.subplots(
        1, 1, figsize=(c_A4_w - 1, 4), sharex=True, layout="compressed"
    )
    # axs = axs.ravel()

    # ax = axs[0]
    ax = axs
    for i, data in enumerate(gdata):
        x_pseudo = i + np.random.uniform(-hw, hw, len(data))
        ax.boxplot(data,positions=[i,],manage_ticks=False,**boxplot_kw,)
        ax.scatter(
            x_pseudo, data, color=colors[i], marker=markers[i], **scatter_kw
        )  # marker=markers[i], **scatter_kw)

    ax.scatter([], [], color=colors[0], marker="o", label="2 cells")
    ax.scatter([], [], color=colors[1], marker="^", label="4 cells")
    ax.scatter([], [], color=colors[2], marker="s", label="8 cells")
    ax.legend(loc="upper right")
    ax.axhline(5, c="#000000", alpha=0.5, lw=0.5)
    # ax.set_ylim(mi, ma)
    ax.set_xticks(np_arange(1, 4, 4), ENV["group_labels"])
    if ENV.get("do_axlabels"):
        ENV["letters"](axs, (0.01, 0.95))
    if ENV.get("extra_letters"):
        kw = dict(horizontalalignment="center", weight="bold", fontsize=20)
        ax = axs
        an = lambda annot, x, y: ax.annotate(annot, (x + 0.05, y), horizontalalignment="center", weight="bold", fontsize=12)
        
        
        ## g1   g2  g1  g2  pvalue  stars
        cmpplot(ax, "*", -0.2, 2.2, 45) ## 0    8   0   2   0.0443  *
        an("a", 0, 55); an("a", 4, 80) ## 1 2   0   2   0.0148  *
        an("b,c", 1, 55); #an("b", 5, 80); ## 1 4   0   4   0.047   *
        an("c,e", 9, 60) ## 2   4   0   4   0.0302  *
        cmpplot(ax, "**", 7.8, 9.2, 50)## 2 4   2   2   0.0055  **
        ## 2    8   0   8   0.0001  ***
        ## 2    8   1   8   0.0094  **
        ## 3    8   2   8   0.0002  ***
        cmpplot(ax, "***", 8.8, 10.2, 90) ## 2  8   2   4   0.0004  ***
        an("b,d", 5, 75); #an("d", 13, 45)## 3  4   1   4   0.027   *
        an("d,e", 13, 50) ## 3  4   2   4   0.015   *
        

    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plotwb_embryos_additional_v0(reset_dir=False):
    fn = get_caller()
    d = fn.__name__.removeprefix("plotwb_")
    outd = Path(f"../workbooks/{d}").resolve()
    if reset_dir and outd.exists(): shutil.rmtree(outd)
    outd.mkdir(exist_ok=True, parents=True)
    desc_p = Path(f"../text/plotwb_descr-{d}.md")

    opts = {
        "group_colors": ["#999999", "#4f81bd", "#c0504d", "#9bbb59"],
        "letters": plt_add_letters(LETTERS_EN),
        "lang": LANG_EN,
        'outd': outd,
        'figsave': plt_figsave_svg_png_600dpi,
    }
    opts = dct_diff_update(opts, ENV)

    embryo_cell_mass_v0 = ckpt_fn_embryo_cell_mass_v0("../data_src")
    plot_FIG9_cell_mass_distribution_v1(embryo_cell_mass_v0, opts=opts)
    plot_FIG9_cell_mass_distribution_v1b(embryo_cell_mass_v0, opts=opts)
    
    # !pandoc -i $desc_p -o $outd/description.docx --resource-path=$outd


def plotwb_embryos_additional_v2(reset_dir=False):
    """
    Даниил, здраствуй. Просьба, добавь к рис 9 , общее число клеток и отношение icm/te, 
    а к рис 10 значения влияния группы, число клеток и взаимодействие (Anova) 
    где есть различия между группами поставь буквы, а по числу клеток звезды.
    """
    fn = get_caller()
    d = fn.__name__.removeprefix("plotwb_")
    outd = Path(f"../workbooks/{d}").resolve()
    # if reset_dir and outd.exists(): shutil.rmtree(outd)
    outd.mkdir(exist_ok=True, parents=True)
    desc_p = Path(f"../text/plotwb_descr-{d}.md")

    opts = {
        "group_colors": ["#999999", "#4f81bd", "#c0504d", "#9bbb59"],
        "letters": plt_add_letters(LETTERS_EN),
        "lang": LANG_EN,
        'outd': outd,
        'figsave': plt_figsave_svg_png_600dpi,
        'do_axlabels': False,
    }
    opts = dct_diff_update(opts, ENV)

    embryo_cell_mass_v0 = ckpt_fn_embryo_cell_mass_v0("../data_src")
    ivf_embryo_methylation_v0 = ckpt_fn_ivf_embryo_methylation_v0()
    # plot_FIG9_cell_mass_distribution_v1(embryo_cell_mass_v0, opts=opts)
    # plot_FIG9_cell_mass_distribution_v1b(embryo_cell_mass_v0, opts=opts)
    plot_FIG9_cell_mass_distribution_v2(embryo_cell_mass_v0, opts=opts)
    plot_FIG10_5mC_PI_ratio_v1(ivf_embryo_methylation_v0, opts=opts)
    plot_FIG10b_5mC_PI_ratio_CV_v1(ivf_embryo_methylation_v0, opts=opts)
    
    # !pandoc -i $desc_p -o $outd/description.docx --resource-path=$outd


def plotwb_embryos_v0(reset_dir=False):
    fn = get_caller()
    d = fn.__name__.removeprefix("plotwb_")
    outd = Path(f"../workbooks/{d}").resolve()
    if reset_dir and outd.exists(): shutil.rmtree(outd)
    outd.mkdir(exist_ok=True, parents=True)
    desc_p = Path(f"../text/plotwb_descr-{d}.md")

    opts = {
        "group_colors": ["#999999", "#4f81bd", "#c0504d", "#9bbb59"],
        "letters": plt_add_letters(LETTERS_EN),
        "lang": LANG_EN,
        'outd': outd,
        'figsave': plt_figsave_svg_png_600dpi,
        'extra_letters': True,
        'do_axlabels': False,
    }
    opts = dct_diff_update(opts, ENV)
    
    src_dir="../data_src"
    pregnancy_offspring_v0 = ckpt_fn_pregnancy_offspring_v0(src_dir)
    blastulation_times_areas_v0 = ckpt_fn_blastulation_times_areas_v0(f"{src_dir}/Эмбр_деления вся дата(фин).csv")
    pronucleus_fusion_v0 = ckpt_fn_pronucleus_fusion_v0(f"{src_dir}/Слияние пронуклеусов (добавлено АЛИЯ) (1).csv")
    embryo_cell_mass_v0 = ckpt_fn_embryo_cell_mass_v0(src_dir)
    ivf_embryo_methylation_v0 = ckpt_fn_ivf_embryo_methylation_v0()
    
    plot_FIG1_embryos_survival_v0(blastulation_times_areas_v0, opts)
    plot_FIG2_embryo_division_times_v0(blastulation_times_areas_v0, pronucleus_fusion_v0, opts)
    plot_FIG3_embryo_dead_alive_div_time_v0(blastulation_times_areas_v0, opts)  
    plot_FIG4_blastomere_areas_v0(blastulation_times_areas_v0, opts)
    plot_FIG5_blastomere_areas_v0(blastulation_times_areas_v0, opts)
    plot_FIG6_blastomere_areas_v0(blastulation_times_areas_v0, opts)
    plot_FIG7_blastomere_areas_v0(blastulation_times_areas_v0, opts)
    plot_FIG9_cell_mass_distribution_v0(embryo_cell_mass_v0, opts)
    # plot_FIG9_cell_mass_distribution_v1(embryo_cell_mass_v0, opts)
    plot_FIG10_5mC_PI_ratio_v0(ivf_embryo_methylation_v0, opts)
    plot_FIG10b_5mC_PI_ratio_CV_v0(ivf_embryo_methylation_v0, opts)
    plot_FIG11_implantation_offspring_rates_v0(pregnancy_offspring_v0, opts)

    # !pandoc -i $desc_p -o $outd/description.docx --resource-path=$outd

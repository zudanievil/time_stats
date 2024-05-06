def plot_FIG1_embryos_survival_v1(blastulation_times_areas_v0: pd.DataFrame, opts=None):
    ENV = get_opts(opts)
    gb = blastulation_times_areas_v0["Death"].reset_index().groupby("Group")["Death"]
    total = gb.count()
    t = pd.DataFrame({"Live": total - gb.sum(), "Total": total})
    t.reset_index(inplace=True)
    t.Group = t.Group.map({1: 35, 2: 37, 3: 39})
    

    fig, ax = plt.subplots(1, 1, figsize=(ENV['axw']*1.5, ENV['axh']+0.2), 
                           layout="compressed")
    ax.bar(t.index, t.Live/t.Total*100, color=ENV['group_colors'])
    # for i, ann in t.apply(lambda row: f"{row.Live}/{row.Total}", axis=1).items():
    #     ax.annotate(ann, (i, 100/8), horizontalalignment="center")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 110)

    ax.set_xticks(t.index, ENV["group_labels"])
    if ENV.get("do_axlabels"):
        ax.set_ylabel("embryos surving first 3 cleavages, %", fontsize=12)
        cmpplot2(ax, "**", 1, 2, 82)
        cmpplot2(ax, "**", 1, 3, 95)
    else:
        ax.set_ylabel(" ", fontsize=12)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)

def plot_FIG11_implantation_offspring_rates_v1(
    pregnancy_offspring_v0: pd.DataFrame, opts=None
):
    t = pregnancy_offspring_v0
    ENV = get_opts(opts)

    do_labels = ENV.get("do_axlabels", True)
    letters = ENV["letters"]
    if do_labels:
        fig, axs = plt.subplots(1, 2, figsize=(ENV['axw']*2, ENV['axh']), 
                                gridspec_kw=ENV['gridkw'], layout="compressed")
        axs = axs.ravel()
    else:
        fig0, ax0 = plt.subplots(1, 1, figsize=(ENV['axw']*1, ENV['axh']), layout="compressed")
        fig1, ax1 = plt.subplots(1, 1, figsize=(ENV['axw']*1, ENV['axh']), layout="compressed")
        axs = np.r_[ax0, ax1]

    def _barplot(ax, t, ok, tot, ylim, ann_height = None):
        ax.bar(t.index, t[ok] / t[tot] * 100,
            color=ENV["group_colors"][1:], alpha=0.9, width=0.6,)
        ax.set_xticks(t.index, ENV["group_labels"][1:])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)
        if ann_height is None: return 
        for i, ann in t.apply(lambda r: f"{r[ok]}/{r[tot]}", axis=1).items():
            ax.annotate(ann, (i, ann_height), horizontalalignment="center")

    ax = axs[0]
    _barplot(ax, t, ok="PregnancySuccess", tot="PregnancyTotal", ylim=(0, 80),)
    ax = axs[1]
    _barplot(ax, t, ok="OffspringAlive", tot="OffspringTotal", ylim=(0, 40),)
    if do_labels:
        letters(axs, ENV['letpos'])
        axs[0].set_ylabel("successful pregnancies, %", fontsize=12)
        axs[1].set_ylabel("survived offspring, %", fontsize=12)
        ax = axs[0]
        cmpplot(ax, "*", 0, 2, 67)
        ax = axs[1]
        cmpplot2(ax, "***", -0, 1, 28)
        cmpplot2(ax, "***", 1, 2, 20)
        ENV['figsave'](ENV["outd"] / f"{plotname()}.svg", fig)
    else:
        for i in [0, 1]: 
            axs[i].set_ylabel(" ", fontsize=12)
            ENV['figsave'](ENV['outd'] / f"{plotname()}-{letters.letters[i]}.svg", [fig0, fig1][i])


def plot_FIG2_embryo_division_times_v1(
    blastulation_times_areas_v0, pronucleus_fusion_v0, opts=None
):
    ENV = get_opts(opts)
    t = blastulation_times_areas_v0
    t2 = pronucleus_fusion_v0

    for_anova = []
    for g in range(1, 4):
        tg = t2.loc[g]
        for_anova.append(pd.DataFrame({"g": g, "d": 2, "t": (tg.C1_time + tg.C2_time).values}))
    for g in range(0, 4):
        tg = t.loc[g]
        T2, T4, T8 = tg.C2_time, tg.C4_time, tg.C8_time
        for_anova.extend([
            pd.DataFrame({"g": g, "d": 4, "t": (T4 - T2).dropna().values}),
            pd.DataFrame({"g": g, "d": 8, "t": (T8 - T4).dropna().values}),
        ])
        
    for_anova = pd.concat(for_anova, ignore_index=True)
    hook['FIG2_for_anova'] = for_anova
    
    cc, c35, c37, c39 = ENV["group_colors"]
    n = 3
    # colors = [cc] * n + [c35] * n + [c37] * n + [c39] * n
    markers = ["o", "^", "s", "*"]

    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    if ENV['do_axlabels']:
        fig, axs = plt.subplots(1, 3, figsize=(ENV['axw']*3, ENV['axh']*1.1), 
            layout="compressed",gridspec_kw=dict(wspace=-0.05, width_ratios=[3, 4, 4]), sharey=True)
        axs = axs.ravel()
    else:
        figs = []; axs =[]
        for i in range(3):
            w = ENV['axw'] *3 / (3+4+4) * (4 if i else 3)
            fig, ax = plt.subplots(1, 1, figsize=(w, ENV['axh']*1.1),
                layout="compressed")
            figs.append(fig); axs.append(ax)

    fa = for_anova.set_index(["g", "d"]).sort_index()
    for j, d in enumerate([2, 4, 8]):
        ax = axs[j]
        for i, g in enumerate([0, 1, 2, 3]):
            if g==0 and d==2:
                continue
            data = fa.loc[g,d].values
            xs = i + (np_pts_to_scatter_histogram(data.ravel(), bins=20)[0] - 0.5)*2*hw
            # xs = i + np.random.uniform(-hw, hw, len(data))
            ax.plot(xs, data,ls="none", color=ENV['group_colors'][i], marker=markers[i], markersize=5, **scatter_kw)
            ax.boxplot(data, positions=[ i,], manage_ticks=False, **boxplot_kw,)
        if d==2:
            ax.set_xticks(np.arange(1, 4), ENV['group_labels'][1:])
            ax.set_xlim(0.5, 3.5)
        else:
            ax.set_xticks(np.arange(len(ENV['group_labels'])), ENV['group_labels'])
    ax.set_ylim(0, 60)
    
    if ENV['do_axlabels']:
        for i in range(3): axs[i].set_title(f"mitosis {i+1}", fontsize=12)
        axs[0].set_ylabel("mitosis time, h", fontsize=12)
        ENV['letters'](axs, ENV['letpos'])
        c = cmpplot2
        ax = axs[0]
        c(ax, "***", 1, 2, 42)
        c(ax, "***", 2, 3, 35)
        ax = axs[1]
        c(ax, "***", 0, 1, 42)
        c(ax, "***", 1, 3, 54)
        c(ax, "***", 1, 2, 47)
        ax = axs[2]
        c(ax, "***", 0, 1, 35)
        c(ax, "**" , 1, 2, 41)
        c(ax, "*"  , 0, 3, 50)
    else:
        for i in range(3): axs[i].set_title(" ", fontsize=12)
        axs[0].set_ylabel(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)
    else:
        for fig, l in zip(figs, "ABCD"):
            ENV['figsave'](ENV['outd']/f"{plotname()}-{l}.svg", fig)

def plot_FIG3_embryo_dead_alive_div_time_v1(blastulation_times_areas_v0, opts=None):
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
    width = boxplot_kw.get("widths", 0.5) * 0.9

    cc, c35, c37, c39 = ENV["group_colors"]
    colors = [c35] * 3 + [c37] * 3 + [c39] * 3
    # colors = ENV['group_colors'] * 3

    fig, axs = plt.subplots(1, 1, figsize=(ENV['axw']*2, ENV['axh']), layout="compressed")
    # axs = axs.ravel()
    ax = axs
    for i, data in enumerate(gdata):
        marker = "o" if i % 3 == 0 else "P"
        if data is empty:
            continue
        xs = i + ptshist(data, width, bins=20)
        # xs = i + np.random.uniform(-width/2, width/2, len(data))
        ax.plot(xs, data, ls="none", color=colors[i], marker=marker, markersize=5, **scatter_kw)  # marker=markers[i], **scatter_kw)
        ax.boxplot(data,positions=[i], manage_ticks=False, **boxplot_kw)
    ax.set_ylim(15, 90)
    ax.set_xticks(np_arange(0.5, 3, 3), ["35°C", "37°C", "39°C"])

    ax.scatter([], [], marker="o", label="alive", color="#999999")
    ax.scatter([], [], marker="P", label="dead", color="#999999")
    ax.legend()
    if ENV["do_axlabels"]:
        ax.set_ylabel("time between division 2 and 3, h", fontsize=12)
        c = cmpplot2
        c(ax, "***", 0, 1, 80)
        c(ax, "***", 3, 4, 63)
        c(ax, "*"  , 6, 7, 40)
    else:
        ax.set_ylabel(" ", fontsize=12)
    ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)


def plot_FIG4_blastomere_areas_v1(blastulation_times_areas_v0, opts=None):
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

    for_anova = []
    for group in range(4):
        for_anova.extend((
            pd.DataFrame({"group": group, "div": 2, "means": m2.loc[group, "m"].values / 1000}),
            pd.DataFrame({"group": group, "div": 4, "means": m4.loc[group, "m"].values / 1000}),
        ))
    for_anova = pd.concat(for_anova)
    hook["FIG4_for_anova"] = for_anova

    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    markers = ["o", "^", "s", "*"]

    if ENV['do_axlabels']:
        fig, axs = plt.subplots(1, 2, figsize=(ENV['axw']*2.5, ENV['axh']), gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
    else:
        fig1, axs1 = plt.subplots(1, 1, figsize=(ENV['axw']*1.25, ENV['axh']), gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        fig0, axs0 = plt.subplots(1, 1, figsize=(ENV['axw']*1.25, ENV['axh']), gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        figs = [fig0, fig1]; axs = [axs0, axs1];
    for_anova = for_anova.set_index(["group", "div"])
    # display(for_anova.index.unique())
    for i, div in enumerate([2, 4]):
        # marker = "^" if i % 3 == 0 else "s"
        ax = axs[i]
        for j, g in enumerate([0, 1, 2, 3]):
            # marker="^"
            data = for_anova.loc[g, div].values
            xs = j + ptshist(data, hw*2, bins=20)
            ax.plot(xs, data, ls="none", color=ENV['group_colors'][j], 
                marker=markers[j], markersize=5, **scatter_kw)
            ax.boxplot(data, positions=[j], manage_ticks=False, **boxplot_kw)
    
    axs[0].set_ylim(4, 7)
    axs[1].set_ylim(2, 4.5)
    ax.set_xticks([0, 1, 2, 3], ENV['group_labels'] )
    if ENV["do_axlabels"]:
        axs[0].set_ylabel(r"average blastomere area $μm^2 \cdot 10^3$  ", fontsize=12)
        ENV['letters'](axs, ENV['letpos'])
        axs[0].set_title("2 cells", fontsize=12)
        axs[1].set_title("4 cells", fontsize=12)
        c = cmpplot2
        ax = axs[0]
        c(ax, "***", 0, 1, 5.7)
        c(ax, "***", 1, 2, 6.0)
        c(ax, "***", 1, 3, 6.6)
        ax = axs[1]
        c(ax, "**" , 0, 1, 4.0)
        c(ax, "*"  , 1, 2, 4.2) 
    else:
        axs[0].set_ylabel(" ", fontsize=12)
        axs[0].set_title(" ", fontsize=12)
        axs[1].set_title(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)
    else:
        for fig, l in zip(figs, "ABCD"):
            ENV["figsave"](ENV["outd"] / f"{plotname()}-{l}.svg", fig)


def plot_FIG5_blastomere_areas_v1(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
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
    for_anova = []
    for group in range(0, 4):
        for death in range(2):
            # display(m2[m2['death']==death])
            if group == 0 and death:
                continue
            v2 = m2[m2["death"] == death].loc[group, "m"].values/1000
            v4 = m4[m4["death"] == death].loc[group, "m"].values/1000
            for_anova.extend((
                pd.DataFrame({"div": 1, "death": death, "group": group, "area": v2}),
                pd.DataFrame({"div": 2, "death": death, "group": group, "area": v4}),
            ))
    for_anova = pd.concat(for_anova)
    hook['FIG5_for_anova'] = for_anova
    datas = for_anova.set_index(["div", "death", "group"]).sort_index()
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    if ENV['do_axlabels']:
        fig, axs = plt.subplots(2, 1, figsize=(ENV['axw']*2.5, ENV['axh']*2), sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        fig0, axs0 = plt.subplots(1, 1, figsize=(ENV['axw']*2.5, ENV['axh']), layout="compressed")
        fig1, axs1 = plt.subplots(1, 1, figsize=(ENV['axw']*2.5, ENV['axh']), layout="compressed")
        axs = np.array([axs0, axs1])

    for div in [1, 2]:
        i = -2
        ax = axs[div-1]
        for group in range(4):
            i+=1
            for death in range(2):
                i+=1    
                if group==0 and death==1:
                    i -= 1
                    continue
                data = datas.loc[div, death, group].values
                xs = i + ptshist(data, hw*2)
                ax.plot(xs, data, color=ENV["group_colors"][group], 
                    ls="none", marker="P" if death else "o", **scatter_kw)
                ax.boxplot(data, positions=[i], manage_ticks=False, **boxplot_kw)
        cc = '#999999'
        ax.scatter([], [], color=cc, marker="o", label="alive")
        ax.scatter([], [], color=cc, marker="P", label="dead")
        ax.legend(loc="upper right", facecolor="none")
        ax.set_xticks([0, 2.5, 5.5, 8.5], ENV['group_labels'])
    axs[0].set_ylim(4.1, 7)
    axs[1].set_ylim(2.4, 4)
    if ENV['do_axlabels']:
        for ax in axs: ax.set_ylabel("blastomere area, μm$^2$/1000", fontsize=12)
        ENV['letters'](axs, (0.01, 0.92))
        axs[0].set_title("2 blastomeres", y=0.9, fontsize=12)
        axs[1].set_title("4 blastomeres", y=0.9, fontsize=12)
        c = cmpplot2
        c(axs[1], '*', 2, 3, 3.6)
        c(axs[1], '**', 5,6, 3.72)
        c(axs[0], '*', 5,6, 6.45)
    else:
        for ax in axs: ax.set_ylabel(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)
    else:
        ENV['figsave'](ENV['outd']/f"{plotname()}-A.svg", fig0)
        ENV['figsave'](ENV['outd']/f"{plotname()}-B.svg", fig1)


def plot_FIG6_blastomere_areas_v1(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
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
    for_anova = []
    for group in range(0, 4):
        v2 = m2.loc[group, "cv"].values *100
        v4 = m4.loc[group, "cv"].values *100
        for_anova.extend((
            pd.DataFrame({"div": 1, "group": group, "areacv": v2}),
            pd.DataFrame({"div": 2, "group": group, "areacv": v4}),
        ))
    for_anova = pd.concat(for_anova)
    hook['FIG6_for_anova'] = for_anova
    datas = for_anova.set_index(["div", "group"]).sort_index()
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    if ENV['do_axlabels']:
        fig, axs = plt.subplots(1, 2, figsize=(ENV['axw']*2.5, ENV['axh']), 
            gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        fig0, axs0 = plt.subplots(1, 1, figsize=(ENV['axw']*1.25, ENV['axh']), sharex=True, layout="compressed")
        fig1, axs1 = plt.subplots(1, 1, figsize=(ENV['axw']*1.25, ENV['axh']), sharex=True, layout="compressed")
        axs = np.array([axs0, axs1])
    for div in [1, 2]:
        i = -1
        ax = axs[div-1]
        for group in range(4):
            i+=1
            data = datas.loc[div, group].values
            xs = i + ptshist(data, hw*2)
            f = plt_multi_boxplot
            ax.plot(xs, data, color=ENV["group_colors"][group], 
                ls="none", marker=["o", "^", "s", "*", ][group], **scatter_kw)
            ax.boxplot(data, positions=[i], manage_ticks=False, **boxplot_kw)
        ax.set_xticks(np.arange(4), ENV['group_labels'])
        ax.set_ylim(0, 35)
    if ENV['do_axlabels']:
        for ax in axs: ax.set_ylabel("blastomere area CV, %", fontsize=12)
        ENV['letters'](axs, (0.01, 0.92))
        axs[0].set_title("2 blastomeres", fontsize=12)
        axs[1].set_title("4 blastomeres", fontsize=12)
    else:
        for ax in axs: 
            ax.set_ylabel(" ", fontsize=12)
            ax.set_title(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)
    else:
        ENV['figsave'](ENV['outd']/f"{plotname()}-A.svg", fig0)
        ENV['figsave'](ENV['outd']/f"{plotname()}-B.svg", fig1)


def plot_FIG7_blastomere_areas_v1(blastulation_times_areas_v0, opts=None):
    ENV = get_opts(opts)
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
    for_anova = []
    for group in range(0, 4):
        for death in range(2):
            # display(m2[m2['death']==death])
            if group == 0 and death:
                continue
            v2 = m2[m2["death"] == death].loc[group, "cv"].values*100
            v4 = m4[m4["death"] == death].loc[group, "cv"].values*100
            for_anova.extend((
                pd.DataFrame({"div": 1, "death": death, "group": group, "areacv": v2}),
                pd.DataFrame({"div": 2, "death": death, "group": group, "areacv": v4}),
            ))
    for_anova = pd.concat(for_anova)
    hook['FIG7_for_anova'] = for_anova
    datas = for_anova.set_index(["div", "death", "group"]).sort_index()
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45
    if ENV['do_axlabels']:
        fig, axs = plt.subplots(2, 1, figsize=(ENV['axw']*2.5, ENV['axh']*2), sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        fig0, axs0 = plt.subplots(1, 1, figsize=(ENV['axw']*2.5, ENV['axh']), layout="compressed")
        fig1, axs1 = plt.subplots(1, 1, figsize=(ENV['axw']*2.5, ENV['axh']), layout="compressed")
        axs = np.array([axs0, axs1])

    for div in [1, 2]:
        i = -2
        ax = axs[div-1]
        for group in range(4):
            i+=1
            for death in range(2):
                i+=1    
                if group==0 and death==1:
                    i -= 1
                    continue
                data = datas.loc[div, death, group].values
                xs = i + ptshist(data, hw*2)
                ax.plot(xs, data, color=ENV["group_colors"][group], 
                    ls="none", marker="P" if death else "o", **scatter_kw)
                ax.boxplot(data, positions=[i], manage_ticks=False, **boxplot_kw)
        cc = '#999999'
        ax.scatter([], [], color=cc, marker="o", label="alive")
        ax.scatter([], [], color=cc, marker="P", label="dead")
        ax.legend(loc="upper right", facecolor="none")
        ax.set_xticks([0, 2.5, 5.5, 8.5], ENV['group_labels'])
    axs[0].set_ylim(0, 37)
    axs[1].set_ylim(0, 43)
    if ENV['do_axlabels']:
        for ax in axs: ax.set_ylabel("blastomere area CV, %", fontsize=12)
        ENV['letters'](axs, (0.01, 0.92))
        axs[0].set_title("2 blastomeres", y=0.9, fontsize=12)
        axs[1].set_title("4 blastomeres", y=0.9, fontsize=12)
        c = cmpplot2
        c(axs[1], '***', 5, 6, 35)
    else:
        for ax in axs: ax.set_ylabel(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV['figsave'](ENV['outd']/f"{plotname()}.svg", fig)
        
    else:
        ENV['figsave'](ENV['outd']/f"{plotname()}-A.svg", fig0)
        ENV['figsave'](ENV['outd']/f"{plotname()}-B.svg", fig1)



def plot_FIG10_5mC_PI_ratio_v2(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    empty = np.array([np.nan])
    gdata = []
    for_anova = []
    for group in range(4):
        for div, ncells in enumerate([2, 4, 8]):
            for_anova.append(pd.DataFrame(
                {"div": div+1, "group": group, "ratio": t.loc[group, ncells].Ratio.values}
            ))
    for_anova = pd.concat(for_anova)
    hook['FIG10_for_anova'] = for_anova

    markers = ["o", "^", "s", "*",]
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    if ENV['do_axlabels']:
        fig, axs = plt.subplots(1, 3, figsize=(ENV['axw']*3.3, ENV['axh']), 
            gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        _ = lambda: plt.subplots(1, 1, figsize=(ENV['axw']*1.1, ENV['axh']),
            sharex=True, layout="compressed")
        figs = []; axs=[]
        for i in range(3):
            fig, ax = _(); figs.append(fig); axs.append(ax)
        
    gdatas = for_anova.set_index(["div", "group"]).sort_index()
    for div in range(1, 4):
        ax = axs[div-1]
        for group in range(4):
            data = gdatas.loc[div, group].values
            xs = group + ptshist(data, hw*2)
            ax.plot(xs, data, color=ENV["group_colors"][group], 
                ls="none", marker=markers[group], **scatter_kw)
            ax.boxplot(data, positions=[group], manage_ticks=False, **boxplot_kw)
        ax.axhline(1, c="k", alpha=0.8, lw=0.8)
        ax.set_xticks(np.arange(4), ENV["group_labels"])
        # ax.set_yscale("log", base=2)
        # ax.set_yticks([1/8, 1/4, 1/2, 1, 2, 4, 8], "1/8 1/4 1/2 1 2 4 8".split())
    axs[2].set_ylim(0, 7.5)
    
    if ENV["do_axlabels"]:
        for i, n in enumerate([2, 4, 8]):
            axs[i].set_ylabel(f"{n} blastomeres 5mC/PI, a.u.", fontsize=12)
        ENV["letters"](axs, ENV['letpos'])
        c = cmpplot2
        c(axs[0], "*", 0, 1, 3.0) # 2bl c vs 35
        c(axs[2], "**", 0, 1, 3.5) # 8bl c vs 35
        c(axs[2], "***", 0, 1, 6.0) # 8bl c vs 37
        c(axs[2], "*", 0, 3, 6.7) # 8bl c vs 39
    else:
        for i in range(3): axs[i].set_ylabel(" ", fontsize=12)
    if ENV["do_axlabels"]:
        ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)
    else:
        for fig, l in zip(figs, "ABC"):
            ENV["figsave"](ENV["outd"] / f"{plotname()}-{l}.svg", fig)



def plot_FIG10b_5mC_PI_ratio_CV_v2(ivf_embryo_methylation_v0, opts=None):
    ENV = get_opts(opts)
    t = ivf_embryo_methylation_v0
    
    gb = t.reset_index().groupby(["Group", "N_cells", "No_batch", "NoEmb"])
    t = gb.std() / gb.mean() *100 # CV
    
    for_anova = []
    for group in range(4):
        for div, ncells in enumerate([2, 4, 8]):
            for_anova.append(pd.DataFrame(
                {"div": div+1, "group": group, "ratio": t.loc[group, ncells].Ratio.values}
            ))
    for_anova = pd.concat(for_anova)
    hook['FIG10_for_anova'] = for_anova

    markers = ["o", "^", "s", "*",]
    boxplot_kw = plt_multi_boxplot.boxplot_kw
    scatter_kw = plt_multi_boxplot.scatter_kw
    hw = boxplot_kw.get("widths", 0.5) * 0.45

    if ENV['do_axlabels']:
        fig, axs = plt.subplots(1, 3, figsize=(ENV['axw']*3.3, ENV['axh']), 
            gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        _ = lambda: plt.subplots(1, 1, figsize=(ENV['axw']*1.1, ENV['axh']),
            sharex=True, layout="compressed")
        figs = []; axs=[]
        for i in range(3):
            fig, ax = _(); figs.append(fig); axs.append(ax)
        
    gdatas = for_anova.set_index(["div", "group"]).sort_index()
    for div in range(1, 4):
        ax = axs[div-1]
        for group in range(4):
            data = gdatas.loc[div, group].values
            # xs = group + np.random.uniform(-hw, hw, len(data))
            xs = group + ptshist(data, hw*2, bins=len(data)//3)
            ax.plot(xs, data, color=ENV["group_colors"][group], 
                ls="none", marker=markers[group], **scatter_kw)
            ax.boxplot(data, positions=[group], manage_ticks=False, **boxplot_kw)
            ax.set_xticks(np.arange(4), ENV["group_labels"])
        # ax.set_yscale("log", base=2)
        # ax.set_yticks([1/8, 1/4, 1/2, 1, 2, 4, 8], "1/8 1/4 1/2 1 2 4 8".split())
    # axs[2].set_ylim(0, 7.5)
    
    if ENV["do_axlabels"]:
        for i, n in enumerate([2, 4, 8]):
            axs[i].set_ylabel(f"{n} blastomeres 5mC/PI, CV, %", fontsize=12)
        ENV["letters"](axs, ENV['letpos'])
        c = cmpplot2
        c(axs[0], '*', 0, 1, 60)
        c(axs[1], '*', 0, 1, 60)
        c(axs[1], '*', 0, 2, 70)
        c(axs[1], '*', 1, 3, 63)
        c(axs[1], '*', 2, 3, 52)
        c(axs[2], '***', 0, 2, 90)
    else:
        for i in range(3): axs[i].set_ylabel(" ", fontsize=12)
    if ENV["do_axlabels"]:
        ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)
        
    else:
        for fig, l in zip(figs, "ABC"):
            ENV["figsave"](ENV["outd"] / f"{plotname()}-{l}.svg", fig)


def plot_FIG9_cell_mass_distribution_v3(embryo_cell_mass_v0, opts=None):
    "boxplots of ICM / TE data"
    ENV = get_opts(opts)
    t = embryo_cell_mass_v0
    gb = t.groupby(["Group"])
    
    gdata_mom = [gb.get_group((g,)) for g in range(4)]
    gdata = [d.ICM.values for d in gdata_mom]
    gdata2 = [d.TE.values for d in gdata_mom]
    gdata3 = [(d.ICM / d.TE).values for d in gdata_mom]
    gdata4 = [(d.Total_cells).values for d in gdata_mom]

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
    hook['FIG9_for_anova'] = for_anova
    
    def __bxp(ax, gdata):
        markers=['o', '^', 's', '*']
        colors = ENV["group_colors"]
        boxplot_kw = plt_multi_boxplot.boxplot_kw
        hw = boxplot_kw.get("widths", 0.5) * 0.45
        for i, data in enumerate(gdata):
            xs = i + ptshist(data, hw*2)
            ax.plot(xs, data, ls='none', color=colors[i], marker=markers[i], markersize=5, **plt_multi_boxplot.scatter_kw)
            ax.boxplot(data, positions=[i,], manage_ticks=False, **boxplot_kw)
        ax.set_xticks(np.arange(4), ENV["group_labels"])

    if ENV['do_axlabels']:
        fig, axs = plt.subplots(2, 2, figsize=(ENV['axw']*2.4, ENV['axh']*2), 
                            gridspec_kw=ENV['gridkw'], sharex=True, layout="compressed")
        axs = axs.ravel()
    else:
        _ = lambda: plt.subplots(1, 1, figsize=(ENV['axw']*1.1, ENV['axh']),
            sharex=True, layout="compressed")
        figs = []; axs=[]
        for i in range(4):
            fig, ax = _(); figs.append(fig); axs.append(ax)
    
    ax = axs[0]
    __bxp(ax, gdata)
    ax.set_ylim(0, 55)
    ax = axs[1]
    __bxp(ax, gdata2)
    ax.set_ylim(0, 55)
    ax = axs[2]
    __bxp(ax, gdata3)
    ax.set_yscale('log', base=2)
    ys = "1/8 1/4 1/2 1 2 4 8".split()
    ax.set_yticks([eval(y) for y in ys], ys)
    ax.set_ylim(1 / 8, 8)
    ax = axs[3]
    __bxp(ax, gdata4)
        
    if ENV["do_axlabels"]:
        axs[0].set_ylabel("number of ICM cells", fontsize=12)
        axs[1].set_ylabel("number of TE cells", fontsize=12)
        axs[2].set_ylabel("ICM / TE ratio", fontsize=12)
        axs[3].set_ylabel("total number of cells", fontsize=12)
        ENV["letters"](axs, ENV['letpos'])

        ax = axs[1]
        c = cmpplot2
        c(ax, "***", 0, 1, 43)
        c(ax, "*", 1, 2, 47)
        ax = axs[2]
        c(ax, "*", 0, 1, 3.9)
        c(ax, "**", 1, 2, 4.9)
        ax = axs[3]
        c(ax, "***", 0, 1, 60)
        c(ax, "*",  0, 2, 65)
    else:
        for i in range(4): axs[i].set_ylabel(" ", fontsize=12)
    if ENV['do_axlabels']:
        ENV["figsave"](ENV["outd"] / f"{plotname()}.svg", fig)
    else:
        for fig, l in zip(figs, "ABCD"):
            ENV["figsave"](ENV["outd"] / f"{plotname()}-{l}.svg", fig)

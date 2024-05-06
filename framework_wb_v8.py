bxp_w = 2.5
bxp_h = 3
bxp_gskw = dict(wspace=0.1) 
bxp_letpos = (0.02, 0.92)

def augment_ts_with_energy_v0(ivf_features_v0, activity_ts_v0, /, kcal_per_kg_feed=3000, div_by_mass=True):
    log('debug', f"augment_ts_with_energy_v0(..., /, {kcal_per_kg_feed=}, {div_by_mass=})")
    def energy_burnt_from_gas_volumes(VO2: 'L/h /kg body mass', RQ: 'a.u.') -> 'kcal/ kg body mass':
        ecarb = 5.05 # kcal per L O2
        efat = 4.69 # kcal per L O2
        E = VO2 * ((RQ - 0.7)/0.3*ecarb + (1-RQ)/0.3*efat)
        return E
        # https://en.wikipedia.org/wiki/Indirect_calorimetry
    
    def energy_received_from_food(food_consumed_kg: 'kg') -> 'kcal':
        return food_consumed_kg * kcal_per_kg_feed # kcal / kg of food
        #https://www.deltafeeds.ru/catalog/dlya_laboratornykh_zhivotnykh_/814/
    
    t = activity_ts_v0
    idx_cols = ["BEHAV", "GROUP", "ANIMAL"]
    # display(t.index.names)
    if t.index.names == [ None, ]:
        no_idx = True
        t = t.set_index(idx_cols)
    elif t.index.names == idx_cols:
        no_idx = False
    else: raise ValueError("incorrect index")
    tmass = ivf_features_v0 \
        .rename(columns={'group': 'GROUP', 'animal_no': 'ANIMAL'})\
        .set_index(['GROUP', 'ANIMAL'])['Mass_13.05'].sort_index()
    tmass /= 1000

    O2_consumed = t.loc[cO2].Y * tmass
    Eout = t.loc[cO2][['X']]
    Eout['Y'] = energy_burnt_from_gas_volumes(O2_consumed.values, t.loc[cRer].Y.values)
    Ein = t.loc[cFeed][['X']]
    Ein['Y'] = energy_received_from_food(t.loc[cFeed].Y.values/1000)
    if div_by_mass: 
        Ein['Y'] /= tmass
        Eout['Y'] /= tmass
    E = pd.concat({'Eout': Eout, 'Ein': Ein, 'Ein-Eout': Ein-Eout}, names=['BEHAV'])
    # display(E)
    if no_idx: E = E.reset_index()
    return pd.concat([activity_ts_v0, E]), E


class ckpt_activity_ts_v1(_NT):
    src: Path
    dst: Path  
    ivf_features_v0_src: Path = None
    div_by_mass: bool = True
    kcal_per_kg_feed: int = 3000
        
    def ckpt_create(self):
        """
        long format (was a mistake, but tideous to fix)
        co2 : g/h
        o2: ml/h
        rer: arb.units
        feed: g/h
        drink: ml/h
        dist: m/h
        """
        t0 = pd.read_csv(self.src, sep="\t")

        columns = [cRer, cFeed, cDist, cO2, cCO2, cDrink, ]
        t0[cFeed] =t0[cFeed].fillna(0) *2 # g -> g/h
        t0[cDrink] = t0[cDrink].fillna(0) *2 # ml -> ml/h
        t0[cDist] = t0[cDist] /100 * 2 # cm -> m/h
        t0[cO2] = t0[cO2] /1e3 # ml/kg /h -> L/kg /h
        t0[cCO2] = t0[cCO2] /1e3 # ml/kg /h -> L/kg /h
        
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
        
        assert not t.isna().any().any()
        if self.ivf_features_v0_src is not None:
            # print(f"reading from : {self.ivf_features_v0_src}")
            ivf_features_v0 = pd.read_csv(self.ivf_features_v0_src, sep="\t")
            t, _ = augment_ts_with_energy_v0(ivf_features_v0, t, 
                kcal_per_kg_feed=self.kcal_per_kg_feed, div_by_mass=self.div_by_mass)
            assert not t.isna().any().any()
        
        t.to_csv(self.dst, index=False, sep="\t")
        return t
    ckpt_load = ckpt_load_default
    @staticmethod
    def ckpt_postprocess(t):
        return t.set_index(["BEHAV", "GROUP", "ANIMAL"]).sort_index()
ckpt_activity_ts_v1.name = ckpt_name(ckpt_activity_ts_v1)


def ckpt_table_ckpt_v1(proj: Path() = None, opts=None):
    ENV = get_opts(opts)
    if proj is None: proj = Path()
    d = get_caller().__name__.removeprefix("ckpt_")
    outd = proj / "checkpoints" / d
    ckpts = dict()
    ckpts[ckpt_ivf_features_v0.name] = ckpt_ivf_features_v0(
        src = proj/"data_src/ITOG2.csv",
        dst = outd/f"{ckpt_ivf_features_v0.name}.csv"
    )
    ckpts[ckpt_activity_ts_v1.name] = ckpt_activity_ts_v1(
        src = proj/"data_src/Phenomaster_2022_Total.csv", 
        dst = outd/f"{ckpt_activity_ts_v1.name}.csv",
        ivf_features_v0_src = outd/f"{ckpt_ivf_features_v0.name}.csv",
        kcal_per_kg_feed = ENV.get('kcal_per_kg_feed', 3000),
        div_by_mass = ENV.get('div_by_mass', True),
    )
    ckpts[ckpt_nonparametric_ts_sum_v0.name] = ckpt_nonparametric_ts_sum_v0(
        src = outd/f"{ckpt_activity_ts_v1.name}.csv",
        dst = outd/f"{ckpt_nonparametric_ts_sum_v0.name}.csv",
        animals = ENV['data_animals'],
        samples_per_hour = ENV['data_samples_per_hour'],
        period  = ENV['data_period'],
        columns = [cRer, cFeed, cDist, cO2, cCO2, cDrink, 'Ein', 'Eout', 'Ein-Eout'],
    )
    ckpts[ckpt_ivf_ts_cosinor_v0.name] = ckpt_ivf_ts_cosinor_v0(
        src = outd/f"{ckpt_activity_ts_v1.name}.csv",
        dst = outd/f"{ckpt_ivf_ts_cosinor_v0.name}.csv",
        animals = ENV['data_animals'],
        columns = [cRer, cFeed, cDist, cO2, cCO2, cDrink, 'Ein', 'Eout'],
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


def prep_extract_cosinor_params(cos_t, groups, data_animals, 
    behavs_models=[(cO2, "N24"), (cRer, "N24"),(cFeed, "P24"),(cDist, "P24"),]):
    """outputs: {'behav': [ {'param': group_val1}, {'param': group_val2}] """
    cosdatas = {}
    for behav, model in behavs_models:
        gdatas = []
        for g in groups:
            animals = data_animals[g]
    
            d = cos_t.loc[model, behav, g].loc[animals].copy()
            if model == "P24":
                amp = (
                    cosinor.predict(d['acrophase_h'].values, **d, link = P24.link) -
                    cosinor.predict(d['acrophase_h'].values + P24.period / 2, **d, link=P24.link)
                ) / 2
                mes = P24.link(d['mesor'].values)
                d['amplitude'] = amp
                d['mesor'] = mes
            gdatas.append(d)
        cosdatas[behav] = gdatas
    return cosdatas


def plot_energy_balance_individ_v0(activity_ts_v0, opts=None):
    t = activity_ts_v0
    ENV= get_opts(opts)
    data_animals = {
        0: [1, 2, 9, 10, 17, 18, 25, 26], 
        1: [3, 4, 5, 11, 12, 13, 19, 20, 21, 27, 28, 29], 
        2: [6, 7, 8, 14, 15, 16, 22, 23, 24, 30, 31, 32],
    }
    n = max(len(data_animals[0]), len(data_animals[2]))
    fig, axs = plt.subplots(n, 2, figsize=(10, 3*n), sharex = True) 
    for g in ENV['groups']:
        for i, a in enumerate(data_animals[g]):
            ax = axs[i, 0 if g==0 else 1]
            ein = t.loc['Ein', g, a]
            eout = t.loc['Eout', g, a]
            ax.plot(ein.X, ein.Y, c='#1f77b4', lw=1)
            ax.plot(ein.X[3:-3], nonparametric.conv_avg(ein.Y.values, 6)[3:-3], c='#1f77b4', lw=2.5)
            ax.plot(eout.X, eout.Y, c='#ff7f0e', lw=1)
            ax.plot(eout.X[3:-3], nonparametric.conv_avg(eout.Y.values, 6)[3:-3], c='#ff7f0e', lw=2.5)
            ediff = ein.Y.values - eout.Y.values
            
            ax.plot(eout.X, ediff, c='#2ca02c', lw=0.5)
            ax.axhline(1., c='#2ca02c', lw=0.8)
            # ax.set_ylim(0, 1.5)
            ax.set_title(f"{'wt' if g==0 else 'ivf'} {a}", y = 0.85)
            if a in animals_bad:
                ax.set_facecolor('#ff000030')
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)

def plot_energy_summary_v0(activity_ts_v0, opts=None):
    t = activity_ts_v0
    ENV= get_opts(opts)

    gdatas = {'Ein': [], 'Eout': [], 'Ein-Eout': []}
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    for g in groups:
        # for v in gdatas.values(): v.append([])
        [v.append([]) for v in gdatas.values()]
        for i, a in enumerate(data_animals[g]):
            ein = t.loc['Ein', g, a].Y.values
            eout = t.loc['Eout', g, a].Y.values
            ediff = t.loc['Ein-Eout', g, a].Y.values
            mdiff = (ein -eout)
            gdatas['Ein'][-1].append(ein.mean() * 24)
            gdatas['Eout'][-1].append(eout.mean() * 24)
            gdatas['Ein-Eout'][-1].append(ediff.mean() * 24)
            # data.append(mdiff)
        [v.__setitem__(-1, np.array(v[-1])) for v in gdatas.values()]
    fig, axs = plt.subplots(1, 3, figsize=((bxp_w-0.5)*3, bxp_h), gridspec_kw=bxp_gskw, layout='compressed')
    ax = axs[0]
    boxplot_main(ax, gdatas['Ein'], ENV)
    ax.set_title('input')
    ax = axs[1]
    boxplot_main(ax, gdatas['Eout'], ENV)
    ax.set_title('expence')
    ax = axs[2]
    boxplot_main(ax, gdatas['Ein-Eout'], ENV)
    ax.set_title('difference')
    axs[0].set_ylabel("mean energy turnover, kkal/24 h", fontsize=12)
    ENV['letters'](axs, (0.03, 1.03))
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)

def plot_energy_nonparametrics_v0(nonparametric_ts_sum_v0, opts=None):
    t1 = nonparametric_ts_sum_v0
    ENV = get_opts(opts)
    
    t1 = t1.set_index(['behav', 'group', 'animal'])
    gdatas = {}
    for metric in ['DFA1', 'IV', 'IS']:
        gdatas[metric] = {}
        for behav in ['Ein', 'Eout', 'Ein-Eout']:
            # gdatas[metric][behav] = []
            gdata = []
            for g in ENV['groups']:
                data = []
                # for v in gdatas.values(): v.append([])
                # [v.append([]) for v in gdatas[metric][behav].values()]
                for i, a in enumerate(ENV['data_animals'][g]):
                    x = t1.loc[(behav, g, a), metric]
                    data.append(x)
                    # data.append(mdiff)
                gdata.append(np.array(data))
            gdatas[metric][behav] = gdata

    fig, axs = plt.subplots(3, 3, figsize=((bxp_w-0.5)*3, (bxp_h-0.5)*3), 
        sharex=True, gridspec_kw=bxp_gskw, layout='compressed')
    for i, metric in enumerate(['DFA1', 'IV', 'IS']):
        for j, behav in enumerate(['Ein', 'Eout', 'Ein-Eout']):
            ax = axs[j][i]
            gdata = gdatas[metric][behav]
            boxplot_main(ax, gdata, ENV)
            if j != 2:
                plt_delete_ticks([ax, ], 'x')
                ax.set_xticks(np.arange(len(ENV['groups'])), None)
            if j == 0:
                ax.set_title(metric, fontsize=12)
            if i == 0:
                ax.set_ylabel(behav, fontsize=12)
    fig.align_ylabels()
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)


def plot_cosinor_params_v5_energy(ivf_ts_cosinor_v0, opts=None):
    ENV = get_opts(opts)
    cosinor_t = ivf_ts_cosinor_v0 #tbls["ivf_ts_cosinor_v0"]
    
    save_to = ENV['outd'] / f"{plotname()}.svg"
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    if ENV['lang']==LANG_RU:
        letters = LETTERS_RU
    elif ENV['lang']==LANG_EN:
        letters = LETTERS_EN
    letters = plt_add_letters(ENV['letters'])
    
    fig, axs = plt.subplots(2, 3, figsize=((bxp_w-0.5)*3, bxp_h*2), sharex=True, 
        gridspec_kw=bxp_gskw, layout="compressed")

    behavs_models = [('Ein', 'P24'), ('Eout', 'N24')]
    params = prep_extract_cosinor_params(cosinor_t, groups, data_animals, behavs_models)
    param_ord = ["amplitude", "mesor", "acrophase_h",]

    for i, behav in enumerate(['Ein', 'Eout']):
        gdatas = params[behav]
        for j, p in enumerate(param_ord):
            ax = axs[i, j]
            boxplot_main(ax, [d[p] for d in gdatas], ENV)

    i=0
    plt_delete_ticks(axs[i])
    axs[i, 0].set_title("amplitude")
    axs[i, 1].set_title("mesor")
    axs[i, 2].set_title("acrophase")

    lbl = lambda row, txt: axs[row, 0].set_ylabel(txt, fontsize=12)
    lbl(0, "Ein")
    lbl(1, "Eout")
    ENV['letters'](axs, bxp_letpos)
    fig.align_ylabels()
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)


def plot_energy_acrophase_difference_v0(ivf_ts_cosinor_v0, opts=None):
    cos_t = ivf_ts_cosinor_v0
    ENV = get_opts(opts)
    
    fig, axs = plt.subplots(1, 1, figsize=(bxp_w, bxp_h), gridspec_kw=bxp_gskw, layout='compressed')
    data_animals = ENV['data_animals']
    ts = (
        cos_t.loc['P24', 'Ein'] - cos_t.loc['N24', 'Eout']
    )['acrophase_h']
    # display(ts)
    gdata = []
    for g in ENV['groups']:
        data = []
        for i, a in enumerate(data_animals[g]):
            if a in animals_bad:
                continue
            data.append(ts.loc[g, a])
        gdata.append(np.array(data))

    ax = axs
    boxplot_main(ax, gdata, ENV)
    ax.set_ylabel("difference in acrophase, h", fontsize=12)
    ENV['figsave'](ENV['outd'] / f"{plotname()}.svg", fig)



def plot_mass_glucose_v1(ivf_features_v0: pd.DataFrame, opts=None):
    ENV = get_opts(opts)
    save_to = ENV['outd'] / f"{plotname()}.svg"
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    if  ENV['lang'] == LANG_RU:
        lables = ["масса тела, г", "AUC глюкозы, ммоль/л×120 мин"]
    else:
        labels = ["body mass, g", "glucose AUC, mmol/L×120 m"]
    # collect data
    t = tbls['ivf_features_v0'].set_index(["animal_no"]).sort_index()
    gdatas = [t.loc[data_animals[g]] for g in groups]
    

    # fig, axs = plt.subplots(1, 2, figsize=(2.5*2, 3), layout="compressed")
    fig, axs = plt.subplots(1, 2, figsize=(bxp_w*2, bxp_h), 
        gridspec_kw=bxp_gskw, layout="compressed")
    axs = axs.ravel()
    ax = axs[0]
    boxplot_main(ax, [d['Mass_13.05'] for d in gdatas], ENV)
    ax.set(ylim=(30, 50), yticks=[30, 35, 40, 45, 50])
    ax.set_ylabel(labels[0], fontsize=12)
    ax = axs[1]
    boxplot_main(ax, [d['AUG'] for d in gdatas], ENV)
    ax.set_ylabel(labels[1], fontsize=12)
    ENV['letters'](axs, bxp_letpos)
    ENV['figsave'](save_to, fig)


def plot_behav_averages_v6(activity_ts_v0, ivf_features_v0, opts=None):
    act_t = activity_ts_v0 # tbls['activity_ts_v0']
    feat_t = ivf_features_v0 # tbls['ivf_features_v0']
    
    ENV = get_opts(opts)
    outd = ENV['outd']
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    
    if ENV['lang'] == LANG_RU:
        title = "!!!!!!!!!"
    elif ENV['lang'] == LANG_EN:
        title = {
            cFeed: "food consumed, g/h",
            "feed/mass": "food consumed, g /(h × kg)",
            cDist: "distance travelled, m/h",
            "dist/feed": "distance/food, m/g",
            cRer: "respiratory exchange ratio",
            cO2: "O$_2$ consumed, L /(h × kg)",
        }
    opts = {
        cFeed: dict(ylim = (0.15, 0.35), yticks = np_arange(0.15, 0.05, 4)), # feed
        "feed/mass": dict(), #dict(ylim = (5.3, 7.3)), # feed/mass
        cDist: dict(ylim=(48, 140), yticks = np_arange(50, 20, 5)), # dist
        "dist/feed": dict(), # dist/feed
        cRer: dict(ylim=(0.93, 1.08), yticks = np_arange(0.95, 0.03, 5)), # rer
        cO2: dict(), 
    }

    fig, axs = plt.subplots(3, 2, figsize=(bxp_w*2, bxp_h*3), 
        gridspec_kw=bxp_gskw, layout="compressed")
    axs = axs.ravel()
    
    ms = act_t.reset_index().groupby(["BEHAV", "ANIMAL"])["Y"].mean()
    tmass = feat_t.set_index("animal_no").sort_index()["Mass_13.05"]
    
    for icol, col in enumerate([cO2, cRer, cFeed, cDist, "feed/mass", "dist/feed",]):
        if col == "feed/mass":
            ftm = ms.loc[cFeed] / tmass*1000
            gdata = [ftm.loc[data_animals[g]] for g in groups]
        elif col == "dist/feed":
            gdata = [
                ms.loc[cDist].loc[data_animals[g]] / ms.loc[cFeed].loc[data_animals[g]] 
                for g in groups]
        else:
            gdata = [ms.loc[col].loc[data_animals[g]] for g in groups]
        ax = axs[icol]
        boxplot_main(ax, gdata, ENV)
        ax.set(**opts[col])
        ax.set_ylabel(title[col], fontsize=12)
        

    plt_delete_ticks(axs[:4])
    for ax in axs[:4]: ax.set_xticks([0, 1], [None, None])
    # if no_fig:
    ENV['letters'](axs, bxp_letpos)
    ENV['figsave'](outd / f"{plotname()}.svg", fig)


def plot_hippocampus_startle_v1(ivf_features_v0: pd.DataFrame, opts=None):
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

    t = ivf_features_v0
    m = t.set_index("animal_no").sort_index()
    gdatas = [m.loc[data_animals[g]] for g in groups]
    
    fig, axs = plt.subplots(1, 2, figsize=(bxp_w*2, bxp_h), 
        gridspec_kw=bxp_gskw, layout="compressed")
    axs = axs.ravel()
    # startle
    ax = axs[0]
    boxplot_main(ax, [d['Startle_max'] for d in gdatas], opts)
    ax.set(ylim=(2., 3.6))
    # hippocampus
    ax = axs[1]
    boxplot_main(ax, [np.concatenate([d['Hippoc_L'], d['Hippoc_R']]) for d in gdatas], opts)
    ax.set(ylim=(22, 36))
    for i in range(2): axs[i].set_ylabel(title[i], fontsize=12)
    letters(axs, bxp_letpos)
    figsave(save_to, fig)


def plot_cosinor_params_v5(ivf_ts_cosinor_v0, opts=None):
    ENV = get_opts(opts)
    cosinor_t = ivf_ts_cosinor_v0 #tbls["ivf_ts_cosinor_v0"]
    
    save_to = ENV['outd'] / f"{plotname()}.svg"
    data_animals = ENV['data_animals']
    groups = ENV['groups']
    if ENV['lang']==LANG_RU:
        letters = LETTERS_RU
    elif ENV['lang']==LANG_EN:
        letters = LETTERS_EN
    letters = plt_add_letters(ENV['letters'])
    
    fig0, axs0 = plt.subplots(2, 3, figsize=((bxp_w-0.5)*3, bxp_h*2), sharex=True, 
        gridspec_kw=bxp_gskw, layout="compressed")
    fig1, axs1 = plt.subplots(2, 3, figsize=((bxp_w-0.5)*3, bxp_h*2), sharex=True, 
        gridspec_kw=bxp_gskw, layout="compressed")
    axs = np.concatenate([axs0, axs1])

    behavs_models = [('VO2(1)', 'N24'), ('RER', 'N24'), ('Feed', 'P24'), ('DistD', 'P24')]
    params = prep_extract_cosinor_params(cosinor_t, groups, data_animals, behavs_models)
    param_ord = ["amplitude", "mesor", "acrophase_h",]

    for i, behav in enumerate([cO2, cRer]):
        gdatas = params[behav]
        for j, p in enumerate(param_ord):
            ax = axs0[i, j]
            boxplot_main(ax, [d[p] for d in gdatas], ENV)

    for i, behav in enumerate([cFeed, cDist]):
        gdatas = params[behav]
        for j, p in enumerate(param_ord):
            ax = axs1[i, j]
            boxplot_main(ax, [d[p] for d in gdatas], ENV)

    for i in [0, 2]:
        plt_delete_ticks(axs[i])
        axs[i, 0].set_title("amplitude")
        axs[i, 1].set_title("mesor")
        axs[i, 2].set_title("acrophase")

    lbl = lambda row, txt: axs[row, 0].set_ylabel(txt, fontsize=12)
    lbl(0, "O$_2$ consumed, L /(h × kg)")
    lbl(1, "respiratory exchange ratio")
    lbl(2, "food consumed, g/h")
    lbl(3, "distance travelled, m/h")
    ENV['letters'](axs0, bxp_letpos)
    ENV['letters'](axs1, bxp_letpos)
    fig0.align_ylabels()
    fig1.align_ylabels()
    ENV['figsave'](ENV['outd'] / f"{plotname()}-main.svg", fig0)
    ENV['figsave'](ENV['outd'] / f"{plotname()}-supp.svg", fig1)

  
def prep_get_behav_24(act_t, sampls=2, period=24, idx_columns = ["BEHAV", "GROUP", "ANIMAL"]):
    gb = act_t.reset_index().groupby(idx_columns)
    accum = []
    time = (np.arange(15, 15+24) % 24) + 1
    time_roll = 15
    time2 = np.roll(time, time_roll)
    # display(time)
    # display(np.roll(time, time_roll))
    values, keys = [], []
    for k in gb.groups:
        y = gb.get_group(k).Y.values
        y_p = nonparametric.periodic_avg(y, sampls*period)
        assert len(y_p) == 48
        y_h = nonparametric.binned_avg(y_p, sampls)
        assert len(y_h) == 24
        y_r = np.roll(y_h, time_roll)
        values.append(y_r); keys.append(k)
    idx = pd.MultiIndex.from_tuples(keys, names=idx_columns)
    df24 = pd.DataFrame.from_records(values, columns=time2, index=idx)
    df24[0] = df24[24]
    df24.sort_index(axis=0, inplace=True)
    df24.sort_index(axis=1, inplace=True)
    # display(df24)
    return df24


def prep_get_behav_24_v2(act_t, sampls=2, period=24, idx_columns = ["BEHAV", "GROUP", "ANIMAL"]):
    """switch from local time to zeitgeber time"""
    gb = act_t.reset_index().groupby(idx_columns)
    accum = []
    # time = (np.arange(15, 15+24) % 24) + 1
    # time_roll = 15
    time = np.arange(0, 24)
    # time2 = np.roll(time, time_roll)
    time2 = time
    # display(time)
    # display(np.roll(time, time_roll))
    values, keys = [], []
    for k in gb.groups:
        y = gb.get_group(k).Y.values
        y_p = nonparametric.periodic_avg(y, sampls*period)
        assert len(y_p) == 48
        y_h = nonparametric.binned_avg(y_p, sampls)
        assert len(y_h) == 24
        # y_r = np.roll(y_h, time_roll)
        y_r = y_h
        values.append(y_r); keys.append(k)
    idx = pd.MultiIndex.from_tuples(keys, names=idx_columns)
    df24 = pd.DataFrame.from_records(values, columns=time2, index=idx)
    df24[24] = df24[0] 
    df24.sort_index(axis=0, inplace=True)
    df24.sort_index(axis=1, inplace=True)
    # display(df24)
    return df24

def plot_behav_day_night_v5(activity_ts_v0, opts=None):
    ENV = get_opts(opts)
    activity_t = activity_ts_v0 #tbls["activity_ts_v0"]
    plot_cols = [cO2, cRer, cFeed, cDist,]
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
    
    df24 = prep_get_behav_24(activity_t)
    dark = df24[[17, 18, 19, 20, 21, 22, 23, 0, 1, 2]].mean(axis='columns')
    light = df24[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,]].mean(axis="columns")
    
    def _do_plot(behavs, labels):
        fig, axs = plt.subplots(2, 2, figsize=(bxp_w*2, bxp_h*2), sharex=True, layout="compressed")
        for i, behav in enumerate(behavs):
            for j, gdatas in enumerate([dark, light]):
                ax = axs[i, j]
                gdata = [gdatas.loc[behav, g, data_animals[g]] for g in groups]
                boxplot_main(ax, gdata, ENV)
        for i in [0, 1]: axs[i,0].set_facecolor('#00000015')
        axs[0,0].set_title("dark phase", fontsize=12)
        axs[0,1].set_title("light phase", fontsize=12)
        for i,lbl in enumerate(labels):
            axs[i,0].set_ylabel(lbl, fontsize=12)
        plt_delete_ticks(axs[0])
        letters(axs, bxp_letpos)
        fig.align_ylabels()
        return fig, axs
    
    fig0, _ =_do_plot(behavs=[cO2, cRer], 
        labels = ["O$_2$ consumed, L /(h × kg)", "respiratory exchange ratio"])
    ENV['figsave'](ENV['outd'] / f"{plotname()}-main.svg", fig0)
    fig1, _ =_do_plot(behavs=[cFeed, cDist], 
        labels=["food consumed, g/h", "distance travelled, m/h"])
    ENV['figsave'](ENV['outd'] / f"{plotname()}-supp.svg", fig1)


def plotcore_lines_errors_2gr_3gr_v1(
    ax, x, gdata,
    labels = None,
    colors = None,
    stat_test=test_t_ind,
):  
    means = [d.mean(axis=0) for d in gdata]
    errs = [d.std(axis=0) / np.sqrt(len(d)) for d in gdata]
    n = len(means)
    if labels is None: labels = [None]*n
    if colors is None: colors = [None]*n
    if n == 3:
        linestyles, markers = ["-", "--", ":"], ["o", "^", "s"]
    elif n == 2:
        linestyles, markers = [None]*3, [None]*3
    else: raise NotImplementedError(f"incorrect number of groups {n=}")
    for i in range(n):
        m = means[i]
        e = errs[i] 
        ax.plot(x, m, c=colors[i], ls=linestyles[i], 
            marker=markers[i], markersize=3, label=labels[i])
        ax.fill_between(x, m-e, m+e, color=colors[i], alpha=0.2)

    if stat_test is None:
        return 
    ms = np.stack(means, axis=0)
    es = np.stack(errs, axis=0)
    top = (ms+es).max(axis=0)
    span = top.max() - (ms-es).min()
    ann1 = top + 0.12*span # line on which pvalues rest
    ann2 = ann1 + 0.12*span
    ann3 = ann2 + 0.12*span
    if n == 2:
        p = stat_test(gdata[0], gdata[1], axis=0)
        l1 = (p<=0.05) 
        l2 = (p<=0.01)
        l3 = p<=0.001
        for l, ann in [(l1, ann1), (l2, ann2), (l3, ann3)]:
            ax.plot(x[l], ann[l], c="k", marker="*", ls="none")
    elif n==3:
        for dat, c, ann in zip(gdata[1:], colors[1:], [ann1, ann2]):
            p = stat_test(gdata[0], dat, axis=0)
            l1 = (p<=0.05) & (p>0.01)
            l2 = (p<=0.01) & (p>0.001)
            l3 = p<=0.001
            # ax.plot(x, ann, lw=0.3, c="k", alpha=0.1)
            ax.plot(x[l1], ann[l1], c=c, marker="$1$", ls="none")
            ax.plot(x[l2], ann[l2], c=c, marker="$2$", ls="none")
            ax.plot(x[l3], ann[l3], c=c, marker="$3$", ls="none")


def plot_behav_24h_2gr_3gr_v3(activity_ts_v1, opts=None):
    'is split into 2 figures'
    ENV = get_opts(opts)
    outd = ENV['outd']
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", 
            "дыхательный коэффициент", "потребление O2, Л/кг/ч"]
    elif ENV['lang']==LANG_EN:
        title = ["O$_2$ consumed, L /(h × kg)", "respiratory exchange coefficient",
                 "food consumed, g/h", "distance travelled, m/h", ]
    
    df24 = prep_get_behav_24(activity_ts_v1)
    def _do_plot(behavs, labels):
        # display(df24)
        n = len(behavs)
        fig, axs = plt.subplots(n, 1, figsize=(bxp_w*3, (bxp_h-0.5)*n), 
            sharex=True, gridspec_kw=bxp_gskw, layout="compressed")
        axs = axs.ravel()
        for i, behav in enumerate(behavs):
            gdata = [df24.loc[behav, g, data_animals[g]] for g in groups]
            ax = axs[i]
            plotcore_lines_errors_2gr_3gr_v1(
                ax, gdata[0].columns, gdata,
                labels=ENV['group_labels'],
                colors=ENV['group_colors'],
                stat_test = ENV['test2'],
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25), minor=True)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.axvspan(0, 2.5, color="k", alpha=0.08)
            ax.axvspan(16.5, 24, color="k", alpha=0.08)
            ax.set_ylabel(labels[i], fontsize=12)
        axs[0].legend(loc="lower right", edgecolor="none", facecolor="none", ncols=2)
        letters(axs, (0.005, 0.91))
        fig.align_ylabels()
        return fig, axs
        
    fig, _ = _do_plot([cO2, cRer], ["O$_2$ consumed, L /(h × kg)", "respiratory exchange ratio"])
    figsave(outd / f"{plotname()}-main.svg", fig)
    fig, _ = _do_plot([cFeed, cDist], ["food consumed, g/h", "distance travelled, m/h"])
    figsave(outd / f"{plotname()}-supp.svg", fig)


def plot_behav_24h_2gr_3gr_v3_energy(activity_ts_v1, opts=None):
    'is split into 2 figures'
    ENV = get_opts(opts)
    outd = ENV['outd']
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", 
            "дыхательный коэффициент", "потребление O2, Л/кг/ч"]
    elif ENV['lang']==LANG_EN:
        title = ["O$_2$ consumed, L /(h × kg)", "respiratory exchange coefficient",
                 "food consumed, g/h", "distance travelled, m/h", ]
    
    df24 = prep_get_behav_24(activity_ts_v1)
    # global dbg
    # dbg = df24
    def _do_plot(behavs, labels):
        # display(df24)
        n = len(behavs)
        fig, axs = plt.subplots(n, 1, figsize=(bxp_w*3, (bxp_h-0.5)*n), 
            sharex=True, gridspec_kw=bxp_gskw, layout="compressed")
        axs = axs.ravel()
        for i, behav in enumerate(behavs):
            gdata = [df24.loc[behav, g, data_animals[g]] for g in groups]
            ax = axs[i]
            plotcore_lines_errors_2gr_3gr_v1(
                ax, gdata[0].columns, gdata,
                labels=ENV['group_labels'],
                colors=ENV['group_colors'],
                stat_test = ENV['test2'],
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25), minor=True)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.axvspan(0, 2.5, color="k", alpha=0.08)
            ax.axvspan(16.5, 24, color="k", alpha=0.08)
            ax.set_ylabel(labels[i], fontsize=12)
        axs[0].legend(loc="lower right", edgecolor="none", facecolor="none", ncols=2)
        letters(axs, (0.005, 0.91))
        fig.align_ylabels()
        return fig, axs
        
    fig, _ = _do_plot(['Ein', 'Eout', 'Ein-Eout'], ['Ein', 'Eout', 'Ein-Eout'])
    figsave(outd / f"{plotname()}.svg", fig)


def plot_behav_24h_2gr_3gr_v4(activity_ts_v1, opts=None):
    'is split into 2 figures, zeitgeber time'
    ENV = get_opts(opts)
    outd = ENV['outd']
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", 
            "дыхательный коэффициент", "потребление O2, Л/кг/ч"]
    elif ENV['lang']==LANG_EN:
        title = ["O$_2$ consumed, L /(h × kg)", "respiratory exchange coefficient",
                 "food consumed, g/h", "distance travelled, m/h", ]
    
    df24 = prep_get_behav_24_v2(activity_ts_v1)
    def _do_plot(behavs, labels):
        # display(df24)
        n = len(behavs)
        fig, axs = plt.subplots(n, 1, figsize=(bxp_w*3, (bxp_h-0.5)*n), 
            sharex=True, gridspec_kw=bxp_gskw, layout="compressed")
        axs = axs.ravel()
        for i, behav in enumerate(behavs):
            gdata = [df24.loc[behav, g, data_animals[g]] for g in groups]
            ax = axs[i]
            plotcore_lines_errors_2gr_3gr_v1(
                ax, gdata[0].columns, gdata,
                labels=ENV['group_labels'],
                colors=ENV['group_colors'],
                stat_test = ENV['test2'],
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25), minor=True)
            ax.set_xticks([0, 6, 12, 18, 24])
            # ax.axvspan(0, 2.5, color="k", alpha=0.08)
            # ax.axvspan(16.5, 24, color="k", alpha=0.08)
            ax.axvspan(0, 10, color="k", alpha=0.1)
            ax.axvspan(23, 24, color="k", alpha=0.1)
            ax.set_ylabel(labels[i], fontsize=12)
        axs[0].legend(loc="upper right", edgecolor="none", facecolor="none", ncols=2)
        letters(axs, (0.005, 0.91))
        fig.align_ylabels()
        return fig, axs
        
    fig, _ = _do_plot([cO2, cRer], ["O$_2$ consumed, L /(h × kg)", "respiratory exchange ratio"])
    figsave(outd / f"{plotname()}-main.svg", fig)
    fig, _ = _do_plot([cFeed, cDist], ["food consumed, g/h", "distance travelled, m/h"])
    figsave(outd / f"{plotname()}-supp.svg", fig)


def plot_behav_24h_2gr_3gr_v4_energy(activity_ts_v1, opts=None):
    'is split into 2 figures, zeitgeber time'
    ENV = get_opts(opts)
    outd = ENV['outd']
    groups = ENV['groups']
    data_animals = ENV['data_animals']
    figsave = ENV['figsave']
    letters = ENV['letters']
    if ENV['lang'] == LANG_RU:
        title = ["пройденный путь, м/ч", "съеденный корм, г/ч", 
            "дыхательный коэффициент", "потребление O2, Л/кг/ч"]
    elif ENV['lang']==LANG_EN:
        title = ["O$_2$ consumed, L /(h × kg)", "respiratory exchange coefficient",
                 "food consumed, g/h", "distance travelled, m/h", ]
    
    df24 = prep_get_behav_24_v2(activity_ts_v1)
    # global dbg
    # dbg = df24
    def _do_plot(behavs, labels):
        # display(df24)
        n = len(behavs)
        fig, axs = plt.subplots(n, 1, figsize=(bxp_w*3, (bxp_h-0.5)*n), 
            sharex=True, gridspec_kw=bxp_gskw, layout="compressed")
        axs = axs.ravel()
        for i, behav in enumerate(behavs):
            gdata = [df24.loc[behav, g, data_animals[g]] for g in groups]
            ax = axs[i]
            plotcore_lines_errors_2gr_3gr_v1(
                ax, gdata[0].columns, gdata,
                labels=ENV['group_labels'],
                colors=ENV['group_colors'],
                stat_test = ENV['test2'],
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25), minor=True)
            ax.set_xticks([0, 6, 12, 18, 24])
            # ax.axvspan(0, 2.5, color="k", alpha=0.08)
            # ax.axvspan(16.5, 24, color="k", alpha=0.08)
            ax.axvspan(0, 10, color="k", alpha=0.1)
            ax.axvspan(23, 24, color="k", alpha=0.1)
            ax.set_ylabel(labels[i], fontsize=12)
        axs[0].legend(loc="upper right", edgecolor="none", facecolor="none", ncols=2)
        letters(axs, (0.005, 0.91))
        fig.align_ylabels()
        return fig, axs
        
    fig, _ = _do_plot(['Ein', 'Eout', 'Ein-Eout'], ['Ein', 'Eout', 'Ein-Eout'])
    figsave(outd / f"{plotname()}.svg", fig)


def plot_behav_day_night_v5_energy(activity_ts_v0, opts=None):
    ENV = get_opts(opts)
    activity_t = activity_ts_v0 #tbls["activity_ts_v0"]
    plot_cols = [cO2, cRer, cFeed, cDist,]
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
    
    df24 = prep_get_behav_24(activity_t)
    dark = df24[[17, 18, 19, 20, 21, 22, 23, 0, 1, 2]].mean(axis='columns')
    light = df24[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,]].mean(axis="columns")
    
    def _do_plot(behavs, labels):
        fig, axs = plt.subplots(2, 2, figsize=(bxp_w*2, bxp_h*2), sharex=True, layout="compressed")
        for i, behav in enumerate(behavs):
            for j, gdatas in enumerate([dark, light]):
                ax = axs[i, j]
                gdata = [gdatas.loc[behav, g, data_animals[g]] for g in groups]
                boxplot_main(ax, gdata, ENV)
        for i in [0, 1]: axs[i,0].set_facecolor('#00000015')
        axs[0,0].set_title("dark phase", fontsize=12)
        axs[0,1].set_title("light phase", fontsize=12)
        for i,lbl in enumerate(labels):
            axs[i,0].set_ylabel(lbl, fontsize=12)
        plt_delete_ticks(axs[0])
        letters(axs, bxp_letpos)
        fig.align_ylabels()
        return fig, axs
    
    fig0, _ =_do_plot(behavs=['Ein', 'Eout'], 
        labels = ["Ein", "Eout"])
    ENV['figsave'](ENV['outd'] / f"{plotname()}-main.svg", fig0)
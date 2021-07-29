import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy import longdouble

import model


def b_slope_analysis(df, cut, title="b Slope Analysis"):
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(ncols=2)
    fig.suptitle(title)
    df = df[["Magnitude"]].groupby(["Magnitude"]).size().reset_index(name='counts')
    df_cut = df[df.Magnitude >= cut]
    fig.set_size_inches(12, 4)

    df.plot(ax=axes[0], title="full", x="Magnitude", y="counts").axvline(cut, c="red", label="cutoff")
    df_cut.plot(ax=axes[1], title="cut", x="Magnitude", y="counts")

    plt.show()


def minimum_year_analysis(df, years, ylim=None, title='Minimum Year Analysis'):
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout(pad=3, h_pad=3)
    fig, axes = plt.subplots(nrows=5)
    fig.set_size_inches(12, 12)
    if ylim:
        df = df[
            (df.Datetime.dt.year >= ylim[0]) &
            (df.Datetime.dt.year <= ylim[1])
            ]
    df_yearly_magnitude = df.groupby(df.Datetime.dt.year)["Magnitude"]
    df_yearly_magnitude_mean = df_yearly_magnitude.mean()
    df_yearly_count = df_yearly_magnitude.count()
    df_yearly_variance = df_yearly_magnitude.var()
    df_yearly_curve_val_b = df_yearly_magnitude.apply(lambda x: model.CalculateFeatures.b_lsq(x))
    df_yearly_min = df_yearly_magnitude.min()
    fig.suptitle(title)
    df_yearly_magnitude_mean.plot(ax=axes[0], kind="line", xlabel="", title="Magnitude Mean", legend=[])
    df_yearly_count.plot(ax=axes[1], kind="line", title="Count", xlabel="", xticks=[])
    df_yearly_curve_val_b.plot(ax=axes[2], kind="line", title="b-value", xlabel="", xticks=[])
    df_yearly_min.plot(ax=axes[3], kind="line", title="MinMag", xlabel="", xticks=[])
    df_yearly_variance.plot(ax=axes[4], kind="line", title="Variance")
    axes[0].xaxis.set_ticks_position('top')
    axes[0].set_xticks(years)
    bottom, _ = axes[-1].get_ylim()
    for ax in axes:
        ax.set_xlim(ylim[0], ylim[1])
    _, top = axes[0].get_ylim()
    for year in years:
        fig.add_artist(ConnectionPatch(xyA=[year, top], xyB=[year, bottom], coordsA="data", coordsB="data",
                                       axesA=axes[0], axesB=axes[4], color="green"))

    plt.show()

def gutenberch_richter_analysis(df_n, cut):
    df_n_cut =  df_n[df_n.Magnitude >= cut]
    unique_n, count_n = model.CalculateFeatures._cumcount_sorted_unique(df_n.Magnitude)
    unique_n_38, count_n_38 = model.CalculateFeatures._cumcount_sorted_unique(df_n_cut.Magnitude)
    b_lsq = model.CalculateFeatures.b_lsq(df_n_cut.Magnitude)
    a_lsq = model.CalculateFeatures.a_lsq(df_n_cut.Magnitude)

    y = np.array(list(map(lambda x : model.CalculateFeatures.log_gutenberg_richter_law(x,a_lsq,b_lsq),unique_n_38)))

    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots()
    fig.suptitle('Ley Gutenberg-Richter en Mexico M >= 3.8')
    fig.set_size_inches(10, 10)
    axes.set_ylabel("log10 N, Number of ocurrences")
    axes.set_xlabel("M, Magnitud")
    axes.plot(unique_n_38,y,"g")
    axes.plot(unique_n,np.log10(count_n.astype(longdouble),dtype=longdouble),"bD",markersize=3)

    axes.text(.975,.975, 'a=%.3f'%a_lsq, horizontalalignment='right',verticalalignment='top', transform=axes.transAxes)
    axes.text(.975, .9, 'b=%.3f'%b_lsq, horizontalalignment='right',verticalalignment='top', transform=axes.transAxes)


    plt.show()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
import model


def b_slope_analysis(df, cut):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Minimum Year Analysis')
    df = df[["Magnitude"]].groupby(["Magnitude"]).size().reset_index(name='counts')
    df_cut = df[df.Magnitude >= cut]
    fig.set_size_inches(12, 4)

    df.plot(ax=axes[0], title="full", x="Magnitude", y="counts").axvline(cut, c="red", label="cutoff")
    df_cut.plot(ax=axes[1], title="cut", x="Magnitude", y="counts")

    plt.show()


def minimum_year_analysis(df, years, ylim=None):
    if ylim:
        df = df[
            (df.Datetime.dt.year >= ylim[0]) &
            (df.Datetime.dt.year <= ylim[1])
            ]
    df_yearly_magnitude = df.groupby(df.Datetime.dt.year)["Magnitude"]
    df_yearly_magnitude_mean = df_yearly_magnitude.mean()
    df_yearly_count = df_yearly_magnitude.count()
    df_yearly_variance = df_yearly_magnitude.var()
    df_yearly_curve_val_b = df_yearly_magnitude.apply(
        lambda x: model.CalculateFeatures.gutenberg_richter_curve_fit(x)[1] if x.shape[0] > 2 else 0)
    df_yearly_min = df_yearly_magnitude.min()
    fig, axes = plt.subplots(nrows=5, ncols=1)
    fig.suptitle('Minimum Year Analysis')
    fig.set_size_inches(12, 8)
    df_yearly_magnitude_mean.plot(ax=axes[0], kind="line", xlabel="", title="Magnitude Mean", legend=[])
    df_yearly_count.plot(ax=axes[1], kind="line", title="Count", xlabel="", xticks=[])
    df_yearly_curve_val_b.plot(ax=axes[2], kind="line", title="b-value", xlabel="2", xticks=[])
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

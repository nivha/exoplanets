__author__ = 'Mojo'

import pylab as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import matplotlib.patches as patches
import pandas as pd
from scipy.stats import ks_2samp
from scipy.misc import comb
import itertools


def vector_dist(series, figsize=(5, 5), title=""):
    """
    show a distribution plot for this series.
    That is, the sorted series as a function of increasing integers
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    s = series.copy()
    s.sort()
    ax.plot(np.arange(s.size), s)
    plt.title(title)


def age_bins_beautify(age_bins):
    float_pat = r"[-+]?\d*\.\d+|\d+"
    intify = lambda x: str(int(float(x.group())))

    res = []
    for age in age_bins:
        beautified = "{} Myr".format(re.sub(float_pat, intify, age))
        res.append(beautified)
    return res

def digitize(series, bins):
    """
    :param series: Series wanted to be binned
    :param bins: ..
    :return: a numpy-array with respective bin label for each entry in series.
             makes sure everything falls in the min-max bins given (so we don't
             allow anything outside the borders)
    """
    def squeeze_to_limits(i):
        if i == 0:
            return 1
        elif i == len(bins):
            return len(bins)-1
        else:
            return i

    r = np.digitize(series, bins, right=True)
    return map(squeeze_to_limits, r)


def rename_labels(df, index_labels, columns_labels):
    """
    rename the labels of the given df with the given labels
    """
    df = df.rename(index=dict(enumerate(index_labels, 1)))
    df = df.rename(columns=dict(enumerate(columns_labels, 1)))
    if df.index.name == 'teff_bins':
        df.index.name = 'Teff [K]'
        df.columns.name = 'Age [Myr]'
    else:
        df.index.name = 'Age [Myr]'
        df.columns.name = 'Teff [K]'
    return df


# def output_for_parameter(df, age_categories, teff_categories, column_name):
#
#     mean_pt = df.pivot_table(index='teff_bins', columns='age_bins', values=column_name, aggfunc=np.mean)
#     mean_pt = rename_labels(mean_pt, teff_categories.levels, age_categories.levels)
#     std_pt = df.pivot_table(index='teff_bins', columns='age_bins', values=column_name, aggfunc=np.std)
#     std_pt = rename_labels(std_pt, teff_categories.levels, age_categories.levels)
#
#     plot_3d_mean_std_pivot_table(mean_pt, std_pt, zlabel="Mean {:s}".format(column_name),
#                                  title="Mean: {:s}".format(column_name))


def plot_3d_pivot_table(pivot_table, zlabel='Z', title='Some bar chart', figsize=(10, 10)):
    """
    Gets a pivot table (2d table with values inside)
    and plots it as a 3d bar chart
    """

    number_of_rows = pivot_table.shape[0]
    number_of_columns = pivot_table.shape[1]
    if number_of_rows > 4:
        raise Exception("Needs to add more colors and fix things to work..")

    colors = ['r', 'g', 'b', 'y']
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(number_of_columns)
    ylocation = 0
    ylocations = []
    for row_i in xrange(number_of_rows):
        c = colors[row_i]
        heights = pivot_table.iloc[row_i].values
        cs = [c] * number_of_columns

        ax.bar(xs, heights, zs=ylocation, zdir='y', color=cs, alpha=0.8)
        ylocations.append(ylocation)
        ylocation += 1

    ax.set_xlabel(pivot_table.axes[1].name)
    ax.set_ylabel(pivot_table.axes[0].name)
    ax.set_zlabel(zlabel)
    plt.xticks(xs, pivot_table.columns.values)
    plt.yticks(ylocations, pivot_table.index.values)
    plt.title(title)



def plot_3d_mean_std_pivot_table(mean_pt, std_pt, zlabel='Z', title='Some bar chart', figsize=(10, 10)):
    """
    Gets a pivot table (2d table with values inside)
    and plots it as a 3d bar chart
    """

    number_of_rows = mean_pt.shape[0]
    number_of_columns = mean_pt.shape[1]
    if number_of_rows > 4:
        raise Exception("Needs to add more colors and fix things to work..")

    colors = ['r', 'g', 'b', 'y']
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(number_of_columns)
    ylocation = 0
    ylocations = []
    for row_i in xrange(number_of_rows):
        c = colors[row_i]
        cs = [c] * number_of_columns
        heights_mean = mean_pt.iloc[row_i].values
        heights_std = std_pt.iloc[row_i].values

        # add fake (transplant) charts for the beauty
        # if row_i > 0:
        #     min_value = min(heights_mean)
        #     for i in xrange(1, np.random.randint(4, 10)):
        #         ax.bar(xs, [min_value] * number_of_columns, zs=ylocation, zdir='y', color=cs, alpha=0)
        #         ylocation += np.random.randint(1, 5)

        width = 0.35
        ax.bar(xs, heights_mean, width=width, zs=ylocation, zdir='y', color=cs, alpha=0.8)
        ax.bar(xs+width, heights_std, width=width, zs=ylocation, zdir='y', color=cs, alpha=0.8, hatch='/')
        ylocations.append(ylocation)
        ylocation += 1

    # ax.view_init(0,30)
    ax.set_xlabel(mean_pt.axes[1].name)
    ax.set_ylabel(mean_pt.axes[0].name)
    ax.set_zlabel(zlabel)
    plt.xticks(xs, mean_pt.columns.values)
    plt.yticks(ylocations, mean_pt.index.values)
    plt.title(title)


def plot_4_2d_pt(pt, title='Some bar chart', figsize=(10, 10)):
    """
    gets a 2d pivot-table.
     plots 4 plots, each contains the values of one line of the pivot-table
    """
    number_of_rows = pt.shape[0]
    number_of_columns = pt.shape[1]
    teff_bins = pt.index.values
    age_bins = pt.columns.values

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=figsize)
    xs = np.arange(number_of_columns)
    for row_i in xrange(number_of_rows):
        heights = pt.iloc[row_i].values

        ax = axes[row_i]
        width = 0.20
        ax.bar(xs, heights, width=width, color='b')
        ax.set_title("{} K".format(teff_bins[row_i]))

    plt.xticks(xs, age_bins_beautify(age_bins), size='x-large')
    fig.suptitle(title, size='xx-large')


def plot_16_cdfs(df, param_name, teff_categories, age_categories, title='Somechart', figsize=(10, 10)):

    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=figsize)

    for teff_bin in xrange(len(teff_categories)):
        for age_bin in xrange(len(age_categories)):
            ax = axes[teff_bin, age_bin]

            series = df[(df.teff_bins == teff_bin+1) & (df.age_bins == age_bin+1)][param_name]
            s = series.copy()
            s.sort()
            ax.plot(np.arange(s.size)/float(s.size), s)

    # Set the labels on cols and rows
    cols = age_categories
    rows = teff_categories
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=50)

    fig.tight_layout()
    fig.suptitle(title, size='xx-large', y=1.08)


def plot_ks_scores(serieses, ax, log=False):
    x = []
    for i, j in itertools.combinations(range(len(serieses)), 2):
        si = serieses[i]
        sj = serieses[j]
        score = ks_2samp(si, sj)

        x.append(score.pvalue)
    ax.barh(range(len(x)), x, log=log)


def add_colorful_yticks(ax, colors, serieses_len):
    # make colorful axis ticks for this pair
    index = 0
    for i, j in itertools.combinations(range(serieses_len), 2):
        ci = colors[j]
        cj = colors[i]

        ax.add_patch(patches.Rectangle((-0.2, index+0.1), 0.05, 0.5,  facecolor=ci, clip_on=False))
        ax.add_patch(patches.Rectangle((-0.1, index+0.1), 0.05, 0.5,  facecolor=cj, clip_on=False))

        index += 1


def plot_4_4_cdfs(df, param_name, teff_categories, age_categories, title='Somechart', figsize=(16, 4)):
    """

    """
    # sequential colors..
    colors = ['#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']

    fig1, axes1 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig2, axes2 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig3, axes3 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    for teff_bin in xrange(len(teff_categories)):
        ax = axes1[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))

        serieses = []
        for age_bin in xrange(len(age_categories)):
            series = df[(df.teff_bins == teff_bin+1) & (df.age_bins == age_bin+1)][param_name]
            s = series.copy()
            s.sort()
            serieses.append(s)
            ax.plot(np.arange(s.size)/float(s.size), s, label=age_categories[age_bin], color=colors[age_bin])

        # plot the KS-score for each age-group (4choose2 graphs)
        plot_ks_scores(serieses, axes2[teff_bin])
        plot_ks_scores(serieses, axes3[teff_bin], log=True)

    axes2[0].axes.get_yaxis().set_visible(False)
    axes3[0].axes.get_yaxis().set_visible(False)
    add_colorful_yticks(axes2[0], colors, len(age_categories))
    fig1.suptitle(title, size='xx-large', y=1.08)
    fig3.suptitle("Distribution-Pairs KS scores", size='xx-large', y=1.00)
    axes1[-1].legend(loc='upper center', bbox_to_anchor=(-1.3, -0.05), fancybox=True, shadow=True, ncol=4)
    fig1.tight_layout()
    fig2.tight_layout()
    return fig1

###############################################################################
#           Moving Window                                                     #
###############################################################################


def roll_window_regardless_values(multiplicity_s, age_s, all_stars, window_size, num_ticks):
    """
    Compute rolling window statistics regardless of the values in the given dataframes.
    multiplicity_s, age_s: pandas-serieses of the multiplicity and age of each row
    all_stars: dataframe of all the kepler stars
    num_ticks: wanted number of age-ticks to be computed in the rolling window statistics
    """
    step = age_s.max() / float(num_ticks)
    age_ticks = np.arange(0, age_s.max(), step)

    rows = []
    for age in age_ticks:
        min_margin = age-window_size
        max_margin = age+window_size

        # get the systems age-chunk
        forward_win = multiplicity_s[(age_s >= age) & (age_s < max_margin)]
        backward_win = multiplicity_s[(age_s < age) & (age_s >= min_margin)]

        # normalize with the amount of stars found in this age-gap in the all-stars sample
        all_stars_chunk = all_stars[(all_stars.Age >= age) & (all_stars.Age < max_margin)]
        all_stars_chunk_size = all_stars_chunk.shape[0]

        mean_age = all_stars_chunk.Age.mean()
        forward_win_mean = np.mean(forward_win)
        forward_win_std = np.std(forward_win)
        forward_win_sum = np.sum(forward_win)
        forward_win_size = forward_win.shape[0]
        backward_win_size = backward_win.shape[0]
        try:
            ks_2score = ks_2samp(forward_win, backward_win).pvalue
        except ValueError:
            ks_2score = np.nan

        age_row = pd.Series({
            'mean_age': mean_age,
            'forward_win_size': forward_win_size,
            'backward_win_size': backward_win_size,
            'all_star_chunk_size': all_stars_chunk_size,
            'mean': forward_win_mean,
            'std': forward_win_std,
            'systems-freq': forward_win_size / float(all_stars_chunk_size),
            'planets-freq': forward_win_sum / float(all_stars_chunk_size),
            'ks-score': ks_2score,
            })
        rows.append(age_row)

    return pd.concat(rows, axis=1).T


def rollBy(what, basis, all_stars, window):
    """
    Compute rolling window statistics regarding the values in the given dataframe.
    For each row in the dataframe we compute the window (backward and forward) and then
    make the rest of the computations.
    all_stars: dataframe of all the kepler stars
    """
    def applyToWindow(val):

        min_margin = val-window
        max_margin = val+window

        # get the systems age-chunk
        forward_win = what[(basis >= val) & (basis < max_margin)]
        backward_win = what[(basis < val) & (basis >= min_margin)]

        # normalize with the amount of stars found in this age-gap in the all-stars sample
        all_stars_chunk = all_stars[(all_stars.Age >= val) & (all_stars.Age < max_margin)]
        all_stars_chunk_size = all_stars_chunk.shape[0]

        mean_age = all_stars_chunk.Age.mean()
        forward_win_mean = np.mean(forward_win)
        forward_win_std = np.std(forward_win)
        forward_win_sum = np.sum(forward_win)
        forward_win_size = forward_win.shape[0]
        backward_win_size = backward_win.shape[0]
        ks_2score = ks_2samp(forward_win, backward_win)

        # print forward_win.value_counts(), backward_win.value_counts(), ks_2score
        # print "="*40

        return pd.Series({
            'mean_age': mean_age,
            'forward_win_size': forward_win_size,
            'backward_win_size': backward_win_size,
            'all_star_chunk_size': all_stars_chunk_size,
            'mean': forward_win_mean,
            'std': forward_win_std,
            'systems-freq': forward_win_size / float(all_stars_chunk_size),
            'planets-freq': forward_win_sum / float(all_stars_chunk_size),
            'ks-score': ks_2score.pvalue,
            })

    return basis.apply(applyToWindow)


def plot_moving_window(param, systems, all_stars, teff_categories, age_categories,
                       age_window_size=1000, window_min_obs=30, figsize=(16, 4)):


    fig0, axes0 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig1, axes1 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig2, axes2 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig3, axes3 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig4, axes4 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=figsize)
    fig0.suptitle("windows size", size='xx-large', y=1.08)
    fig1.suptitle("Multiplicity - Mean & Std", size='xx-large', y=1.08)
    fig2.suptitle("KS test pvalue", size='xx-large', y=1.08)
    fig3.suptitle("System Frequency", size='xx-large', y=1.08)
    fig4.suptitle("Planets Frequency", size='xx-large', y=1.08)

    min_age = 1e7
    max_age = 0
    for teff_bin in xrange(len(teff_categories)):
        print teff_bin

        # q is the relevant teff_bin, sorted by Age
        q = systems[systems.teff_bins == teff_bin+1].sort('Age')
        # now compute the relevant mean for this bin
        param_col = getattr(q, param)

        rolling_df = rollBy(param_col, q.Age, all_stars, age_window_size)

        ax = axes0[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))
        rolling_df.plot('mean_age', 'forward_win_size', color='b', ax=ax, label='forward')
        rolling_df.plot('mean_age', 'backward_win_size', color='r', ax=ax, label='backward')
        ax.axhline(y=window_min_obs, color='g')

        # filter values with small statistical significance (forward_win_size < window_min_obs)
        rolling_df = rolling_df[rolling_df.forward_win_size >= window_min_obs]

        # determine x-axis
        if rolling_df.mean_age.max() > max_age:
            max_age = rolling_df.mean_age.max()
        if rolling_df.mean_age.min() < min_age:
            min_age = rolling_df.mean_age.min()

        ax = axes1[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))
        rolling_df.plot('mean_age', 'mean', color='b', ax=ax, label='mean')
        rolling_df.plot('mean_age', 'std', color='g', ax=ax, label='std')

        ax = axes2[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))
        df = rolling_df[(rolling_df.forward_win_size >= window_min_obs) & (rolling_df.backward_win_size >= window_min_obs)]
        df.plot('mean_age', 'ks-score', color='b', ax=ax, logy=True)

        ax = axes3[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))
        rolling_df.plot('mean_age', 'systems-freq', color='b', ax=ax)
        ax.legend().set_visible(False)

        ax = axes4[teff_bin]
        ax.set_title("{} K".format(teff_categories[teff_bin]))
        rolling_df.plot('mean_age', 'planets-freq', color='b', ax=ax)

    for axes in [axes0, axes1, axes2, axes3, axes4]:
        axes[0].set_xlim([min_age, max_age])
        for ax in axes:
            plt.setp(ax.get_yticklabels(), visible=True)
            ax.set_xlabel("mean age [Myr]")

    # fig1.savefig('multiplicity.png', bbox_inches='tight')
    # fig2.savefig('ks-score p-value.png', bbox_inches='tight')
    # fig3.savefig('systems-frequency.png', bbox_inches='tight')
    # fig4.savefig('planets-frequency.png', bbox_inches='tight')


def plot_radius_vd_period(p, teff_categories, age_categories, title='', figsize=(10, 10)):
    pass

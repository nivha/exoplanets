__author__ = 'Mojo'
import pandas as pd
import os
import numpy as np
import pylab as plt
import re

BASE_DIR = r"D:\Users\Mojo\Google Drive\project - astrophysics\raw_data"
pd.set_option('display.width', 1000)
###############################################################################
#                   Import Stars table                                        #
###############################################################################
stars_path = os.path.join(BASE_DIR, "stars.xlsx")
stars = pd.read_excel(stars_path)
# Most of the Mass is between 0.28 and 1.3
# Filter stars without Teff or without Age
stars = stars[stars.Age.notnull()]
stars = stars[stars.Teff.notnull()]


stars = stars[stars.Age < 4000]

pl_stars = stars[stars.nasa_pl_freq.notnull()]
npl_stars = stars[stars.nasa_pl_freq.isnull()]


###############################################################################
#      Bin Stars according to planets Age, Temperature quantile-wise          #
###############################################################################

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


teff_categories, teff_qbins = pd.qcut(pl_stars.Teff.values, 4, retbins=True)
age_categories, age_qbins = pd.qcut(pl_stars.Age.values, 4, retbins=True)

pl_stars['teff_bins'] = digitize(pl_stars.Teff, teff_qbins)
pl_stars['age_bins'] = digitize(pl_stars.Age, age_qbins)
stars['teff_bins'] = digitize(stars.Teff, bins=teff_qbins)
stars['age_bins'] = digitize(stars.Age, bins=age_qbins)


###############################################################################
#         Planets frequency (host stars vs non-host stars)                    #
###############################################################################

hosts_table = pl_stars.pivot_table(index='age_bins', columns='teff_bins', values='nasa_pl_freq', margins=True, aggfunc=len)
all_table = stars.pivot_table(index='age_bins', columns='teff_bins', values='nasa_pl_freq', margins=True, aggfunc=len)
host_frequency_table = hosts_table / all_table


def rename_labels(df, index_labels, columns_labels):
    df = df.rename(index=dict(enumerate(index_labels, 1)))
    df = df.rename(columns=dict(enumerate(columns_labels, 1)))
    return df

print rename_labels(hosts_table, age_categories.levels, teff_categories.levels)
print
print rename_labels(all_table, age_categories.levels, teff_categories.levels)
print
print rename_labels(host_frequency_table, age_categories.levels, teff_categories.levels)

#c.drop(5).drop(5,1).T.plot(kind='bar')













##### With Crosstab
#
# df.to_excel(os.path.join(BASE_DIR, save_to_path))
#
# freq_bin = pd.crosstab(pl_stars.age_bins, pl_stars.mass_bins, margins=True)
# stars_freq_bin = pd.crosstab(npl_stars.age_bins, npl_stars.mass_bins, margins=True)
# save_crosstab(freq_bin, age_categories.levels, mass_categories.levels, "freq_bin1.xlsx")
# save_crosstab(nfreq_bin, age_categories.levels, mass_categories.levels, "nfreq_bin.xlsx")




# http://stackoverflow.com/questions/16947336/binning-a-dataframe-in-pandas-in-python
# age_bins = np.linspace(stars.Age.min(), stars.Age.max(), 10)
# groups = stars.groupby(np.digitize(stars.Age, age_bins))

# def binindex2binname(i, categories, bins):
#     largest_bin = bins[-1]
#     if i == len(categories.levels)+1:
#         return "> {:g}".format(largest_bin)
#     else:
#         return categories.levels[i-1]
# def get_binindex2binname_dict(categories, bins):
#     d = dict(enumerate(age_categories.levels))
#     # add the last bin, everything greater than the largest bin
# def get_bin_name_for_series(series, categories, bins):
#     series_bins_indexes = np.digitize(series, bins, right=True)
#     series_bins = map(lambda i: binindex2binname(i, categories, bins), series_bins_indexes)
#     return series_bins
# pl_stars['mass_bins_names'] = get_bin_name_for_series(pl_stars.Mass, mass_categories, mass_qbins)
# pl_stars['age_bins_names'] = get_bin_name_for_series(pl_stars.Age, age_categories, age_qbins)
# freq_bin_names = pd.crosstab(pl_stars.mass_bins_names, pl_stars.age_bins_names, margins=True)
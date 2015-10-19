# __author__ = 'Mojo'
# import pandas as pd
# import os
# import numpy as np
# import pylab as plt
# from pylab import show
# import re
# # from utils import rename_labels, output_for_parameter
#
# BASE_DIR = r"D:\Users\Mojo\Google Drive\project - astrophysics\raw_data"
# pd.set_option('display.width', 1000)
# ###############################################################################
# #                   Import Planets table                                      #
# ###############################################################################
# planets_path = os.path.join(BASE_DIR, "planets.xlsx")
# planets = pd.read_excel(planets_path)
# p = planets
# num_rows = p.shape[0]
# print "Read {:d} planets from file {:s}".format(num_rows, "planets.xlsx")
#
# ###############################################################################
# #             Clean irrelevant data (?)                                       #
# ###############################################################################
# print "Cleaning planets..."
# # only deal with planets with age and temperature
# p = p[p.Age.notnull() & p.Teff.notnull()]
# print "Dropped {:d} rows without Age or Teff".format(num_rows - p.shape[0])
# num_rows = p.shape[0]
# print "Num rows: {:d}".format(num_rows)
# # remove those planets with uncertain temperature (temp-diff < 400K)
# p = p[p.teff_34030_8826_diff.fillna(0) < 400]
# print "Dropped {:d} with teff_34030_8826_diff > 400 (undefined temperature)".format(num_rows - p.shape[0])
# num_rows = p.shape[0]
# print "Num rows: {:d}".format(num_rows)
# # remove planets with wrong period
# p = p[p.koi_period < 5000]
# print "Dropped {:d} with koi_period > 5000 days".format(num_rows - p.shape[0])
# num_rows = p.shape[0]
# print "Num rows: {:d}".format(num_rows)

# ###############################################################################
# #             Filter...                                                       #
# ###############################################################################
# print "FILTERING..."
# # leave only planets with small period (to be more correct...)
# # this is because planets with long period might be "wrong" and not really planets :(
# koi_period_threshold = 100
# p = p[p.koi_period < koi_period_threshold]
# print "Dropped {:d} with koi_period > {:d}".format(num_rows - p.shape[0], koi_period_threshold)
# num_rows = p.shape[0]
# print "Num rows: {:d}".format(num_rows)


###############################################################################
#   Create Bin Planets according to system Age, Temperature quantile-wise     #
###############################################################################

## here was function digitize... I moved it to utils.py

# # create bins according only to systems (and not planets..)
# systems = p[['KIC', 'Teff', 'Age']].groupby('KIC').agg(max)
# teff_categories, teff_qbins = pd.qcut(systems.Teff.values, 4, retbins=True)
# age_categories, age_qbins = pd.qcut(systems.Age.values, 4, retbins=True)
# p['teff_bins'] = digitize(p.Teff, teff_qbins)
# p['age_bins'] = digitize(p.Age, age_qbins)
# p['niv'] = 0
# pt = p.pivot_table(index='teff_bins', columns='age_bins', values='niv', aggfunc=len)
# pt = rename_labels(pt, teff_categories.levels, age_categories.levels)
# print "Binning with Systems:"
# print pt
#
# # # create bins according to all planets
# teff_categories, teff_qbins = pd.qcut(p.Teff.values, 4, retbins=True)
# age_categories, age_qbins = pd.qcut(p.Age.values, 4, retbins=True)
# p['teff_bins'] = digitize(p.Teff, teff_qbins)
# p['age_bins'] = digitize(p.Age, age_qbins)
# p['niv'] = 0
# pt = p.pivot_table(index='teff_bins', columns='age_bins', values='niv', aggfunc=len)
# pt = rename_labels(pt, teff_categories.levels, age_categories.levels)
# print "Binning with All Planets:"
# print pt

###############################################################################
#                   Start messing up with the data                            #
###############################################################################






###############################################################################
#             Comparing Planetary-Period to Star-Rotation                     #
###############################################################################
# p = p[p.Age.notnull()]
# p = p[p.Prot.notnull()]
# p[['KIC', 'Age', 'Prot', 'koi_period', 'pl_num']].to_excel(os.path.join(BASE_DIR, 'prot_koi_period_comp.xlsx'))
# p[p.koi_period < 5000].plot(x='Prot', y='koi_period', style='o')
# ax = plt.gca()
# ax.set_xlim(ax.get_xlim()[::-1])




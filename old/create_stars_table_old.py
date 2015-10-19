__author__ = 'Mojo'
import pandas as pd
import os
from math import sqrt
import numpy as np
from pylab import show
import pylab as plt
import re


# Output table:
# Stars table
# KIC KOI st_mass st_teff st_bmv st_prot st_age st_numpl
# Planet table
# KIC pl_mass pl_rot pl_ecc

BASE_DIR = r"D:\Users\Mojo\Google Drive\project - astrophysics\raw_data"

###############################################################################
#                   Stars Table                                               #
###############################################################################

# read 34030 stars without planets
stars_without_planets_34030_path = os.path.join(BASE_DIR, "stars_without_planets_34030.csv")
stars_without_planets_34030 = pd.read_csv(stars_without_planets_34030_path, skiprows=[1, 2])
stars = stars_without_planets_34030[['KIC', 'Teff', 'Mass', 'Prot', 'e_Prot']]
stars.rename(columns={'e_Prot': 'Prot_e'}, inplace=True)
stars.set_index('KIC', inplace=True)
# read 3354 stars with planets
# stars_3354_with_planets_path = os.path.join(BASE_DIR, "stars_hagai_kois_stars_3355.xlsx")
# stars_3354_with_planets = pd.read_excel(stars_3354_with_planets_path, skiprows=range(38))
# df = stars_3354_with_planets[['KIC', 'KOI', 'Teff', 'Prot', 'Prot_e', 'G']]
# df.set_index('KIC', inplace=True)
# stars = stars.combine_first(df)
# read 933 KOI with planets number per star
stars_933_with_planets_num_path = os.path.join(BASE_DIR, "KOI_993_periodic_pl.xlsx")
stars_933_with_planets_num = pd.read_excel(stars_933_with_planets_num_path)
df = stars_933_with_planets_num[['koi_id', 'keplerid', 'teff', 'period', 'period_err', 'pl_rad', 'pl_per', 'pl_num']]
df.rename(columns={
          'koi_id': 'KOI',
          'keplerid': 'KIC',
          'teff': 'Teff',
          'period': 'Prot',
          'period_err': 'Prot_e',
          }, inplace=True)
df.set_index('KIC', inplace=True)
stars = stars.combine_first(df)

###############################################################################
#                   Planets Table                                             #
###############################################################################

#################
# NASA
#################
planets_8828_nasa_all_kois_path = os.path.join(BASE_DIR, "planets_nasa_all_kois_8826.xlsx")
planets_8828_nasa_all_kois = pd.read_excel(planets_8828_nasa_all_kois_path, skiprows=range(155))
planets = planets_8828_nasa_all_kois[['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2', 'koi_ror', 'koi_ror_err1', 'koi_ror_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_sma', 'koi_sma_err1', 'koi_sma_err2', 'koi_dor', 'koi_dor_err1', 'koi_dor_err2', 'koi_count', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_smet', 'koi_smet_err1', 'koi_smet_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 'koi_sage', 'koi_sage_err1', 'koi_sage_err2']]
planets.rename(columns={
    'kepid': 'KIC',
}, inplace=True)

# remove false-positive planets
planets = planets[planets.koi_disposition.isin(['CONFIRMED', 'CANDIDATE'])]
# Add confirmed planets stars frequency
stars['nasa_pl_freq'] = planets.groupby('KIC').size()


#################
# Exoplanets
#################
# planets_exoplanets_path = os.path.join(BASE_DIR, "zz_exoplanets_org_alldata_modified.xlsx")
# planets_exoplanets = pd.read_excel(planets_exoplanets_path)
# planets_exoplanets = planets_exoplanets[['NAME', 'OTHERNAME', 'STAR']]
# # get a frequency histogram for only Kepler planets
# planets_exoplanets[planets_exoplanets.NAME.str.startswith('K')].groupby('STAR').size().value_counts()


###############################################################################
#                   Compute Ages                                              #
###############################################################################
# According to Color-index article in wikipedia (inverse from T(B-V))
def get_bmv(temperature):
    return (0.0217391*(230000-58*temperature+sqrt(52900000000+729*temperature**2)))/temperature

# Compute age from P and B-V according to latest formula
# from here: http://arxiv.org/pdf/1502.06965v1.pdf
def get_age(period, bmv):
    a = 0.40  # +0.3 -0.05
    b = 0.31  # +0.05 -0.02
    c = 0.45
    n = 0.55  # +0.02 -0.09
    age = (period / (a*(bmv-c)**b))**(1.0/n)
    return age


stars['B-V'] = stars['Teff'].apply(get_bmv)
stars['Age'] = stars.apply(lambda row: get_age(row['Prot'], row['B-V']), axis=1)

from pandas.util.testing import assert_frame_equal

###############################################################################
#                        Update Mass (from all_KIC)                           #
#  Not needed anymore because we're working with Temperature instead of Mass  #
###############################################################################

# stars_all_kic_path = os.path.join(BASE_DIR, "all_kic.xlsx")
# # parse only kepid and mass columns..
# df = pd.read_excel(stars_all_kic_path, parse_cols=[0, 1, 21])
# df = df[['kepid', 'mass', 'st_delivname']]
# df = df[df.st_delivname == 'q1_q17_dr24_stellar']
# df.rename(columns={
#           'kepid': 'KIC',
#           'mass': 'Mass',
#           }, inplace=True)
# df.set_index('KIC', inplace=True)
# df = df[df.Mass.notnull()]
#
# for row in stars[stars.Mass.isnull()].iterrows():
#     kic = row[0]
#     if kic in df.index:
#         mass = df.loc[row[0]].Mass
#         stars.loc[kic].Mass = mass


###############################################################################
#              Save stars table to excel file                                 #
###############################################################################

# stars_path = os.path.join(BASE_DIR, "stars.xlsx")
# stars.reset_index().to_excel(stars_path)



###############################################################################
#                   Finish Planets Table                                      #
###############################################################################

# planets['Age'] = np.nan
# stars.reset_index()
stars_kic_age = stars[stars.Age.notnull()].reset_index()[['Age', 'KIC', 'Teff', 'Prot', 'pl_num']]
p = pd.merge(planets, stars_kic_age, on='KIC', how='left')
# p = p[p.Age.notnull() & p.Teff.notnull()]
p = p[p.Age.notnull()]
p = p[p.Prot.notnull() & p.koi_period.notnull()]

p[['KIC', 'Age', 'Prot', 'koi_period', 'pl_num']].to_excel(os.path.join(BASE_DIR, 'prot_koi_period_comp1.xlsx'))

p[p.koi_period < 5000].plot(x='Prot', y='koi_period', style='o')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

__author__ = 'Mojo'
import pandas as pd
import os
from math import sqrt
import numpy as np
from pylab import show
import pylab as plt
import re

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

###############################################################################
#              Save stars table to excel file                                 #
###############################################################################

stars_path = os.path.join(BASE_DIR, "stars.xlsx")
stars.reset_index().to_excel(stars_path)


###############################################################################
#                   Finish Planets Table                                      #
###############################################################################

# Build planets table from stars into the 8826 file, filtered with Age and Temperature
stars_kic_age = stars[stars.Age.notnull()].reset_index()[['Age', 'KIC', 'Teff', 'Prot', 'pl_num']]
p = pd.merge(planets, stars_kic_age, on='KIC', how='left')
# Add a column: abs-diff of Temperature between 34030 and 8826 files
p['teff_34030_8826_diff'] = (p['Teff']-p['koi_steff']).abs()


p.to_excel(os.path.join(BASE_DIR, 'planets.xlsx'))

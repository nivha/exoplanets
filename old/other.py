__author__ = 'Mojo'
import pandas as pd
from pylab import *
import re

# Get ages for 30,000 stars
stars_periods_csv = r"D:\Users\Mojo\Google Drive\project - astrophysics\34030.csv"
kic_with_rot = pd.read_csv(stars_periods_csv, skiprows=[1, 2], index_col=2)

# Derive Color index (B-V) from effective temperature (Teff)
# # Method 1
# # formula from here: http://www.astro.sunysb.edu/fwalter/AST443/b-v_temp.html
# def BV(T):
#     return -0.865+(8540.0/T)
# # Method 2 - pyAstronomy
# # drop rows with Temperature > 4964 (cause it gets negative result under root)
# # temparature must be >= 3836
# from PyAstronomy import pyasl
# # Create class instance
# r = pyasl.Ramirez2005()
# # Convert B-V to effective temperature and back
# bv = 0.75
# feh = 0.0
# teff = 3836
# bv1 = r.teffToColor("B-V", teff, feh)
# Chosen method, according to Color-index article in wikipedia (inverse from T(B-V))
def BV(T):
    return (0.0217391*(230000-58*T+sqrt(52900000000+729*T**2)))/T

# Compute age from P and B-V according to latest formula
# from here: http://arxiv.org/pdf/1502.06965v1.pdf
def Age(P, BV):
    a = 0.40  # +0.3 -0.05
    b = 0.31  # +0.05 -0.02
    c = 0.45
    n = 0.55  # +0.02 -0.09

    age = (P / (a*(BV-c)**b))**(1.0/n)
    return age

kic_with_rot['B-V'] = kic_with_rot['Teff'].apply(BV)
kic_with_rot['Age'] =  kic_with_rot.apply(lambda row: Age(row['Prot'], row['B-V']), axis=1)

def get_age_for_KIC(KIC):
    age = kic_with_rot[kic_with_rot['KIC']==KIC]['Age'].values
    if age:
        return age[0]


# Get a mapping from KIC to KOI
hagai_koi_path = "D:\Users\Mojo\Google Drive\project - astrophysics\hagai_kois_stars_3355.xlsx"
hagai_koi = pd.read_excel(hagai_koi_path, skiprows=range(38), index_col=1)
hagai_koi['B-V'] = hagai_koi['Teff'].apply(BV)
hagai_koi['Age'] = df.apply(lambda row: Age(row['Prot'], row['B-V']), axis=1)

def get_KIC_from_KOI(KOI):
    kic = df_kois[df_kois['KOI']==KOI]['KIC'].values
    if kic:
        return kic[0]


#########################################################################
#     NASA ARCHIVES                                                     #
#     cumulative koi                                                    #
# http://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html #
#########################################################################

nasa_all_kois_path = r"D:\Users\Mojo\Google Drive\project - astrophysics\nasa_all_kois_8828.xlsx"
nasa = pd.read_excel(nasa_all_kois_path, skiprows=range(155))
# nasa[nasa.pl_hostname.str.contains('Kepler')][['pl_hostname','st_age', 'st_bmvj','st_vsini']]
nasa

##########################
#     EXOPLANETS.ORG     #
##########################

# all database from exoplanets.org
exoplanets_org_alldb_path = "D:\Users\Mojo\Google Drive\project - astrophysics\exoplanets.csv"
exoplanets = pd.read_csv(exoplanets_org_alldb_path)
# map KOI to KIC
def get_koi_from_star_name(star_name):
    match = re.match("kepler.(\d+)", star_name.lower())
    if match is None:
        return
    koi = int(match.groups()[0])
    kic = get_KIC_from_KOI(koi)
    return kic
exoplanets['KIC'] = exoplanets['STAR'].apply(get_koi_from_star_name)


# only those with KIC
exoplanets = exoplanets[exoplanets['KIC'].notnull()]
# apply age extraction to exoplanets
exoplanets['Age'] = exoplanets['KIC'].apply(get_age_for_KIC_hagai)
# # only those with age
exoplanets = exoplanets[exoplanets['Age'].notnull()]

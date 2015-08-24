__author__ = 'Mojo'
import pandas as pd
import os
import numpy as np
import pylab as plt
from pylab import show
import re

BASE_DIR = r"D:\Users\Mojo\Google Drive\project - astrophysics\raw_data"
pd.set_option('display.width', 1000)
###############################################################################
#                   Import Stars table                                        #
###############################################################################
planets_path = os.path.join(BASE_DIR, "planets.xlsx")
planets = pd.read_excel(planets_path)
p = planets

###############################################################################
#             Filter...                     #
###############################################################################








###############################################################################
#             Comparing Planetary-Period to Star-Rotation                     #
###############################################################################
p = p[p.Age.notnull()]
p = p[p.Prot.notnull()]
p[['KIC', 'Age', 'Prot', 'koi_period', 'pl_num']].to_excel(os.path.join(BASE_DIR, 'prot_koi_period_comp.xlsx'))
p[p.koi_period < 5000].plot(x='Prot', y='koi_period', style='o')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

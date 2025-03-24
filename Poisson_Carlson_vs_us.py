#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:31:06 2024

@author: simon_alamos
"""

import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

#%%

myalpha = 100
cultureOD = np.logspace(-3,-1)
frac_single = myalpha*cultureOD * np.exp(-myalpha*cultureOD)
frac_one_or_more = 1 - np.exp(-myalpha*cultureOD)
frac_none = np.exp(-myalpha*cultureOD)
frac_one_fromall = frac_single/frac_one_or_more

theiralpha = 11
cultureOD = np.logspace(-4,0)
frac_single_them = theiralpha*cultureOD * np.exp(-theiralpha*cultureOD)
frac_one_or_more_them = 1 - np.exp(-theiralpha*cultureOD)
frac_none_them = np.exp(-theiralpha*cultureOD)
frac_one_fromall_them = frac_single_them/frac_one_or_more_them


fig = plt.figure()
fig.set_size_inches(2, 2)
#plt.plot(cultureOD,frac_one_or_more,label = 'one or more')
plt.plot(cultureOD,frac_single,label = 'us')
#plt.plot(cultureOD,frac_none,label = 'zero')

#plt.plot(cultureOD,frac_one_or_more_them,'--',label = 'one or more')
plt.plot(cultureOD,frac_single_them,'--',label = 'them')
#plt.plot(cultureOD,frac_none_them,'--',label = 'zero')
plt.legend(bbox_to_anchor =(1, 1.04))
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('culture OD')
plt.ylabel('fraction of transformable cells \n transformed by a single strain')


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(cultureOD,frac_one_fromall,label = 'us')
plt.plot(cultureOD,frac_one_fromall_them,'--',label = 'them')
# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
plt.legend(bbox_to_anchor =(1, 1.04))
plt.xscale('log')
plt.grid('minor')
#plt.yscale('log')
plt.xlabel('log10 culture OD')
plt.ylabel('fraction of transformed cells \n transformed by a single strain')

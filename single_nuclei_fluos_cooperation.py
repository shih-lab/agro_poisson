#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:59:14 2024

@author: simon_alamos
"""

#%%
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib as mpl
# to enable LaTeX in labels
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# This is to figure out what is the path separator symbol in the user operating system
import os
filesep = os.sep
from os import listdir # to open all files with certain extension within a folder


# This is to let the user pick the folder where the raw data is stored without having to type the path
# I got it from https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()


from sklearn.metrics import r2_score

from string import digits

from scipy import stats
import scipy as scipy
import math
import scipy.special
from scipy import optimize
from scipy.stats import iqr
from scipy.optimize import curve_fit
from scipy.stats import sem
from scipy import ndimage
from scipy.spatial.distance import cdist

# Utilities for image processing, see https://scikit-image.org/
import skimage.io
import skimage.exposure
import skimage.measure
import skimage as sk
from skimage import filters
from skimage import morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
from skimage.feature import blob_dog, blob_log, blob_doh

# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.interpolate import interpn
from matplotlib.colors import Normalize 

# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.stats import pearsonr

#%%
def find_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix )]


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


#%%
parentFolder = '/Volumes/JSALAMOS/LeicaDM6B/1222024'
nucleiCountFiles = find_filenames(parentFolder, suffix="fluos.csv" )
nucleiCountFiles = [i for i in nucleiCountFiles if '._' not in i] #remove weird files

#%%
RFPfluosRFPonlyCtrl = [] #nuclei detected in RFP but not GFP in the control
GFPfluosGFPonlyCtrl = [] #nuclei detected in GFP but not RFP in the control
RFPfluosRFPcontransCtrl = [] #nuclei detected in RFP and RFP in the control
GFPfluosGFPcontransCtrl = [] #nuclei detected in GFP and RFP in the control
RFPfluosRFPonlyCoop = [] #nuclei detected in RFP but not GFP in he cooperation exp
GFPfluosGFPonlyCoop = [] #nuclei detected in GFP but not RFP in he cooperation exp
RFPfluosRFPcontransCoop = [] # nuclei detected in RFP and RFP in the cooperation exp
GFPfluosGFPcontransCoop = [] # nuclei detected in GFP and RFP in the cooperation exp



for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    # keep only nuclei that were detected in RFP and GFP
    Fluos_df1 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df1['normGFP'] = Fluos_df1['GFP_fluo']/np.mean(Fluos_df1['GFP_fluo'])
    Fluos_df1['normRFP'] = Fluos_df1['RFP_fluo']/np.mean(Fluos_df1['RFP_fluo'])
    Fluos_df1 = Fluos_df1.reset_index()
    # plot
    isControl = Fluos_df1['filename'][0].find('control')
    if isControl==0: #if it's a control experiment
        #plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'bo',alpha=0.2,mec='None',ms=2)
        # concatenate
        RFPfluosRFPcontransCtrl = np.append(RFPfluosRFPcontransCtrl, Fluos_df1['RFP_fluo'])
        GFPfluosGFPcontransCtrl = np.append(GFPfluosGFPcontransCtrl, Fluos_df1['GFP_fluo'])
    else: #if it's a cooperation experiment
        #plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'ro',alpha=0.3,mec='None',ms=2) 
        # concatenate
        RFPfluosRFPcontransCoop = np.append(RFPfluosRFPcontransCoop, Fluos_df1['RFP_fluo'])
        GFPfluosGFPcontransCoop = np.append(GFPfluosGFPcontransCoop, Fluos_df1['GFP_fluo'])
        
    # Now keep only nuclei that were detected in RFP but NOT in GFP
    Fluos_df2 = Fluos_df[(Fluos_df['was_detectedGFP']==False)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df2['normGFP'] = Fluos_df2['GFP_fluo']/np.mean(Fluos_df2['GFP_fluo'])
    Fluos_df2['normRFP'] = Fluos_df2['RFP_fluo']/np.mean(Fluos_df2['RFP_fluo'])
    Fluos_df2 = Fluos_df2.reset_index()
    # plot
    isControl = Fluos_df2['filename'][0].find('control')
    if isControl==0: #if it's a control experiment
        #plt.plot(Fluos_df2['GFP_fluo'],Fluos_df2['RFP_fluo'],'ko',alpha=0.15,mec='None',ms=2)
        # concatenate
        RFPfluosRFPonlyCtrl = np.append(RFPfluosRFPonlyCtrl, Fluos_df2['RFP_fluo'])
    else: #if it's a cooperation experiment
        #plt.plot(Fluos_df2['GFP_fluo'],Fluos_df2['RFP_fluo'],'yo',alpha=0.3,mec='None',ms=2)   
        # concatenate
        RFPfluosRFPonlyCoop = np.append(RFPfluosRFPonlyCoop, Fluos_df2['RFP_fluo'])
        
    # Now keep only nuclei that were detected in GFP but NOT in RFP
    Fluos_df3 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==False)]
    # get their RFP fluorescence
    Fluos_df3['normGFP'] = Fluos_df3['GFP_fluo']/np.mean(Fluos_df3['GFP_fluo'])
    Fluos_df3['normRFP'] = Fluos_df3['RFP_fluo']/np.mean(Fluos_df3['RFP_fluo'])
    Fluos_df3 = Fluos_df3.reset_index()
    # plot
    isControl = Fluos_df3['filename'][0].find('control')
    if isControl==0: #if it's a control experiment
        # concatenate
        GFPfluosGFPonlyCtrl = np.append(GFPfluosGFPonlyCtrl, Fluos_df3['GFP_fluo'])
    else: #if it's a cooperation experiment
        # concatenate
        GFPfluosGFPonlyCoop = np.append(GFPfluosGFPonlyCoop, Fluos_df3['GFP_fluo'])
            
        
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)      
# plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
# plt.plot(GFPfluosGFPcontransCoop,RFPfluosRFPcontransCoop,'ro',alpha=0.2,mec='None',ms=2)

# plt.legend(['cotransformed in control','cotransformed cooperation'],loc='upper right', bbox_to_anchor=(2.2, 1))

# plt.xlabel('GFP')
# plt.ylabel('RFP')
# plt.xscale('log')
# plt.yscale('log')

#%%
fig, ax = plt.subplots()
fig.set_size_inches(1.1, 1.1)
x = np.log10(GFPfluosGFPcontransCoop)
y = np.log10(RFPfluosRFPcontransCoop)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluosGFPcontransCoop[mask], RFPfluosRFPcontransCoop[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.9,ms=1.5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
plt.ylim(2.75,4.75)
plt.xlim(3,4.75)



fig, ax = plt.subplots()
fig.set_size_inches(1.4, 1.15)
plt.hist2d(x[mask], y[mask], (20, 20), range = [[3.1, 4.45], [2.75, 4.75]], cmap=plt.cm.magma_r)
plt.colorbar()
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')




fig, ax = plt.subplots()
fig.set_size_inches(1.2, 1.2)
density_scatter(x[mask], y[mask], ax=ax,bins = [30,30],cmap=plt.cm.magma_r,s=2)
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
corr, _ = pearsonr(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
x = GFPfluosGFPcontransCoop
y = RFPfluosRFPcontransCoop
mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'ro',alpha=0.5,mec='None',ms=2)
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(2.2, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')

fig, ax = plt.subplots()
fig.set_size_inches(2, 1)
density_scatter(x[mask], y[mask], bins = [30,30],cmap=plt.cm.jet )

fig, ax = plt.subplots()
fig.set_size_inches(2, 1)
density_scatter(x[mask], y[mask], bins = [30,30],cmap=plt.cm.rainbow )
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
corr, _ = pearsonr(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')

fig, ax = plt.subplots()
fig.set_size_inches(1.4, 1.15)
plt.hist2d(x[mask], y[mask], (30, 30), cmap=plt.cm.magma_r)
plt.colorbar()




#%% /*\*/*\*/*\*/*\*/*\* NOW THE CONTROL

fig, ax = plt.subplots()
fig.set_size_inches(1.1, 1.1)
x = np.log10(GFPfluosGFPcontransCtrl)
y = np.log10(RFPfluosRFPcontransCtrl)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)

slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluosGFPcontransCtrl[mask], RFPfluosRFPcontransCtrl[mask])
#plt.plot(x,intercept+(x*slope),'k-')    
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.9,ms=1.5,mew=0.25,color='tan')
plt.legend(['control no cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')
plt.ylim(2.75,4.75)
plt.xlim(3,4.75)


# plt.xscale('log')
# plt.yscale('log')
fig, ax = plt.subplots()
fig.set_size_inches(2, 1)
density_scatter(x[mask], y[mask], bins = [30,30] )

fig, ax = plt.subplots()
fig.set_size_inches(2, 1)
density_scatter(x[mask], y[mask], bins = [30,30],cmap=plt.cm.rainbow )
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')

fig, ax = plt.subplots()
fig.set_size_inches(1.4, 1.15)
plt.hist2d(x[mask], y[mask], (20, 20), range = [[3.1, 4.45], [2.75, 4.75]], cmap=plt.cm.magma_r)
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
plt.colorbar()


fig, ax = plt.subplots()
fig.set_size_inches(2, 1)
density_scatter(x[mask], y[mask], ax = ax, bins = [30,30],cmap=plt.cm.pink_r,s=2)
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
corr, _ = pearsonr(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')

fig, ax = plt.subplots()
fig.set_size_inches(1.4, 1.15)
plt.hist2d(x[mask], y[mask], (30, 30), cmap=plt.cm.magma_r)
plt.colorbar()





fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
x = GFPfluosGFPcontransCtrl
y = RFPfluosRFPcontransCtrl
mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
     
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'bo',alpha=0.5,mec='None',ms=2)

plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(2.2, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')
plt.xscale('log')
plt.yscale('log')

fig, ax = plt.subplots()
fig.set_size_inches(2.15, 2)
density_scatter(x[mask], y[mask], bins = [30,30],cmap=plt.cm.rainbow )
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
corr, _ = pearsonr(x[mask], y[mask])
plt.plot(x,intercept+(x*slope),'k-')
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('GFP')
plt.ylabel('RFP')


# fig, ax = plt.subplots()
# fig.set_size_inches(2.3, 2)
# density_scatter(x[mask], y[mask], bins = [30,30] )


#%%
#%%

parentFolder = '/Volumes/JSALAMOS/lsm710/2023/9-19-23'
nucleiCountFiles = find_filenames(parentFolder, suffix="fluos.csv" )
nucleiCountFiles = [i for i in nucleiCountFiles if '._' not in i] #remove weird files
#nucleiCountFiles = [i for i in nucleiCountFiles if 'inf4' in i] # keep only inf 4

#

RFPfluos05 = [] #nuclei detected in RFP in ODtot 05
GFPfluos05 = [] 
RFPfluos2 = [] #nuclei detected in RFP in ODtot 2
GFPfluos2 = [] 

for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    # keep only nuclei that were detected in RFP and GFP
    Fluos_df1 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df1['normGFP'] = Fluos_df1['GFP_fluo']/np.mean(Fluos_df1['GFP_fluo'])
    Fluos_df1['normRFP'] = Fluos_df1['RFP_fluo']/np.mean(Fluos_df1['RFP_fluo'])
    Fluos_df1 = Fluos_df1.reset_index()
    isOD05 = Fluos_df1['filename'][0].find('OD05')
    if isOD05==0: #if it's a control experiment
        #plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'bo',alpha=0.2,mec='None',ms=2)
        # concatenate
        RFPfluos05 = np.append(RFPfluos05, Fluos_df1['GFP_fluo'])
        GFPfluos05 = np.append(GFPfluos05, Fluos_df1['RFP_fluo'])
    else: #if it's an OD 2 experiment
        #plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'ro',alpha=0.3,mec='None',ms=2) 
        # concatenate
        RFPfluos2 = np.append(RFPfluos2, Fluos_df1['GFP_fluo'])
        GFPfluos2 = np.append(GFPfluos2, Fluos_df1['RFP_fluo'])

#

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
x = np.log10(GFPfluos05)
y = np.log10(RFPfluos05)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos05[mask], RFPfluos05[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)

fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()





fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
x = np.log10(GFPfluos2)
y = np.log10(RFPfluos2)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos2[mask], RFPfluos2[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)


fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()


#%%

parentFolder = '/Volumes/JSALAMOS/lsm710/2023/11-20-23'
nucleiCountFiles = find_filenames(parentFolder, suffix="fluos_int.csv" )
nucleiCountFiles = [i for i in nucleiCountFiles if '._' not in i] #remove weird files
#nucleiCountFiles = [i for i in nucleiCountFiles if 'inf4' in i] # keep only inf 4

#

RFPfluos = [] #nuclei detected in RFP in ODtot 05
GFPfluos = [] 

for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    # keep only nuclei that were detected in RFP and GFP
    Fluos_df1 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df1['normGFP'] = Fluos_df1['GFP_fluo']/np.mean(Fluos_df1['GFP_fluo'])
    Fluos_df1['normRFP'] = Fluos_df1['RFP_fluo']/np.mean(Fluos_df1['RFP_fluo'])
    Fluos_df1 = Fluos_df1.reset_index()
    isOD05 = Fluos_df1['filename'][0].find('OD05')
    RFPfluos = np.append(RFPfluos, Fluos_df1['normGFP'])
    GFPfluos = np.append(GFPfluos, Fluos_df1['normRFP'])

#

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
nonZeroRule = RFPfluos*GFPfluos>0
GFPfluos = GFPfluos[nonZeroRule]
RFPfluos = RFPfluos[nonZeroRule]
x = np.log10(GFPfluos)
y = np.log10(RFPfluos)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos[mask], RFPfluos[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)

fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()





fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
x = np.log10(GFPfluos)
y = np.log10(RFPfluos)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos[mask], RFPfluos[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)


fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()

#%%

parentFolder1 = '/Volumes/JSALAMOS/lsm710/2023/11-20-23'
nucleiCountFiles1 = find_filenames(parentFolder1, suffix="fluos_int.csv" )
nucleiCountFiles1 = [i for i in nucleiCountFiles1 if '._' not in i] #remove weird files
nucleiCountFiles1 = [i for i in nucleiCountFiles1 if 'inf5' not in i] # keep only inf 1 and 2


parentFolder2 = '/Volumes/JSALAMOS/lsm710/2023/10-16-23'
nucleiCountFiles2 = find_filenames(parentFolder2, suffix="fluos_int.csv" )
nucleiCountFiles2 = [i for i in nucleiCountFiles2 if '._' not in i] #remove weird files


nucleiCountFiles = nucleiCountFiles1 + nucleiCountFiles2
#nucleiCountFiles = [i for i in nucleiCountFiles if 'inf4' in i] # keep only inf 4

#

RFPfluos = [] #nuclei detected in RFP in ODtot 05
GFPfluos = [] 

for fileName in nucleiCountFiles1:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder1 + filesep + fileName)
    # keep only nuclei that were detected in RFP and GFP
    Fluos_df1 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df1['normGFP'] = Fluos_df1['GFP_fluo']/np.mean(Fluos_df1['GFP_fluo'])
    Fluos_df1['normRFP'] = Fluos_df1['RFP_fluo']/np.mean(Fluos_df1['RFP_fluo'])
    Fluos_df1 = Fluos_df1.reset_index()
    isOD05 = Fluos_df1['filename'][0].find('OD05')
    RFPfluos = np.append(RFPfluos, Fluos_df1['GFP_fluo'])
    GFPfluos = np.append(GFPfluos, Fluos_df1['RFP_fluo'])
    
for fileName in nucleiCountFiles2:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder2 + filesep + fileName)
    # keep only nuclei that were detected in RFP and GFP
    Fluos_df1 = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df1['normGFP'] = Fluos_df1['GFP_fluo']/np.mean(Fluos_df1['GFP_fluo'])
    Fluos_df1['normRFP'] = Fluos_df1['RFP_fluo']/np.mean(Fluos_df1['RFP_fluo'])
    Fluos_df1 = Fluos_df1.reset_index()
    isOD05 = Fluos_df1['filename'][0].find('OD05')
    RFPfluos = np.append(RFPfluos, Fluos_df1['GFP_fluo'])
    GFPfluos = np.append(GFPfluos, Fluos_df1['RFP_fluo'])


#

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
nonZeroRule = RFPfluos*GFPfluos>0
GFPfluos = GFPfluos[nonZeroRule]
RFPfluos = RFPfluos[nonZeroRule]
x = np.log10(GFPfluos)
y = np.log10(RFPfluos)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos[mask], RFPfluos[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)

fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()





fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
x = np.log10(GFPfluos)
y = np.log10(RFPfluos)
mask = ~np.isnan(x) & ~np.isnan(y)
#mask = np.isfinite(x) &  np.isfinite(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(GFPfluos[mask], RFPfluos[mask])
#plt.plot(GFPfluosGFPcontransCtrl,RFPfluosRFPcontransCtrl,'bo',alpha=0.2,mec='None',ms=2)
plt.plot(x,y,'o',mec='w',alpha=0.6,ms=5,mew=0.25,color='cornflowerblue')
plt.legend(['cotransformed cooperation'],loc='upper right', bbox_to_anchor=(3.5, 1))
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.xlabel('log10 GFP')
plt.ylabel('log10 RFP')
# plt.ylim(2.75,4.75)
# plt.xlim(3,4.75)


fig, ax = plt.subplots()
fig.set_size_inches(2.3, 2)
plt.hist2d(x[mask], y[mask], (20, 20), cmap=plt.cm.magma_r)
plt.title('slope =' + str(np.round(slope,2)) + '\n r = ' + str(np.round(r_value,2)))
plt.colorbar()









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:09:09 2024

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

import ast


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
from skimage.measure import label, regionprops, regionprops_table

from scipy.stats import sem
from scipy import ndimage
from scipy.spatial.distance import cdist


# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from pathlib import Path


#%% define functions
def euclidian_distance(point, other):
     return ((a - b) ** 2 for a, b in zip(point, other)) ** 0.5
 
def findKneighbors(x,y,Xlist,Ylist,Kneighbors):
    point = [x,y]
    restofpoints = np.column_stack((Xlist, Ylist))
    distances = [math.dist(point, p) for p in restofpoints]
    IndicesOfKclosest = np.argsort(distances)[:Kneighbors]

    return IndicesOfKclosest





def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return
    # if verbose:
    #     print('Number of labels: ' + str(nlabels))
    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]
        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]
        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def AreaFilter(imMask,minArea,maxArea,showImages):
    # min and max Area is in pixels squared
    # first label each nucleus with an ID
    distance = ndimage.distance_transform_edt(imMask)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)), labels=imMask)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=imMask)# this is a labeled image with IDs   
    # label each nucleus
    num_cells = np.max(labels_ws)                                   
    # # Print out how many we identified.
    print("Number of objects found before area filter: %s" %num_cells)  
    new_cmap = rand_cmap(num_cells, type='bright', first_color_black=True, last_color_black=False, verbose=False)
    
    # now filter by area   
    im_lab = labels_ws
    areaFiltMask = np.zeros(labels_ws.shape) #initialize the return array
    # Make an array where we'll store the cell areas.
    newAreas = []

    # Loop through each object. Remember that we have to start indexing at 1 in
    # this case!
    for i in range(num_cells):
        # Slice out the cell of interest. 
        nucleus = (im_lab == i + 1)  
        # Compute the area and add it to our array
        nucleusArea = np.sum(nucleus)
        #print(str(nucleusArea))
        # fill the return array
        if nucleusArea > minArea and nucleusArea < maxArea:
            areaFiltMask = areaFiltMask + nucleus
            newAreas.append(nucleusArea)

    # relabel the areas to remove the filtered ones from the list
    areaFiltMaskbool = areaFiltMask>0
    distance = ndimage.distance_transform_edt(areaFiltMaskbool)
    local_maxi = peak_local_max(
    distance, indices=False, footprint=np.ones((10, 10)), labels=areaFiltMaskbool)
    markers = measure.label(local_maxi)
    labels_ws2 = watershed(-distance, markers, mask=areaFiltMaskbool)
    
    num_cells2 = np.max(labels_ws2)                                   
    # Print out how many we identified.
    print("Number of objects found after area filter: %s" %num_cells2)
    if showImages:
        new_cmap = rand_cmap(num_cells, type='bright', first_color_black=True, last_color_black=False, verbose=False)
        plt.figure()
        plt.imshow(labels_ws2, cmap=new_cmap)
        plt.show() 
    
    meanArea = np.mean(newAreas)
    sdArea = np.std(newAreas)
    return(areaFiltMask,labels_ws2,num_cells2,meanArea,sdArea)


#%% test with one image

# load the images
Ch01maskPath = '/Volumes/JSALAMOS/LeicaDM6B/12112023/Masks/comp_plant1_inf1_Ch01.npy'
BFPMask = np.load(Ch01maskPath)
Ch02maskPath = '/Volumes/JSALAMOS/LeicaDM6B/12112023/Masks/comp_plant1_inf1_Ch02.npy'
GFPMask = np.load(Ch02maskPath)
Ch03maskPath = '/Volumes/JSALAMOS/LeicaDM6B/12112023/Masks/comp_plant1_inf1_Ch03.npy'
RFPMask = np.load(Ch03maskPath)

# filter objects by area
minArea = 40
maxArea = 900
[areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
newGFPMask = areaFiltMaskBFP * GFPMask
newRFPMask = areaFiltMaskBFP * RFPMask
# now apply area filter and count number of nuclei in these GFP and RFP masks as well
[areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
[areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)

#%% find positions and wether they express GFP and/or RFP
# go over each of the nuclei in BFP, store the position, then ask if there's a nucleus in GFP and/or RFP
Xpositions = []
Ypositions = []
hasGFP = []
hasRFP = []
for i in range(1,np.max(labels_ws2BFP)):
    #thisNucleusMask = (labels_ws2BFP == i).astype(int)
    label_img = label(labels_ws2BFP == i)
    regions = regionprops(label_img)
    [y0, x0] = regions[0].centroid # the x and y position of this BFP nucleus
    detectedInGFP = areaFiltMaskGFP[int(np.round(y0)), int(np.round(x0))]>0
    detectedInRFP = areaFiltMaskRFP[int(np.round(y0)), int(np.round(x0))]>0
    Xpositions.append(int(np.round(x0)))
    Ypositions.append(int(np.round(y0)))
    hasGFP.append(detectedInGFP)
    hasRFP.append(detectedInRFP)
# create a dataframe with the positions and such
# dictionary of lists 
Datadict = {'Xpos': Xpositions, 'Ypos': Ypositions, 'GFP': hasGFP,'RFP': hasRFP} 
df = pd.DataFrame(Datadict)

#%% plot the fraction of nuclei expressing either FP as a function of distance.
Xlist = df['Xpos']
Ylist = df['Ypos']
Kneighbors = np.arange(5,2000,100)
fractionGFP_mean = np.zeros(Kneighbors.size)
fractionRFP_mean = np.zeros(Kneighbors.size)
fractionGFP_sd = np.zeros(Kneighbors.size)
fractionRFP_sd = np.zeros(Kneighbors.size)
overallGFPFraction = np.sum(df['GFP'])/len(df)
overallRFPFraction = np.sum(df['RFP'])/len(df)
for count, kval in enumerate(Kneighbors):
    print(kval)
    KclosestGFPfraction = []
    KclosestRFPfraction = []
    for index, row in df.iterrows():
        if row['GFP']:
            x = row['Xpos']
            y = row['Ypos']
            KclosestIdx = findKneighbors(x,y,Xlist,Ylist,kval+1) # because the distance to itself it's 0, itslef it's its closest neighbor, so we add +1
            KclosestIdx = KclosestIdx[1:]
            KclosestGFPfraction.append(np.sum(df.loc[KclosestIdx,'GFP'])/kval)
            KclosestRFPfraction.append(np.sum(df.loc[KclosestIdx,'RFP'])/kval)
        fractionGFP_mean[count] = np.mean(KclosestGFPfraction/overallGFPFraction)
        fractionRFP_mean[count] = np.mean(KclosestRFPfraction/overallRFPFraction)
        fractionGFP_sd[count] = np.std(KclosestGFPfraction/overallGFPFraction)#/np.sqrt(len(KclosestGFPfraction))
        fractionRFP_sd[count] = np.std(KclosestRFPfraction/overallRFPFraction)#/np.sqrt(len(KclosestGFPfraction))
        
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)  
plt.errorbar(Kneighbors,fractionGFP_mean,fractionGFP_sd/30, color='g') 

#plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
plt.errorbar(Kneighbors,fractionRFP_mean,fractionRFP_sd/30, color='r')   
#plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
plt.xlabel('k closest neighboring nuclei')   
plt.ylabel('fraction of nuclei')

#%% ALTERNATIVE METHOD: moving window, plot the fraction of nuclei expressing either FP as a function of distance.
Xlist = df['Xpos']
Ylist = df['Ypos']
windows = np.hstack(([2,5,10,20,30,40],np.arange(60,400,20),np.arange(400,1000,50)))
# windowSize = 10
# Kneighbors = np.arange(windowSize,500,windowSize+1)
fractionGFP_mean = np.zeros(windows.size)
fractionRFP_mean = np.zeros(windows.size)
fractionGFP_sd = np.zeros(windows.size)
fractionRFP_sd = np.zeros(windows.size)
overallGFPFraction = np.sum(df['GFP'])/len(df)
overallRFPFraction = np.sum(df['RFP'])/len(df)
for windowidx, value in enumerate(windows[0:-1]):
    print('window starts: ' + str(windows[windowidx]) + ', window ends: ' + str(windows[windowidx+1]))
    #print(windowidx,value)
    KclosestGFPfraction = []
    KclosestRFPfraction = []
    for nucleusidx, row in df.iterrows():
        if row['GFP']:
            x = row['Xpos']
            y = row['Ypos']
            closestIdx = findKneighbors(x,y,Xlist,Ylist,np.max(1000)) # because the distance to itself it's 0, itslef it's its closest neighbor, so we add +1
            closestIdx = closestIdx[1:] # discard the first, that's the point itself
            windowframes = closestIdx[windows[windowidx]:windows[windowidx+1]]
            KclosestGFPfraction.append(np.sum(df.loc[windowframes,'GFP']) / len(windowframes))
            KclosestGFPfraction.append(np.sum(df.loc[windowframes,'RFP']) / len(windowframes))
        fractionGFP_mean[windowidx] = np.mean(KclosestGFPfraction/overallGFPFraction)
        fractionRFP_mean[windowidx] = np.mean(KclosestRFPfraction/overallRFPFraction)
        fractionGFP_sd[windowidx] = np.std(KclosestGFPfraction/overallGFPFraction)/np.sqrt(len(KclosestGFPfraction))
        fractionRFP_sd[windowidx] = np.std(KclosestRFPfraction/overallRFPFraction)/np.sqrt(len(KclosestGFPfraction))
        
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)  
plt.errorbar(windows,fractionGFP_mean,fractionGFP_sd, color='g') 

#plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
plt.errorbar(windows,fractionRFP_mean,fractionRFP_sd, color='r')   
#plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
plt.xlabel('k closest neighboring nuclei')   
plt.ylabel('fraction of nuclei')
plt.xscale('log')
plt.ylim(0.95,1.3)

#%% ALTERNATIVE METHOD 2, MOVING AVERAGE

def SpatialEnrichment_MovWindow(singleMaskFilePrefix,windowSize,FPcolor='GFP'):
    # load the binary mask of the BFP channel corresponding to this one image
    BFP_mask_file = Path(singleMaskFilePrefix + '_Ch01.npy')
    FilteredMasksOutputFolder = singleMaskFilePrefix[0:singleMaskFilePrefix.rfind('/')] + '/filteredMasks'
    isExist = os.path.exists(FilteredMasksOutputFolder)
    if not isExist: # Create a new directory because it does not exist
       os.makedirs(FilteredMasksOutputFolder)
    if BFP_mask_file.is_file(): # if the max projections exist
        print(singleMaskFilePrefix)
        filtMaskFileName = FilteredMasksOutputFolder + singleMaskFilePrefix[singleMaskFilePrefix.rfind('/'):]
        filtMaskFileNameCh01 = filtMaskFileName + '_Ch01.npy'
        filtMaskFileNameCh02 = filtMaskFileName + '_Ch02.npy'
        filtMaskFileNameCh03 = filtMaskFileName + '_Ch03.npy'
        
        if Path(filtMaskFileNameCh01).is_file():
            areaFiltMaskBFP = np.load(filtMaskFileNameCh01)
        else:
            print('loading previously generated filtered BFP mask')            
            BFPMask = np.load(singleMaskFilePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
            np.save(filtMaskFileNameCh01,areaFiltMaskBFP)
        
        if Path(filtMaskFileNameCh02).is_file():
            areaFiltMaskGFP = np.load(filtMaskFileNameCh02) 
        else:
            print('loading previously generated filtered GFP mask') 
            GFPMask = np.load(singleMaskFilePrefix + '_Ch02.npy')
            newGFPMask = areaFiltMaskBFP * GFPMask
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh02,areaFiltMaskGFP)
            
        if Path(filtMaskFileNameCh03).is_file():
            areaFiltMaskRFP = np.load(filtMaskFileNameCh03) 
        else:
            print('loading previously generated filtered RFP mask') 
            RFPMask = np.load(singleMaskFilePrefix + '_Ch03.npy')
            newRFPMask = areaFiltMaskBFP * RFPMask
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh03,areaFiltMaskRFP)
            
        df = makePositionsDf(areaFiltMaskBFP,areaFiltMaskGFP,areaFiltMaskRFP)
        
        Xlist = df['Xpos']
        Ylist = df['Ypos']
        windows = np.arange(0,100,1)
        # windowSize = 10
        # Kneighbors = np.arange(windowSize,500,windowSize+1)
        fractionGFP_mean = np.zeros(windows.size)
        fractionRFP_mean = np.zeros(windows.size)
        fractionGFP_sd = np.zeros(windows.size)
        fractionRFP_sd = np.zeros(windows.size)
        overallGFPFraction = np.sum(df['GFP'])/len(df)
        overallRFPFraction = np.sum(df['RFP'])/len(df)
        for windowidx, value in enumerate(windows[0:-windowSize]):
            print('window starts: ' + str(windows[windowidx]) + ', window ends: ' + str(windows[windowidx+windowSize]))
            #print(windowidx,value)
            KclosestGFPfraction = []
            KclosestRFPfraction = []
            for nucleusidx, row in df.iterrows():
                if row[FPcolor]:
                    x = row['Xpos']
                    y = row['Ypos']
                    closestIdx = findKneighbors(x,y,Xlist,Ylist,np.max(1000)) # because the distance to itself it's 0, itslef it's its closest neighbor, so we add +1
                    closestIdx = closestIdx[1:] # discard the first, that's the point itself
                    windowframes = closestIdx[windows[windowidx]:windows[windowidx+1]]
                    KclosestGFPfraction.append(np.sum(df.loc[windowframes,'GFP']) / len(windowframes))
                    KclosestRFPfraction.append(np.sum(df.loc[windowframes,'RFP']) / len(windowframes))
                fractionGFP_mean[windowidx] = np.mean(KclosestGFPfraction/overallGFPFraction)
                fractionRFP_mean[windowidx] = np.mean(KclosestRFPfraction/overallRFPFraction)
                fractionGFP_sd[windowidx] = np.std(KclosestGFPfraction/overallGFPFraction)/np.sqrt(len(KclosestGFPfraction))
                fractionRFP_sd[windowidx] = np.std(KclosestRFPfraction/overallRFPFraction)/np.sqrt(len(KclosestGFPfraction))
                
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)  
        plt.errorbar(windows,fractionGFP_mean,fractionGFP_sd, color='g') 
        
        #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
        plt.errorbar(windows,fractionRFP_mean,fractionRFP_sd, color='r')   
        #plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
        plt.xlabel('k closest neighboring nuclei')   
        plt.ylabel('fraction of nuclei')
        plt.xscale('log')
        #plt.ylim(0.95,1.3)
    
    return [fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd]


SpatialEnrichment_MovWindow(singleMaskFilePrefix,5,FPcolor='RFP')

#%% now do as a function for a dataset

def makePositionsDf(areaFiltMaskBFP,areaFiltMaskGFP,areaFiltMaskRFP):
    print('caclulating nuclei positions')
    Xpositions = []
    Ypositions = []
    hasGFP = []
    hasRFP = []
    distance = ndimage.distance_transform_edt(areaFiltMaskBFP)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)), labels=areaFiltMaskBFP.astype(int))
    markers = measure.label(local_maxi)
    labels_wsBFP = watershed(-distance, markers, mask=areaFiltMaskBFP)# this is a labeled image with IDs  
    for i in range(1,np.max(labels_wsBFP)):
        #thisNucleusMask = (labels_wsBFP == i).astype(int)
        label_img = label(labels_wsBFP == i)
        regions = regionprops(label_img)
        [y0, x0] = regions[0].centroid # the x and y position of this BFP nucleus
        detectedInGFP = areaFiltMaskGFP[int(np.round(y0)), int(np.round(x0))]>0
        detectedInRFP = areaFiltMaskRFP[int(np.round(y0)), int(np.round(x0))]>0
        Xpositions.append(int(np.round(x0)))
        Ypositions.append(int(np.round(y0)))
        hasGFP.append(detectedInGFP)
        hasRFP.append(detectedInRFP)
    # create a dataframe with the positions and such
    # dictionary of lists 
    Datadict = {'Xpos': Xpositions, 'Ypos': Ypositions, 'GFP': hasGFP,'RFP': hasRFP} 
    df = pd.DataFrame(Datadict)
    return df
 
def SpatialEnrichment(singleMaskFilePrefix,Nneighbors=1000,FPcolor='GFP'):
    # load the binary mask of the BFP channel corresponding to this one image
    BFP_mask_file = Path(singleMaskFilePrefix + '_Ch01.npy')
    FilteredMasksOutputFolder = singleMaskFilePrefix[0:singleMaskFilePrefix.rfind('/')] + '/filteredMasks'
    isExist = os.path.exists(FilteredMasksOutputFolder)
    if not isExist: # Create a new directory because it does not exist
       os.makedirs(FilteredMasksOutputFolder)
    if BFP_mask_file.is_file(): # if the max projections exist
        print(singleMaskFilePrefix)
        filtMaskFileName = FilteredMasksOutputFolder + singleMaskFilePrefix[singleMaskFilePrefix.rfind('/'):]
        filtMaskFileNameCh01 = filtMaskFileName + '_Ch01.npy'
        filtMaskFileNameCh02 = filtMaskFileName + '_Ch02.npy'
        filtMaskFileNameCh03 = filtMaskFileName + '_Ch03.npy'
        
        if Path(filtMaskFileNameCh01).is_file():
            areaFiltMaskBFP = np.load(filtMaskFileNameCh01)
        else:
            print('loading previously generated filtered BFP mask')            
            BFPMask = np.load(singleMaskFilePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
            np.save(filtMaskFileNameCh01,areaFiltMaskBFP)
        
        if Path(filtMaskFileNameCh02).is_file():
            areaFiltMaskGFP = np.load(filtMaskFileNameCh02) 
        else:
            print('loading previously generated filtered GFP mask') 
            GFPMask = np.load(singleMaskFilePrefix + '_Ch02.npy')
            newGFPMask = areaFiltMaskBFP * GFPMask
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh02,areaFiltMaskGFP)
            
        if Path(filtMaskFileNameCh03).is_file():
            areaFiltMaskRFP = np.load(filtMaskFileNameCh03) 
        else:
            print('loading previously generated filtered RFP mask') 
            RFPMask = np.load(singleMaskFilePrefix + '_Ch03.npy')
            newRFPMask = areaFiltMaskBFP * RFPMask
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh03,areaFiltMaskRFP)
            
        df = makePositionsDf(areaFiltMaskBFP,areaFiltMaskGFP,areaFiltMaskRFP)
        Xlist = df['Xpos']
        Ylist = df['Ypos']
        Kneighbors = np.hstack((np.arange(2,20,1),np.arange(20,400,20),np.arange(400,1000,50)))
        #Kneighbors = np.arange(2,Nneighbors,10)
        fractionGFP_mean = np.zeros(Kneighbors.size)
        fractionRFP_mean = np.zeros(Kneighbors.size)
        fractionGFP_sd = np.zeros(Kneighbors.size)
        fractionRFP_sd = np.zeros(Kneighbors.size)
        overallGFPFraction = np.sum(df['GFP'])/len(df)
        overallRFPFraction = np.sum(df['RFP'])/len(df)
        for count, kval in enumerate(Kneighbors):
            print(kval)
            KclosestGFPfraction = []
            KclosestRFPfraction = []
            for index, row in df.iterrows():
                if row[FPcolor]:
                    x = row['Xpos']
                    y = row['Ypos']
                    KclosestIdx = findKneighbors(x,y,Xlist,Ylist,kval+1) 
                    notSelfKclosestIdx = KclosestIdx[1:] # because the distance to itself it's 0, itslef it's its closest neighbor, so we added +1
                    KclosestGFPfraction.append(np.sum(df.loc[notSelfKclosestIdx,'GFP'])/kval)
                    KclosestRFPfraction.append(np.sum(df.loc[notSelfKclosestIdx,'RFP'])/kval)
                fractionGFP_mean[count] = np.mean(KclosestGFPfraction/overallGFPFraction)
                fractionRFP_mean[count] = np.mean(KclosestRFPfraction/overallRFPFraction)
                fractionGFP_sd[count] = np.std(KclosestGFPfraction/overallGFPFraction)#/np.sqrt(len(KclosestGFPfraction))
                fractionRFP_sd[count] = np.std(KclosestRFPfraction/overallRFPFraction)#/np.sqrt(len(KclosestGFPfraction))
      
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)  
        plt.errorbar(Kneighbors,fractionGFP_mean,fractionGFP_sd, color='g')        
        #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
        plt.errorbar(Kneighbors,fractionRFP_mean,fractionRFP_sd, color='r')   
        #plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
        plt.xlabel('k closest neighboring nuclei')   
        plt.ylabel('fraction of nuclei')
    
    return [fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd]




def SpatialEnrichment_window(singleMaskFilePrefix,FPcolor='GFP'):
    # load the binary mask of the BFP channel corresponding to this one image
    BFP_mask_file = Path(singleMaskFilePrefix + '_Ch01.npy')
    FilteredMasksOutputFolder = singleMaskFilePrefix[0:singleMaskFilePrefix.rfind('/')] + '/filteredMasks'
    isExist = os.path.exists(FilteredMasksOutputFolder)
    if not isExist: # Create a new directory because it does not exist
       os.makedirs(FilteredMasksOutputFolder)
    if BFP_mask_file.is_file(): # if the max projections exist
        print(singleMaskFilePrefix)
        filtMaskFileName = FilteredMasksOutputFolder + singleMaskFilePrefix[singleMaskFilePrefix.rfind('/'):]
        filtMaskFileNameCh01 = filtMaskFileName + '_Ch01.npy'
        filtMaskFileNameCh02 = filtMaskFileName + '_Ch02.npy'
        filtMaskFileNameCh03 = filtMaskFileName + '_Ch03.npy'
        
        if Path(filtMaskFileNameCh01).is_file():
            areaFiltMaskBFP = np.load(filtMaskFileNameCh01)
        else:
            print('loading previously generated filtered BFP mask')            
            BFPMask = np.load(singleMaskFilePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
            np.save(filtMaskFileNameCh01,areaFiltMaskBFP)
        
        if Path(filtMaskFileNameCh02).is_file():
            areaFiltMaskGFP = np.load(filtMaskFileNameCh02) 
        else:
            print('loading previously generated filtered GFP mask') 
            GFPMask = np.load(singleMaskFilePrefix + '_Ch02.npy')
            newGFPMask = areaFiltMaskBFP * GFPMask
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh02,areaFiltMaskGFP)
            
        if Path(filtMaskFileNameCh03).is_file():
            areaFiltMaskRFP = np.load(filtMaskFileNameCh03) 
        else:
            print('loading previously generated filtered RFP mask') 
            RFPMask = np.load(singleMaskFilePrefix + '_Ch03.npy')
            newRFPMask = areaFiltMaskBFP * RFPMask
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh03,areaFiltMaskRFP)
            
        df = makePositionsDf(areaFiltMaskBFP,areaFiltMaskGFP,areaFiltMaskRFP)
        
        Xlist = df['Xpos']
        Ylist = df['Ypos']
        windows = np.hstack(([0,3,10,20,30,40],np.arange(50,400,20),np.arange(390,1100,50)))
        # windowSize = 10
        # Kneighbors = np.arange(windowSize,500,windowSize+1)
        fractionGFP_mean = np.zeros(windows.size)
        fractionRFP_mean = np.zeros(windows.size)
        fractionGFP_sd = np.zeros(windows.size)
        fractionRFP_sd = np.zeros(windows.size)
        overallGFPFraction = np.sum(df['GFP'])/len(df)
        overallRFPFraction = np.sum(df['RFP'])/len(df)
        for windowidx, value in enumerate(windows[0:-1]):
            print('window starts: ' + str(windows[windowidx]) + ', window ends: ' + str(windows[windowidx+1]))
            #print(windowidx,value)
            KclosestGFPfraction = []
            KclosestRFPfraction = []
            for nucleusidx, row in df.iterrows():
                if row[FPcolor]:
                    x = row['Xpos']
                    y = row['Ypos']
                    closestIdx = findKneighbors(x,y,Xlist,Ylist,np.max(1000)) # because the distance to itself it's 0, itslef it's its closest neighbor, so we add +1
                    closestIdx = closestIdx[1:] # discard the first, that's the point itself
                    windowframes = closestIdx[windows[windowidx]:windows[windowidx+1]]
                    KclosestGFPfraction.append(np.sum(df.loc[windowframes,'GFP']) / len(windowframes))
                    KclosestRFPfraction.append(np.sum(df.loc[windowframes,'RFP']) / len(windowframes))
                fractionGFP_mean[windowidx] = np.mean(KclosestGFPfraction/overallGFPFraction)
                fractionRFP_mean[windowidx] = np.mean(KclosestRFPfraction/overallRFPFraction)
                fractionGFP_sd[windowidx] = np.std(KclosestGFPfraction/overallGFPFraction)/np.sqrt(len(KclosestGFPfraction))
                fractionRFP_sd[windowidx] = np.std(KclosestRFPfraction/overallRFPFraction)/np.sqrt(len(KclosestGFPfraction))
                
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)  
        plt.errorbar(windows,fractionGFP_mean,fractionGFP_sd, color='g') 
        
        #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
        plt.errorbar(windows,fractionRFP_mean,fractionRFP_sd, color='r')   
        #plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
        plt.xlabel('k closest neighboring nuclei')   
        plt.ylabel('fraction of nuclei')
        plt.xscale('log')
        #plt.ylim(0.95,1.3)
    
    return [fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd]



def SpatialEnrichment_MovWindow(singleMaskFilePrefix,windowSize,FPcolor='GFP'):
    # load the binary mask of the BFP channel corresponding to this one image
    BFP_mask_file = Path(singleMaskFilePrefix + '_Ch01.npy')
    FilteredMasksOutputFolder = singleMaskFilePrefix[0:singleMaskFilePrefix.rfind('/')] + '/filteredMasks'
    isExist = os.path.exists(FilteredMasksOutputFolder)
    if not isExist: # Create a new directory because it does not exist
       os.makedirs(FilteredMasksOutputFolder)
    if BFP_mask_file.is_file(): # if the max projections exist
        print(singleMaskFilePrefix)
        filtMaskFileName = FilteredMasksOutputFolder + singleMaskFilePrefix[singleMaskFilePrefix.rfind('/'):]
        filtMaskFileNameCh01 = filtMaskFileName + '_Ch01.npy'
        filtMaskFileNameCh02 = filtMaskFileName + '_Ch02.npy'
        filtMaskFileNameCh03 = filtMaskFileName + '_Ch03.npy'
        
        if Path(filtMaskFileNameCh01).is_file():
            areaFiltMaskBFP = np.load(filtMaskFileNameCh01)
        else:
            print('loading previously generated filtered BFP mask')            
            BFPMask = np.load(singleMaskFilePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
            np.save(filtMaskFileNameCh01,areaFiltMaskBFP)
        
        if Path(filtMaskFileNameCh02).is_file():
            areaFiltMaskGFP = np.load(filtMaskFileNameCh02) 
        else:
            print('loading previously generated filtered GFP mask') 
            GFPMask = np.load(singleMaskFilePrefix + '_Ch02.npy')
            newGFPMask = areaFiltMaskBFP * GFPMask
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh02,areaFiltMaskGFP)
            
        if Path(filtMaskFileNameCh03).is_file():
            areaFiltMaskRFP = np.load(filtMaskFileNameCh03) 
        else:
            print('loading previously generated filtered RFP mask') 
            RFPMask = np.load(singleMaskFilePrefix + '_Ch03.npy')
            newRFPMask = areaFiltMaskBFP * RFPMask
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            np.save(filtMaskFileNameCh03,areaFiltMaskRFP)
            
        df = makePositionsDf(areaFiltMaskBFP,areaFiltMaskGFP,areaFiltMaskRFP)
        
        Xlist = df['Xpos']
        Ylist = df['Ypos']
        windows = np.arange(0,100,1)
        # windowSize = 10
        # Kneighbors = np.arange(windowSize,500,windowSize+1)
        fractionGFP_mean = np.zeros(windows.size)
        fractionRFP_mean = np.zeros(windows.size)
        fractionGFP_sd = np.zeros(windows.size)
        fractionRFP_sd = np.zeros(windows.size)
        overallGFPFraction = np.sum(df['GFP'])/len(df)
        overallRFPFraction = np.sum(df['RFP'])/len(df)
        for windowidx, value in enumerate(windows[0:-windowSize]):
            print('window starts: ' + str(windows[windowidx]) + ', window ends: ' + str(windows[windowidx+windowSize]))
            #print(windowidx,value)
            KclosestGFPfraction = []
            KclosestRFPfraction = []
            for nucleusidx, row in df.iterrows():
                if row[FPcolor]:
                    x = row['Xpos']
                    y = row['Ypos']
                    closestIdx = findKneighbors(x,y,Xlist,Ylist,np.max(1000)) # because the distance to itself it's 0, itslef it's its closest neighbor, so we add +1
                    closestIdx = closestIdx[1:] # discard the first, that's the point itself
                    windowframes = closestIdx[windows[windowidx]:windows[windowidx+1]]
                    KclosestGFPfraction.append(np.sum(df.loc[windowframes,'GFP']) / len(windowframes))
                    KclosestRFPfraction.append(np.sum(df.loc[windowframes,'RFP']) / len(windowframes))
                fractionGFP_mean[windowidx] = np.mean(KclosestGFPfraction/overallGFPFraction)
                fractionRFP_mean[windowidx] = np.mean(KclosestRFPfraction/overallRFPFraction)
                fractionGFP_sd[windowidx] = np.std(KclosestGFPfraction/overallGFPFraction)/np.sqrt(len(KclosestGFPfraction))
                fractionRFP_sd[windowidx] = np.std(KclosestRFPfraction/overallRFPFraction)/np.sqrt(len(KclosestGFPfraction))
                
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)  
        plt.errorbar(windows,fractionGFP_mean,fractionGFP_sd, color='g') 
        
        #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
        plt.errorbar(windows,fractionRFP_mean,fractionRFP_sd, color='r')   
        #plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
        plt.xlabel('k closest neighboring nuclei')   
        plt.ylabel('fraction of nuclei')
        plt.xscale('log')
        #plt.ylim(0.95,1.3)
    
    return [fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd]






def spatialFraction(experiment_database,experimentID,minArea=40,maxArea=900,FPcolor='GFP'):
#    ODtotdict = {"OD005":0.05,"OD01":0.1,"OD05":0.5,"OD1":1,"OD2":2,"OD3":3}
    "minArea for filtering objects by area, 75 for Leica, 60 for Zeiss"
    "maxArea, 750 for Leica, 850 for Zeiss"
    "distanceThreshold = 10 # pixels, max distance between the centroids of objects detected in two channels for counting them as a double transformation"
    "Here we do the analysis but imposing the blue nuclear mask onto the green and red masks"
    # grab the info about this experiment
    ThisExperiment_database = experiment_database[experiment_database['Experiment_ID']==experimentID]
    ODdict = ast.literal_eval(ThisExperiment_database['ODdict'].values[0])
    plantNames = ThisExperiment_database['plantNames']
    plantNames = plantNames.values[0].split(',')
    date = str(ThisExperiment_database['Date'].values[0])
    system = ThisExperiment_database['System'].values[0]
    
#    nuclei_counts_dataframe_name = 'experiment_' + str(experimentID) + '_nuclei_counts'

    if system == 'LeicaDM6B':
        #MaxProjFileList = glob.glob("/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/*.npy")
        MaskFilesLocation = '/Volumes/JSALAMOS' + filesep + system + filesep + date + '/Masks'
        # outputFolder = '/Volumes/JSALAMOS' + filesep + system + filesep +  date + filesep + nuclei_counts_dataframe_name + '.csv'
        # maxProjectionPath = "/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/"
    elif system == 'lsm710':
        #MaxProjFileList = glob.glob("/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/*.npy")
        MaskFilesLocation = '/Volumes/JSALAMOS' + filesep + system + filesep + '2024/' + date + '/Masks'
        # outputFolder = '/Volumes/JSALAMOS' + filesep + system + filesep + '2024/' +  date + filesep + nuclei_counts_dataframe_name + '.csv'
        # maxProjectionPath = "/Volumes/JSALAMOS" + filesep + system + filesep + '2024/' + date + '/Max_projections/'
        
    # #initialize a dataframe to store values
    # cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
    #         'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
    # Nuclei_counts = pd.DataFrame([], columns=cols)
    
    fractionGFP_mean_all = np.zeros([len(plantNames)*len(list(ODdict.keys())),200])
    fractionRFP_mean_all = np.zeros([len(plantNames)*len(list(ODdict.keys())),200])
    counter = 0
    for plant, plantName in enumerate(plantNames): #loop over plants
        print(plantName)
        for inf, infName in enumerate(list(ODdict.keys())): # loop over infiltration IDs
            print(infName)
            #print('infiltration: ' + str(inf/len(list(ODdict.keys())))) # for keeping track of progress
            filename = plantName + '_' + infName
            singleMaskFilePrefix = MaskFilesLocation + filesep + filename
            [fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd] = SpatialEnrichment(singleMaskFilePrefix,Nneighbors=500,FPcolor=FPcolor)
            #[fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd] = SpatialEnrichment_window(singleMaskFilePrefix,FPcolor='RFP')
            #[fractionGFP_mean,fractionRFP_mean,fractionGFP_sd,fractionRFP_sd] = SpatialEnrichment_MovWindow(singleMaskFilePrefix,10,FPcolor=FPcolor)
            fractionGFP_mean_all[counter,0:len(fractionGFP_mean)] = fractionGFP_mean
            fractionRFP_mean_all[counter,0:len(fractionRFP_mean)] = fractionRFP_mean
            counter = counter+1

    return [fractionGFP_mean_all,fractionRFP_mean_all]
#%% load the experiment database

print('navigate to the folder where the experiment database file is stored - then select any file')
file_path = filedialog.askopenfilename() # store the file path as a string
lastFileSep = file_path.rfind(filesep) # find the position of the last path separator
folderpath = file_path[0:lastFileSep] # get the part of the path corresponding to the folder where the chosen file was located
experiment_database_filePath = folderpath + filesep + 'experiment_database.csv'
experiment_database = pd.read_csv(experiment_database_filePath)

experimentIDs = experiment_database['Experiment_ID']
#experimentIDs = [30]
rawDataParentFolder = '/Volumes/JSALAMOS'

for exp in experimentIDs[0:1]:
    print(exp)
    [fractionGFP_mean_allg,fractionRFP_mean_allg] = spatialFraction(experiment_database,exp,minArea=40,maxArea=900,FPcolor='RFP')




#%%
# meanG = np.mean(fractionGFP_mean_allg,0)
# errorG = np.std(fractionGFP_mean_allg,0)/np.sqrt(len(fractionGFP_mean_allg))
# meanR = np.mean(fractionRFP_mean_allg,0)
# errorR = np.std(fractionRFP_mean_allg,0)/np.sqrt(len(fractionRFP_mean_allg))

meanG = np.mean(fractionGFP_mean_allg,0)
meanG = meanG[meanG>0]
errorG = np.std(fractionGFP_mean_allg,0)/np.sqrt(len(fractionGFP_mean_allg))
errorG = errorG[errorG>0]
meanR = np.mean(fractionRFP_mean_allg,0)
meanR = meanR[meanR>0]
errorR = np.std(fractionRFP_mean_allg,0)/np.sqrt(len(fractionRFP_mean_allg))
errorR = errorR[errorR>0]

X = np.hstack((np.arange(2,20,1),np.arange(20,400,20),np.arange(400,1000,50)))
X = np.arange(1,len(meanG)+1,1)
X =  np.hstack((np.arange(2,20,1),np.arange(20,400,20),np.arange(400,1000,50)))
# windows = np.hstack(([0,3,10,20,30,40],np.arange(50,400,20),np.arange(390,1100,50)))
# windows = windows[1:]
# end = len(windows)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)  
plt.plot(X,meanG,'-',color='limegreen')
plt.fill_between(X, meanG-errorG, meanG+errorG,alpha=0.3, edgecolor='none', facecolor='limegreen')
plt.plot(X,meanR,'-',color='orchid')
plt.fill_between(X, meanR-errorR, meanR+errorR,alpha=0.3, edgecolor='none', facecolor='orchid')

# plt.errorbar(X,meanG,errorG, color='g')        
# #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
# plt.errorbar(X,meanR,errorR, color='r')        
# #plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.grid('major')
plt.xlabel('k closest neighboring nuclei')   
plt.ylabel('enrichment in the fraction \n of detected nuclei')
#plt.xscale('log')
#plt.ylim(0.8,3.85)
plt.xlim(2.,50)



#%%
meanG = np.mean(fractionGFP_mean_allr,0)
errorG = np.std(fractionGFP_mean_allr,0)/np.sqrt(len(fractionGFP_mean_allr-1))
meanR = np.mean(fractionRFP_mean_allr,0)
errorR = np.std(fractionRFP_mean_allr,0)/np.sqrt(len(fractionRFP_mean_allr-1))
end = 49
X = np.hstack((np.arange(2,20,1),np.arange(20,400,20),np.arange(400,1000,50)))
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)  
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)  

plt.plot(X,meanG[0:end],'-',color='limegreen')
plt.fill_between(X, meanG[0:end]-errorG[0:end], meanG[0:end]+errorG[0:end],alpha=0.3, edgecolor='none', facecolor='limegreen')
plt.plot(X,meanR[0:end],'-',color='orchid')
plt.fill_between(X, meanR[0:end]-errorR[0:end], meanR[0:end]+errorR[0:end],alpha=0.3, edgecolor='none', facecolor='orchid')

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')

# plt.errorbar(np.arange(1,end+1)*40,meanG[0:end],errorG[0:end], color='g')        
# #plt.plot([0,np.max(Kneighbors)],[overallGFPFraction,overallGFPFraction],'g--')
# plt.errorbar(np.arange(1,end+1)*40,meanR[0:end],errorR[0:end], color='r')        
#plt.plot([0,np.max(Kneighbors)],[overallRFPFraction,overallRFPFraction],'r--')
plt.xlabel('k closest neighboring nuclei')   
plt.ylabel('enrichment in the fraction \n of detected nuclei')
plt.xscale('log')











#%%

SpatialEnrichment(experiment_database,exp,minArea=40,maxArea=900,distanceThreshold=5)









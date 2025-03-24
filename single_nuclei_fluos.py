#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 07:52:50 2024

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

#%% define functions
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




def find_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix )]




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

#%% load the files

# parentFolder = '/Volumes/JSALAMOS/lsm710/2024/1-15-24'
# parentFolder = '/Volumes/JSALAMOS/lsm710/2023/9-19-23'
# parentFolder = '/Volumes/JSALAMOS/LeicaDM6B/1222024'
parentFolder= '/Volumes/JSALAMOS/LeicaDM6B/10232024'

# list the lsm files in this folder
LSMNames = find_filenames(parentFolder, suffix=".lsm" )
LIFNames = find_filenames(parentFolder, suffix=".lif" )

# LSMNames = [i for i in LSMNames if 'sep' not in i]
# LSMNames = [i for i in LSMNames if 'inf3' not in i]
# LSMNames = [i for i in LSMNames if 'inf4' not in i]



# remove weird hidden files that sometimes appear whose name starts with '._'
LSMNames = [i for i in LSMNames if '._' not in i[0:2]]
print(LSMNames)

# put the file names into a dataframe
LSMNamesDF = pd.DataFrame(LSMNames)
LSMNamesDF2 = LSMNamesDF.rename(columns={0: 'fileName'})
LIFNamesDF = pd.DataFrame(LIFNames)
LIFNamesDF2 = LIFNamesDF.rename(columns={0: 'fileName'})

# # keep only the names of the files we are interested in now
# condition1 = LSMNamesDF['fileName'].str.contains('OD05|OD2')
# condition2 = LSMNamesDF['fileName'].str.contains('inf6')
# LSMNamesDF2 = LSMNamesDF[condition1]
# LSMNamesDF2 = LSMNamesDF[condition2]
# LSMNamesDF2 = LSMNamesDF2.reset_index()


#%% loop over the files and:
# 1) find the binary BFP mask (generated by the 'prepare_images' script)
# 2) apply the BFP mask to the GFP and RFP channels
# 3) calculate the fluo of the mask-filtered GFP and RFP nuclei
#fileNames = LIFNamesDF2['fileName']

#fileNames = LSMNamesDF2['fileName']
fileNames = LIFNamesDF2['fileName']
Masks_folder = parentFolder + filesep + 'Masks'
Max_projections_folder = parentFolder + filesep + 'Max_projections'
minArea = 40
maxArea = 900
pickled_files = ['vircoop_plant3_inf4-12','vircoop_plant4_inf4-15','vircoop_plant1_inf4-2','vircoop_plant3_inf3-10',
                 'vircoop_plant3_inf3-11','vircoop_plant4_inf4-13','vircoop_plant4_inf3-13','vircoop_plant5_inf4-18','vircoop_plant5_inf3-17', 'vircoop_plant1_inf4-4','vircoop_plant2_inf3-5','vircoop_plant2_inf4-6','vircoop_plant3_inf3-9','vircoop_plant3_inf4-10','vircoop_plant1_inf3-1']

pickled_files_list = [x + '.lif' for x in pickled_files]

for j, fileName in enumerate(pickled_files_list):
    name = fileName[0:fileName.find('.lif')]
    #name = fileName[0:fileName.find('.lsm')]
    print(name)
    # plant = name[name.find('plant')+5:name.find('inf')-1]
    plant = name[name.find('plant')+5:]
    # ODtot = name[0:name.find('_')]
    # ODtot = int(''.join(c for c in ODtot if c.isdigit()))
                
    # find the BFP mask of this one file
    BFPmaskPath = Masks_folder + filesep + name + '_Ch01.npy'
    BFPmask = np.load(BFPmaskPath)
    # find the other masks
    GFPmaskPath = Masks_folder + filesep + name + '_Ch03.npy' # in Leica DM6B Ch03 is RFP
    GFPmask = np.load(GFPmaskPath)
    RFPmaskPath = Masks_folder + filesep + name + '_Ch02.npy' # in Leica DM6B Ch02 is GFP
    RFPmask = np.load(RFPmaskPath)
    # find the GFP and RFP max projections, remember that in the Zeiss GFP is channel 3 and RFP is 2
    GFPimagePath = Max_projections_folder + filesep + name + '_Ch03.npy'
    GFPimage = np.load(GFPimagePath)   
    RFPimagePath = Max_projections_folder + filesep + name + '_Ch02.npy'
    RFPimage = np.load(RFPimagePath)

    # look at the images
    plt.subplot(131)
    plt.imshow(BFPmask,cmap='Blues_r')
    plt.subplot(132)
    plt.imshow(GFPimage,cmap='Greens_r')
    plt.subplot(133)
    plt.imshow(RFPimage,cmap = 'Reds_r')
    plt.show() 
    
    [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPmask,minArea,maxArea,False)
    [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(GFPmask,minArea,maxArea,False)
    MaskedGFPimage = GFPimage * areaFiltMaskGFP
    [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(RFPmask,minArea,maxArea,False)
    MaskedRFPimage = RFPimage * areaFiltMaskRFP

    # look at the images
    plt.subplot(131)
    plt.imshow(areaFiltMaskBFP,cmap=rand_cmap(NBFP))
    plt.subplot(132)
    plt.imshow(areaFiltMaskGFP,cmap=rand_cmap(NGFP))
    plt.subplot(133)
    plt.imshow(areaFiltMaskRFP,cmap=rand_cmap(NRFP))

    #initialize a dataframe to store values
    cols = ['filename','GFP_fluo','RFP_fluo','was_detectedGFP','was_detectedRFP']  
    Nuclei_fluos = pd.DataFrame([], columns=cols)

    for i in range(NBFP): # Loop through each object. Remember that we have to start indexing at 1 in this case!
         print(str(100*np.round(i/NBFP,2))+ ' % of nuclei processed')
         
         # select the binary mask corresponding to this nucleus   
         singleNucleusMask = labels_ws2BFP == (i + 1)
         # erode the edges of the BFP mask a bit
         singleNucleusMask2 = skimage.morphology.binary_erosion(singleNucleusMask,np.ones([3,3]))
         if np.sum(singleNucleusMask2)>0:
             
             # apply the mask to each channel by multiplying the binary mask by the max projected image
             singleNucleusCh02 = MaskedRFPimage * singleNucleusMask2
             singleNucleusCh03 = MaskedGFPimage * singleNucleusMask2
             
             # select only the pixels corresponding to the mask in each channel
             nucleusPixCh02 = singleNucleusCh02[singleNucleusCh02>0]
             nucleusPixCh03 = singleNucleusCh03[singleNucleusCh03>0]
             
             if any(nucleusPixCh02): #if a BFP-detected nucleus was detected in channel 2
                 wasDetectedRFP = True
                 lowerCh02 = np.percentile(nucleusPixCh02,5) # the fluo value of the 25th percetile
                 upperCh02 = np.percentile(nucleusPixCh02,95) # the fluo value of the 75th percetile
                 idx2 = (nucleusPixCh02>lowerCh02)*(nucleusPixCh02<upperCh02) #pixels between the 25th and 75th percentiles
                 fluoCh02= np.sum(nucleusPixCh02[idx2]) # average their fluorescence
             else:
                 wasDetectedRFP = False
                 singleNucleusCh02 = RFPimage * singleNucleusMask2
                 nucleusPixCh02 = singleNucleusCh02[singleNucleusCh02>0]
                 if np.sum(singleNucleusCh02) > 0:
                     lowerCh02 = np.percentile(nucleusPixCh02,5) # the fluo value of the 25th percetile
                     upperCh02 = np.percentile(nucleusPixCh02,95) # the fluo value of the 75th percetile
                     idx2 = (nucleusPixCh02>lowerCh02)*(nucleusPixCh02<upperCh02) #pixels between the 25th and 75th percentiles
                     fluoCh02= np.sum(nucleusPixCh02[idx2]) # average their fluorescence
                 else:
                     fluoCh02 = 0
           
             if any(nucleusPixCh03): #if a BFP-detected nucleus was detected in channel 3
                 wasDetectedGFP = True
                 lowerCh03 = np.percentile(nucleusPixCh03,5) # the fluo value of the 25th percetile
                 upperCh03 = np.percentile(nucleusPixCh03,95) # the fluo value of the 75th percetile
                 idx3 = (nucleusPixCh03>lowerCh03)*(nucleusPixCh03<upperCh03) #pixels between the 25th and 75th percentiles
                 fluoCh03= np.sum(nucleusPixCh03[idx3])
             else:
                 wasDetectedGFP = False
                 singleNucleusCh03 = GFPimage * singleNucleusMask2
                 nucleusPixCh03 = singleNucleusCh03[singleNucleusCh03>0]
                 if np.sum(nucleusPixCh03) > 0:
                     lowerCh03 = np.percentile(nucleusPixCh03,5) # the fluo value of the 25th percetile
                     upperCh03 = np.percentile(nucleusPixCh03,95) # the fluo value of the 75th percetile
                     idx3 = (nucleusPixCh03>lowerCh03)*(nucleusPixCh03<upperCh03) #pixels between the 25th and 75th percentiles
                     fluoCh03= np.sum(nucleusPixCh03[idx3]) # average their fluorescence
                 else:
                     fluoCh03 = 0
    
             data = [fileName,fluoCh03,fluoCh02,wasDetectedGFP,wasDetectedRFP]                  
             Nuclei_fluos.loc[len(Nuclei_fluos)+1, cols] = data
         
    #save
    outputPath = parentFolder + filesep + name + '_nuclei_fluos_int.csv'
    Nuclei_fluos.to_csv(outputPath)
         
         
#%% compare fluos
OD05plants = ['OD05_plant4_inf4','OD05_plant5_inf4','OD05_plant6_inf4']
OD2plants = ['OD2_plant4_inf4','OD2_plant5_inf4','OD2_plant6_inf4']

# OD05plants = ['OD05_plant4_inf2','OD05_plant5_inf2','OD05_plant6_inf2']
# OD2plants = ['OD2_plant4_inf2','OD2_plant5_inf2','OD2_plant6_inf2']

cols = ['filename','GFP_fluo','RFP_fluo']  
OD05AllNucleiFluos = pd.DataFrame([], columns=cols)
OD2AllNucleiFluos = pd.DataFrame([], columns=cols)


# open the nuclei_counts results of each of the experiments we're interested in
commonPath = parentFolder

for plant in OD05plants:
    nuclei_fluos_CSVname = commonPath + filesep + plant + '_nuclei_fluos.csv'
    nuclei_fluos = pd.read_csv(nuclei_fluos_CSVname)
    OD05AllNucleiFluos = pd.concat([OD05AllNucleiFluos,nuclei_fluos])

for plant in OD2plants:
    nuclei_fluos_CSVname = commonPath + filesep + plant + '_nuclei_fluos.csv'
    nuclei_fluos = pd.read_csv(nuclei_fluos_CSVname)
    OD2AllNucleiFluos = pd.concat([OD2AllNucleiFluos,nuclei_fluos])

OD05AllNucleiFluos['logGFP'] = np.log10(OD05AllNucleiFluos['GFP_fluo'])
OD05AllNucleiFluos['logRFP'] = np.log10(OD05AllNucleiFluos['RFP_fluo'])
OD2AllNucleiFluos['logGFP'] = np.log10(OD2AllNucleiFluos['GFP_fluo'])
OD2AllNucleiFluos['logRFP'] = np.log10(OD2AllNucleiFluos['RFP_fluo'])

#  look at the nuclei that were  detected
OD2detectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==True]
OD05detectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==True]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2detectedNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumorchid',element = 'step',fill=True)
sns.histplot(data=OD05detectedNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability', binwidth=0.1,color='mediumturquoise',element = 'step',fill=True,alpha=0.6)
#plt.yscale('log')
plt.legend(['OD 2','OD 05'])
plt.title('detected GFP nuclei')

# now look at the nuclei that were not detected
OD2undetectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==False]
OD05undetectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==False]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2undetectedNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumorchid',element = 'step',fill=True)
sns.histplot(data=OD05undetectedNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability', binwidth=0.1,color='mediumturquoise',element = 'step',fill=True,alpha=0.6)
#plt.yscale('log')
plt.legend(['OD 2','OD 05'])
plt.title('undetected GFP nuclei')

# now look at all nuclei
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2AllNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability',bins=30,color='mediumorchid',element = 'step',fill=True)
sns.histplot(data=OD05AllNucleiFluos,x='logGFP',
             log_scale = False, stat = 'probability', bins=30,color='mediumturquoise',element = 'step',fill=True,alpha=0.6)
#plt.yscale('log')
plt.legend(['OD 2','OD 05'])
plt.title('all GFP nuclei \n detected + undetected')
plt.yscale('log')


# now the same for RFP
OD2detectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedRFP']==True]
OD05detectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedRFP']==True]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2detectedNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumorchid',lw=0)
sns.histplot(data=OD05detectedNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumturquoise',lw=0,alpha=0.6)
plt.legend(['OD 2','OD 05'])
plt.title('detected RFP nuclei')

OD2undetectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedRFP']==False]
OD05undetectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedRFP']==False]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2undetectedNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumorchid',lw=0)
sns.histplot(data=OD05undetectedNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumturquoise',lw=0,alpha=0.6)
plt.legend(['OD 2','OD 05'])
plt.yscale('log')
plt.title('undetected RFP nuclei')

# now look at all nuclei
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.histplot(data=OD2AllNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumorchid',element = 'step',fill=True)
sns.histplot(data=OD05AllNucleiFluos,x='logRFP',
             log_scale = False, stat = 'probability',binwidth=0.1,color='mediumturquoise',element = 'step',fill=True,alpha=0.6)
#plt.yscale('log')
plt.legend(['OD 2','OD 05'])
plt.title('all RFP nuclei \n detected + undetected')
plt.yscale('log')



#%% violinplots GFP
OD2detectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==True]
OD05detectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==True]
OD2undetectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==False]
OD05undetectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==False]
OD2detectedNucleiFluos['OD'] = 2
OD2undetectedNucleiFluos['OD'] = 2
OD05detectedNucleiFluos['OD'] = 0.5
OD05undetectedNucleiFluos['OD'] = 0.5
detectedNuclei = pd.concat([OD2detectedNucleiFluos,OD05detectedNucleiFluos])
undetectedNuclei = pd.concat([OD2undetectedNucleiFluos,OD05undetectedNucleiFluos])
allNucleiGFP = pd.concat([detectedNuclei,undetectedNuclei])


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
#sns.swarmplot(data = detectedNuclei, x='OD',y='logGFP',size=2,color='green')
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('detected nuclei in GFP')
# Change major ticks to show every 20.
ax.yaxis.set_major_locator(MultipleLocator(0.25))
# Change minor ticks to show every 5. (20/4 = 5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
sns.violinplot(data = detectedNuclei, x='OD',y='logGFP',color='limegreen',alpha=0.3)
plt.ylim(2.5,4.5)


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='green')
sns.violinplot(data = undetectedNuclei, x='OD',y='logGFP',color='gray',alpha=0.3)
# Change major ticks to show every 20.
ax.yaxis.set_major_locator(MultipleLocator(0.25))
# Change minor ticks to show every 5. (20/4 = 5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('undetected nuclei in GFP')
plt.ylim(2.5,4.5)



fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
#sns.swarmplot(data = detectedNuclei, x='OD',y='logGFP',size=2,color='green')
sns.violinplot(data = allNucleiGFP, x='OD',y='logGFP',hue = 'was_detectedGFP',common_norm=False)
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('detected nuclei in GFP')

#%% violinplots RFP
OD2detectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedRFP']==True]
OD05detectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedRFP']==True]
OD2undetectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedRFP']==False]
OD05undetectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedRFP']==False]
OD2detectedNucleiFluos['OD'] = 2
OD2undetectedNucleiFluos['OD'] = 2
OD05detectedNucleiFluos['OD'] = 0.5
OD05undetectedNucleiFluos['OD'] = 0.5
detectedNuclei = pd.concat([OD2detectedNucleiFluos,OD05detectedNucleiFluos])
undetectedNuclei = pd.concat([OD2undetectedNucleiFluos,OD05undetectedNucleiFluos])
allNucleiGFP = pd.concat([detectedNuclei,undetectedNuclei])


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
#sns.swarmplot(data = detectedNuclei, x='OD',y='logGFP',size=2,color='green')
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('detected nuclei in RFP')
# Change major ticks to show every 20.
ax.yaxis.set_major_locator(MultipleLocator(0.5))
# Change minor ticks to show every 5. (20/4 = 5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
sns.violinplot(data = detectedNuclei, x='OD',y='logRFP',color='orchid',alpha=0.3)
plt.ylim(1.75,5)


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='green')
sns.violinplot(data = undetectedNuclei, x='OD',y='logRFP',color='gray',alpha=0.3)
# Change major ticks to show every 20.
ax.yaxis.set_major_locator(MultipleLocator(0.25))
# Change minor ticks to show every 5. (20/4 = 5)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('undetected nuclei in RFP')
plt.ylim(1.75,5)



fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
#sns.swarmplot(data = detectedNuclei, x='OD',y='logGFP',size=2,color='green')
sns.violinplot(data = allNucleiGFP, x='OD',y='logGFP',hue = 'was_detectedRFP',common_norm=False)
#sns.swarmplot(data = undetectedNuclei, x='OD',y='logGFP',size=2,color='blue',alpha=0.3)
plt.title('detected nuclei in RFP')



#%% SCATTERPLOTS


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(OD05AllNucleiFluos['GFP_fluo'],OD05AllNucleiFluos['RFP_fluo'],'bo',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD2AllNucleiFluos['GFP_fluo'],OD2AllNucleiFluos['RFP_fluo'],'go',markersize=4,alpha=0.2,markeredgecolor='None')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500,16000)
plt.ylim(100,30000)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(OD05AllNucleiFluos['GFP_fluo'],OD05AllNucleiFluos['RFP_fluo'],'bo',markersize=4,alpha=0.2,markeredgecolor='None')
#plt.plot(OD2AllNucleiFluos['GFP_fluo'],OD2AllNucleiFluos['RFP_fluo'],'o',markersize=4,alpha=0.2,markeredgecolor='None')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500,16000)
plt.ylim(100,30000)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
#plt.plot(OD05AllNucleiFluos['GFP_fluo'],OD05AllNucleiFluos['RFP_fluo'],'o',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD2AllNucleiFluos['GFP_fluo'],OD2AllNucleiFluos['RFP_fluo'],'go',markersize=4,alpha=0.2,markeredgecolor='None')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500,16000)
plt.ylim(100,30000)

OD05detectedBoth = OD05AllNucleiFluos[(OD05AllNucleiFluos['was_detectedGFP']==True) & (OD05AllNucleiFluos['was_detectedRFP']==True)]
OD05undetectedNucleiFluos = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==False]
OD05undetectedNucleiFluos = OD05undetectedNucleiFluos[OD05undetectedNucleiFluos['was_detectedRFP']==False]
OD05detectedNucleiFluosR = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedRFP']==True]
OD05detectedNucleiFluosR = OD05detectedNucleiFluosR[OD05detectedNucleiFluosR['was_detectedGFP']==False]
OD05detectedNucleiFluosG = OD05AllNucleiFluos[OD05AllNucleiFluos['was_detectedGFP']==True]
OD05detectedNucleiFluosG = OD05detectedNucleiFluosG[OD05detectedNucleiFluosG['was_detectedRFP']==False]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(OD05undetectedNucleiFluos['GFP_fluo'],OD05undetectedNucleiFluos['RFP_fluo'],'ko',markersize=4,alpha=0.15,markeredgecolor='None')
plt.plot(OD05detectedNucleiFluosR['GFP_fluo'],OD05detectedNucleiFluosR['RFP_fluo'],'o',color='orchid',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD05detectedNucleiFluosG['GFP_fluo'],OD05detectedNucleiFluosG['RFP_fluo'],'o',color='limegreen',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD05detectedBoth['GFP_fluo'],OD05detectedBoth['RFP_fluo'],'o',color='gold',markersize=4,alpha=0.3,markeredgecolor='None')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500,16000)
plt.ylim(100,30000)
plt.title('OD 0.5')

OD2detectedBoth = OD2AllNucleiFluos[(OD2AllNucleiFluos['was_detectedGFP']==True) & (OD2AllNucleiFluos['was_detectedRFP']==True)]
OD2undetectedNucleiFluos = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==False]
OD2undetectedNucleiFluos = OD2undetectedNucleiFluos[OD2undetectedNucleiFluos['was_detectedRFP']==False]
OD2detectedNucleiFluosR = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedRFP']==True]
OD2detectedNucleiFluosR = OD2detectedNucleiFluosR[OD2detectedNucleiFluosR['was_detectedGFP']==False]
OD2detectedNucleiFluosG = OD2AllNucleiFluos[OD2AllNucleiFluos['was_detectedGFP']==True]
OD2detectedNucleiFluosG = OD2detectedNucleiFluosG[OD2detectedNucleiFluosG['was_detectedRFP']==False]
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(OD2undetectedNucleiFluos['GFP_fluo'],OD2undetectedNucleiFluos['RFP_fluo'],'ko',markersize=4,alpha=0.15,markeredgecolor='None')
plt.plot(OD2detectedNucleiFluosR['GFP_fluo'],OD2detectedNucleiFluosR['RFP_fluo'],'o',color='orchid',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD2detectedNucleiFluosG['GFP_fluo'],OD2detectedNucleiFluosG['RFP_fluo'],'o',color='limegreen',markersize=4,alpha=0.2,markeredgecolor='None')
plt.plot(OD2detectedBoth['GFP_fluo'],OD2detectedBoth['RFP_fluo'],'o',color='gold',markersize=4,alpha=0.3,markeredgecolor='None')
plt.xscale('log')
plt.yscale('log')
plt.xlim(500,16000)
plt.ylim(100,30000)
plt.title('OD 2')



#%%

Nuclei_fluos['detected_both'] = Nuclei_fluos['was_detectedRFP']&Nuclei_fluos['was_detectedGFP']


plt.plot(Nuclei_fluos['GFP_fluo'],Nuclei_fluos['RFP_fluo'],'o',alpha=0.4)
plt.yscale('log')
plt.xscale('log')

#%% cooperation experiment of 1/22/24
parentFolder = '/Volumes/JSALAMOS/LeicaDM6B/1222024'
nucleiCountFiles = find_filenames(parentFolder, suffix=".csv" )
nucleiCountFiles = [i for i in nucleiCountFiles if '._' not in i] #remove weird files

GFPFluos = []
RFPFluos = []

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    Fluos_df = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    Fluos_df['normGFP'] = Fluos_df['GFP_fluo']/np.mean(Fluos_df['GFP_fluo'])
    Fluos_df['normRFP'] = Fluos_df['RFP_fluo']/np.mean(Fluos_df['RFP_fluo'])
    Fluos_df = Fluos_df.reset_index()

    isControl = Fluos_df['filename'][0].find('control')
    if isControl==0: #if it's a control experiment
        plt.plot(Fluos_df['GFP_fluo'],Fluos_df['RFP_fluo'],'bo',alpha=0.2,mec='None',ms=2)
    else: #if it's a cooperation experiment
        plt.plot(Fluos_df['GFP_fluo'],Fluos_df['RFP_fluo'],'ro',alpha=0.3,mec='None',ms=2)    
    # concatenate
    GFPFluos = np.append(GFPFluos, Fluos_df['GFP_fluo'])
    RFPFluos = np.append(GFPFluos, Fluos_df['RFP_fluo'])

plt.xlabel('GFP')
plt.ylabel('RFP')
plt.xscale('log')
plt.yscale('log')




#%%
RFPfluosRFPonlyCtrl = [] #nuclei detected in RFP but not GFP in the control
GFPfluosGFPonlyCtrl = [] #nuclei detected in GFP but not RFP in the control
RFPfluosRFPcontransCtrl = [] #nuclei detected in RFP and RFP in the control
GFPfluosGFPcontransCtrl = [] #nuclei detected in GFP and RFP in the control
RFPfluosRFPonlyCoop = [] #nuclei detected in RFP but not GFP in he cooperation exp
GFPfluosGFPonlyCoop = [] #nuclei detected in GFP but not RFP in he cooperation exp
RFPfluosRFPcontransCoop = [] # nuclei detected in RFP and RFP in the cooperation exp
GFPfluosGFPcontransCoop = [] # nuclei detected in GFP and RFP in the cooperation exp


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
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
        plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'bo',alpha=0.2,mec='None',ms=2)
        # concatenate
        RFPfluosRFPcontransCtrl = np.append(RFPfluosRFPcontransCtrl, Fluos_df1['normRFP'])
    else: #if it's a cooperation experiment
        plt.plot(Fluos_df1['GFP_fluo'],Fluos_df1['RFP_fluo'],'ro',alpha=0.3,mec='None',ms=2) 
        # concatenate
        RFPfluosRFPcontransCoop = np.append(RFPfluosRFPcontransCoop, Fluos_df1['normRFP'])
    
    # Now keep only nuclei that were detected in RFP but NOT in GFP
    Fluos_df2 = Fluos_df[(Fluos_df['was_detectedGFP']==False)&(Fluos_df['was_detectedRFP']==True)]
    # get their RFP fluorescence
    Fluos_df2['normGFP'] = Fluos_df2['GFP_fluo']/np.mean(Fluos_df2['GFP_fluo'])
    Fluos_df2['normRFP'] = Fluos_df2['RFP_fluo']/np.mean(Fluos_df2['RFP_fluo'])
    Fluos_df2 = Fluos_df2.reset_index()
    # plot
    isControl = Fluos_df['filename'][0].find('control')
    if isControl==0: #if it's a control experiment
        plt.plot(Fluos_df2['GFP_fluo'],Fluos_df2['RFP_fluo'],'ko',alpha=0.15,mec='None',ms=2)
        # concatenate
        RFPfluosRFPonlyCtrl = np.append(RFPfluosRFPonlyCtrl, Fluos_df2['normRFP'])
    else: #if it's a cooperation experiment
        plt.plot(Fluos_df2['GFP_fluo'],Fluos_df2['RFP_fluo'],'yo',alpha=0.3,mec='None',ms=2)   
        # concatenate
        RFPfluosRFPonlyCoop = np.append(RFPfluosRFPonlyCoop, Fluos_df2['normRFP'])

plt.xlabel('GFP')
plt.ylabel('RFP')
plt.xscale('log')
plt.yscale('log')


#%% #%% cooperation experiment of 10/23/2024

parentFolder = '/Volumes/JSALAMOS/LeicaDM6B/10232024'
nucleiCountFiles = find_filenames(parentFolder, suffix=".csv" )
nucleiCountFiles = [i for i in nucleiCountFiles if '._' not in i] #remove weird files
nucleiCountFiles_wt = [i for i in nucleiCountFiles if 'inf3' in i]
nucleiCountFiles_vire12 = [i for i in nucleiCountFiles if 'inf4' in i]

# initialize dataframes
#wt_fluos = pd.DataFrame([], columns=['filename', 'GFP_fluo','RFP_fluo','was_detectedGFP','was_detectedRFP'])
wt_df_list = []
#vire12_fluos = pd.DataFrame([], columns=['filename', 'GFP_fluo','RFP_fluo','was_detectedGFP','was_detectedRFP'])
vire12_df_list = []

for fileName in nucleiCountFiles_wt:
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    wt_df_list.append(Fluos_df)
wt_fluos = pd.concat(wt_df_list)
wt_fluos = wt_fluos.rename(columns={"GFP_fluo": "RFP", "RFP_fluo": "GFP"})
wt_fluos = wt_fluos.rename(columns={"was_detectedRFP": "was_detectedGFP", "was_detectedGFP": "was_detectedRFP"})

wt_fluos['logGFP'] = np.log(wt_fluos['GFP'])
wt_fluos['logRFP'] = np.log(wt_fluos['RFP'])
wt_fluos.loc[(wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP']),'class'] = 'both'
wt_fluos.loc[(wt_fluos['was_detectedRFP'] & ~wt_fluos['was_detectedGFP']),'class'] = 'RFP_only'
wt_fluos.loc[(~wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP']),'class'] = 'GFP_only'
wt_fluos.loc[(~wt_fluos['was_detectedRFP'] & ~wt_fluos['was_detectedGFP']),'class'] = 'neither'

wt_fluos.dropna(subset=['class'], inplace=True)
wt_fluos['experiment'] = 'wt'


for fileName in nucleiCountFiles_vire12:
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    vire12_df_list.append(Fluos_df)
vire12_fluos = pd.concat(vire12_df_list)
vire12_fluos = vire12_fluos.rename(columns={"GFP_fluo": "RFP", "RFP_fluo": "GFP"})
vire12_fluos = vire12_fluos.rename(columns={"was_detectedRFP": "was_detectedGFP", "was_detectedGFP": "was_detectedRFP"})

vire12_fluos['logGFP'] = np.log(vire12_fluos['GFP'])
vire12_fluos['logRFP'] = np.log(vire12_fluos['RFP'])
vire12_fluos.loc[(vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP']),'class'] = 'both'
vire12_fluos.loc[(vire12_fluos['was_detectedRFP'] & ~vire12_fluos['was_detectedGFP']),'class'] = 'RFP_only'
vire12_fluos.loc[(~vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP']),'class'] = 'GFP_only'
vire12_fluos.loc[(~vire12_fluos['was_detectedRFP'] & ~vire12_fluos['was_detectedGFP']),'class'] = 'neither'
vire12_fluos.dropna(subset=['class'], inplace=True)
vire12_fluos['experiment'] = 'vire12'




alldata = pd.concat([wt_fluos, vire12_fluos])
# Replace -inf with NaN
alldata.replace(-np.inf, np.nan, inplace=True)
# Drop rows with NaN values
alldata.dropna(inplace=True)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.violinplot(data = alldata, x = 'class', y = 'logRFP', hue = 'experiment')
plt.xticks(rotation='vertical')

#%%

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.scatterplot(data=wt_fluos, x='logGFP', y='logRFP',s=2)
# plt.xscale('log')
# plt.yscale('log')

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.scatterplot(data=vire12_fluos, x='logGFP', y='logRFP',s=2)
# plt.xscale('log')
# plt.yscale('log')


fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.scatterplot(data=wt_fluos[wt_fluos['was_detectedRFP']], x='logGFP', y='logRFP',s=2,alpha=0.1)
sns.scatterplot(data=wt_fluos[(wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP'])], x='logGFP', y='logRFP',s=2,alpha=0.1)

# plt.xscale('log')
# plt.yscale('log')

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
sns.scatterplot(data=vire12_fluos[vire12_fluos['was_detectedRFP']], x='logGFP', y='logRFP',s=2,alpha=0.1)
sns.scatterplot(data=vire12_fluos[(vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP'])], x='logGFP', y='logRFP',s=2,alpha=0.1)
# plt.xscale('log')
# plt.yscale('log')



a = np.mean(wt_fluos[(wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP'])]['RFP'])
b = np.mean(wt_fluos[(wt_fluos['was_detectedRFP'] & ~wt_fluos['was_detectedGFP'])]['RFP'])
print('RFP in nuclei that express wt GFP = ' + str(a))
print("RFP in nuclei that don't express wt GFP = " + str(b))

c = np.mean(vire12_fluos[(vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP'])]['RFP'])
d = np.mean(vire12_fluos[(vire12_fluos['was_detectedRFP'] & ~vire12_fluos['was_detectedGFP'])]['RFP'])
print('RFP in nuclei that express vire12 GFP = ' + str(c))
print("RFP in nuclei that don't express vire12 GFP = " + str(d))


GFP_in_GFP_only = wt_fluos[(~wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP'])]['GFP']
GFP_in_both = wt_fluos[(wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP'])]['GFP']
RFP_in_RFP_only = wt_fluos[(wt_fluos['was_detectedRFP'] & ~wt_fluos['was_detectedGFP'])]['RFP']
RFP_in_both = wt_fluos[(wt_fluos['was_detectedRFP'] & wt_fluos['was_detectedGFP'])]['RFP']

GFP_in_GFP_only_vir = vire12_fluos[(~vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP'])]['GFP']
GFP_in_both_vir = vire12_fluos[(vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP'])]['GFP']
RFP_in_RFP_only_vir = vire12_fluos[(vire12_fluos['was_detectedRFP'] & ~vire12_fluos['was_detectedGFP'])]['RFP']
RFP_in_both_vir = vire12_fluos[(vire12_fluos['was_detectedRFP'] & vire12_fluos['was_detectedGFP'])]['RFP']

#%%

GFPFluos = []
RFPFluos = []

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
for fileName in nucleiCountFiles[2:]:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    Fluos_df = Fluos_df[(Fluos_df['was_detectedGFP']==True)&(Fluos_df['was_detectedRFP']==True)]
    Fluos_df['normGFP'] = Fluos_df['GFP_fluo']/np.mean(Fluos_df['GFP_fluo'])
    Fluos_df['normRFP'] = Fluos_df['RFP_fluo']/np.mean(Fluos_df['RFP_fluo'])
    Fluos_df = Fluos_df.reset_index()

    isControl = Fluos_df['filename'][0].find('inf3')
    if isControl==0: #if it's a control experiment
        plt.plot(Fluos_df['GFP_fluo'],Fluos_df['RFP_fluo'],'bo',alpha=0.2,mec='None',ms=2)
    else: #if it's a cooperation experiment
        plt.plot(Fluos_df['GFP_fluo'],Fluos_df['RFP_fluo'],'ro',alpha=0.3,mec='None',ms=2)    
    # concatenate
    GFPFluos = np.append(GFPFluos, Fluos_df['GFP_fluo'])
    RFPFluos = np.append(GFPFluos, Fluos_df['RFP_fluo'])

plt.xlabel('GFP')
plt.ylabel('RFP')
plt.xscale('log')
plt.yscale('log')











#%%

plt.plot(RFPfluosRFPonlyCoop)













fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.hist(RFPfluosRFPcontransCtrl,alpha=0.2,density=True,color='b')
plt.hist(RFPfluosRFPcontransCoop,alpha=0.2,density=True,color='r')
plt.hist(RFPfluosRFPonlyCtrl,alpha=0.2,density=True,color='k')
plt.hist(RFPfluosRFPonlyCoop,alpha=0.2,density=True,color='y')
plt.xscale('log')
plt.yscale('log')











RFPOnlyFluos_ctrl = []
RFPOnlyFluos_ctrlMeans = []
RFPContransfFluos_ctrl = []
RFPContransfFluos_ctrlMeans = []
RFPOnlyFluos_coop = []
RFPOnlyFluos_coopMeans = []
RFPContransfFluos_coop = []
RFPContransfFluos_coopMeans = []

for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    Fluos_df['normRFP'] = Fluos_df['RFP_fluo']/np.mean(Fluos_df['RFP_fluo'])
    # keep only nuclei that express RFP
    Fluos_df = Fluos_df[Fluos_df['was_detectedRFP']==True]

    Fluos_df_detectedInGFP = Fluos_df[Fluos_df['was_detectedGFP']==True]
    Fluos_df_NotDetectedInGFP = Fluos_df[Fluos_df['was_detectedGFP']==False]

    Fluos_df = Fluos_df.reset_index()
    isControl = Fluos_df['filename'][0].find('control')
    
    metric = 'RFP_fluo'
    if isControl==0: #if it's a control experiment
        RFPOnlyFluos_ctrl = np.append(RFPOnlyFluos_ctrl, Fluos_df_NotDetectedInGFP[metric])
        RFPOnlyFluos_ctrlMeans = np.append(RFPOnlyFluos_ctrlMeans,np.nanmean(Fluos_df_NotDetectedInGFP[metric]))
        RFPContransfFluos_ctrl = np.append(RFPContransfFluos_ctrl, Fluos_df_detectedInGFP[metric])
        RFPContransfFluos_ctrlMeans = np.append(RFPContransfFluos_ctrlMeans,np.nanmean(Fluos_df_detectedInGFP[metric]))
    else: #if it's a cooperation experiment
        RFPOnlyFluos_coop = np.append(RFPOnlyFluos_coop, Fluos_df_NotDetectedInGFP[metric])
        RFPOnlyFluos_coopMeans = np.append(RFPOnlyFluos_coopMeans,np.mean(Fluos_df_NotDetectedInGFP[metric]))
        RFPContransfFluos_coop = np.append(RFPContransfFluos_coop, Fluos_df_detectedInGFP[metric])
        RFPContransfFluos_coopMeans = np.append(RFPContransfFluos_coopMeans,np.mean(Fluos_df_detectedInGFP[metric]))

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(RFPOnlyFluos_ctrlMeans,RFPContransfFluos_ctrlMeans,'bo')
plt.plot(RFPOnlyFluos_coopMeans,RFPContransfFluos_coopMeans,'ro')
plt.xlabel('RFP fluo in cells \n that only express RFP')
plt.ylabel('RFP fluo in cells \n that also express GFP')
plt.legend(['control','cooperation'])

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
foldChangeRFPFluos_coop = RFPContransfFluos_coopMeans/RFPOnlyFluos_coopMeans
foldChangeRFPFluos_comp = RFPContransfFluos_ctrlMeans/RFPOnlyFluos_ctrlMeans
plt.plot(np.ones(len(foldChangeRFPFluos_coop)),foldChangeRFPFluos_coop,'bo')
plt.plot(2*np.ones(len(foldChangeRFPFluos_comp)),foldChangeRFPFluos_comp,'ro')
plt.boxplot([foldChangeRFPFluos_coop,foldChangeRFPFluos_comp])
plt.ylabel('fold change in RFP fluorescence \n nuclei that express both/nuclei with RFP only')
plt.legend(['cooperation', 'control'])




GFPOnlyFluos_ctrl = []
GFPOnlyFluos_ctrlMeans = []
GFPContransfFluos_ctrl = []
GFPContransfFluos_ctrlMeans = []
GFPOnlyFluos_coop = []
GFPOnlyFluos_coopMeans = []
GFPContransfFluos_coop = []
GFPContransfFluos_coopMeans = []

for fileName in nucleiCountFiles:
    # read the nuclei fluos from this image
    Fluos_df = pd.read_csv(parentFolder + filesep + fileName)
    Fluos_df['normGFP'] = Fluos_df['GFP_fluo']/np.mean(Fluos_df['GFP_fluo'])
    # keep only nuclei that express GFP
    Fluos_df = Fluos_df[Fluos_df['was_detectedGFP']==True]

    Fluos_df_detectedInRFP = Fluos_df[Fluos_df['was_detectedRFP']==True]
    Fluos_df_NotDetectedInRFP = Fluos_df[Fluos_df['was_detectedRFP']==False]

    Fluos_df = Fluos_df.reset_index()
    isControl = Fluos_df['filename'][0].find('control')
    
    metric = 'normGFP'
    if isControl==0: #if it's a control experiment
        GFPOnlyFluos_ctrl = np.append(GFPOnlyFluos_ctrl, Fluos_df_NotDetectedInRFP[metric])
        GFPOnlyFluos_ctrlMeans = np.append(GFPOnlyFluos_ctrlMeans,np.nanmean(Fluos_df_NotDetectedInRFP[metric]))
        GFPContransfFluos_ctrl = np.append(GFPContransfFluos_ctrl, Fluos_df_detectedInRFP[metric])
        GFPContransfFluos_ctrlMeans = np.append(GFPContransfFluos_ctrlMeans,np.nanmean(Fluos_df_detectedInRFP[metric]))
    else: #if it's a cooperation experiment
        GFPOnlyFluos_coop = np.append(GFPOnlyFluos_coop, Fluos_df_NotDetectedInRFP[metric])
        GFPOnlyFluos_coopMeans = np.append(GFPOnlyFluos_coopMeans,np.mean(Fluos_df_NotDetectedInRFP[metric]))
        GFPContransfFluos_coop = np.append(GFPContransfFluos_coop, Fluos_df_detectedInRFP[metric])
        GFPContransfFluos_coopMeans = np.append(GFPContransfFluos_coopMeans,np.mean(Fluos_df_detectedInRFP[metric]))

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.plot(GFPOnlyFluos_ctrlMeans,GFPContransfFluos_ctrlMeans,'bo')
plt.plot(GFPOnlyFluos_coopMeans,GFPContransfFluos_coopMeans,'ro')
plt.xlabel('GFP fluo in cells \n that only express GFP')
plt.ylabel('GFP fluo in cells \n that also express RFP')
plt.legend(['control','cooperation'])

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
foldChangeGFPFluos_coop = GFPContransfFluos_coopMeans/GFPOnlyFluos_coopMeans
foldChangeGFPFluos_comp = GFPContransfFluos_ctrlMeans/GFPOnlyFluos_ctrlMeans
plt.plot(np.ones(len(foldChangeGFPFluos_coop)),foldChangeGFPFluos_coop,'bo')
plt.plot(2*np.ones(len(foldChangeGFPFluos_comp)),foldChangeGFPFluos_comp,'ro')
plt.boxplot([foldChangeGFPFluos_coop,foldChangeGFPFluos_comp])
plt.ylabel('fold change in GFP fluorescence \n nuclei that express both/nuclei with GFP only')
plt.legend(['cooperation', 'control'])





















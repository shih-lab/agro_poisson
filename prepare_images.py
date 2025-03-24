#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:36:10 2023

@author: simon_alamos
"""
#%% import packages
import matplotlib
import matplotlib.pyplot as plt
from readlif.reader import LifFile # to read leica LIF files
import numpy as np
import os
filesep = os.sep # this is the folder separator, it can change depending on the operating system
import glob # to list all the files of some kind within a folder 
import random

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

from scipy.stats import sem
from scipy import ndimage
from scipy.spatial.distance import cdist

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import tifffile

from pathlib import Path
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



#[areaFiltMaskBFP,labels_ws2BFP,NBFP,[],[]] = AreaFilter(BFPMask,minArea,maxArea)


def AreaFilter(imMask,minArea,maxArea,showImages):
    # min and max Area is in pixels squared
    # first label each nucleus with an ID
    distance = ndimage.distance_transform_edt(imMask)
    local_maxi = peak_local_max(
    distance, indices=False, footprint=np.ones((10, 10)), labels=imMask)
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


def makeImMask(filename,Thresholds,sigma):
    
    Im = np.load(filename)
    
    # load the previously generated Otsu threshold of this image
    threshVal = Thresholds.loc[Thresholds['filename']==filename,'threshold'].values[0]
    threshVal = threshVal/1.5 # make it a bit more relaxed
    
    # create DOG filtered image and then segment using Otsu threshold
    DOGim = skimage.filters.difference_of_gaussians(Im,sigma)
    # show
    fig = plt.figure()
    fig.set_size_inches(4.5, 4.5)
    plt.imshow(DOGim,cmap='cubehelix',vmax=np.max(DOGim)/5)
    plt.title('DOG-filtered nuclear label channel')
    plt.show() 
    # apply Otsu threshold to DOG filtered image
    imMask = DOGim > threshVal
    #show
    fig = plt.figure()
    fig.set_size_inches(4.5, 4.5)
    plt.imshow(imMask,cmap='cubehelix',vmax=np.max(imMask)/5)
    plt.title('DOG-filtered nuclear label channel')
    plt.show() 
    
    # apply watershed algorithm to separate joined nuclei
    distance = ndimage.distance_transform_edt(imMask)
    local_maxi = peak_local_max(
    distance, indices=False, footprint=np.ones((10, 10)), labels=imMask)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=imMask)
    # label each nucleus
    num_cells = np.max(labels_ws)                                   
    # Print out how many we identified.
    # print("Number of objects found: %s" %num_cells)
    new_cmap = rand_cmap(num_cells, type='bright', first_color_black=True, last_color_black=False, verbose=True)
       
    return [Im, imMask,labels_ws]


def getPositions(labels_ws2,num_cells2):
    props = skimage.measure.regionprops(labels_ws2,labels_ws2)
    rowPos = np.zeros(num_cells2) #to store position values
    colPos = np.zeros(num_cells2)   
    # Loop through each object and extract the properties. 
    for i in range(len(props)):
        # Extract the positions
        rowPos[i] = props[i].centroid[0]
        colPos[i] = props[i].centroid[1]
    return [rowPos,colPos]


def countOverlaps(Y1,X1,Y2,X2,distanceThreshold):
    counts = 0
    for i, x1 in enumerate(X1):
        y1 = Y1[i]
        for j, x2 in enumerate(X2):
            y2 = Y2[j]
            #print(x1,y1,x2,y2)
            distance = np.sqrt(np.power(x2-x1,2)+np.power(y2-y1,2))
            if distance <= distanceThreshold:
                counts +=1
    return [counts]



def calculateNucleiFluos(labeledMask,intensityImage,numCells):
# tlabeledMask and numCells are generated by the AreaFilter function
# intensityImage is a raw image such as a slice or a max projection
    MeanFluos = np.zeros(numCells) # to store values
    IntFluos = np.zeros(numCells) # to store values
    for i in range(numCells): # Loop through each nucleus. Remember that we have to start indexing at 1 in this case!
        #print(str(100*np.round(i/num_cells2,3))+ ' % of nuclei processed') # for keeping track of progress
        # select the binary mask corresponding to this nucleus   
        singleNucleusMask = labeledMask == i + 1   
        # apply the mask to each channel by multiplying the binary mask by the max projected image
        singleNucleus = intensityImage * singleNucleusMask # an image of zeros everywhere and intensity values for a single nucleus
        # select only the pixels corresponding to the mask in each channel, not the zero pixels
        nucleusPix = singleNucleus[singleNucleus>0]
        bottom = np.percentile(nucleusPix,10) # the fluo value of the 10th percetile
        top = np.percentile(nucleusPix,90) # the fluo value of the 90th percetile
        MeanFluos[i] = np.mean(nucleusPix[np.where((nucleusPix >= bottom) & (nucleusPix <=top))]) # integrate the fluo of the middle 70% pixels
        IntFluos[i] = np.sum(nucleusPix[np.where((nucleusPix >= bottom) & (nucleusPix <=top))]) # integrate the fluo of the middle 70% pixels
    # calculate averages and standard deviations across nuclei in this image
    meanAvgFluo = np.mean(MeanFluos)
    sdAvgFluo = np.std(MeanFluos)
    meanIntFluo = np.mean(IntFluos)
    sdIntFluo = np.std(IntFluos)
    
    return [meanAvgFluo,sdAvgFluo,meanIntFluo,sdIntFluo]
        
#%% try first for a single image: LIF file from a Leica microscope
# load  file
# if LIF
new = LifFile('/Volumes/JSALAMOS/LeicaDM6B/08282023/OD01_plant3_inf7.lif')


# Access a specific image directly
img_0 = new.get_image(0)
# Create a list of images using a generator
img_list = [i for i in new.get_iter_image()]


# Access a specific item
img_0.get_frame(z=0, t=0, c=0)
# Iterate over different items
frame_list   = [i for i in img_0.get_iter_t(c=0, z=0)]
z_list       = [i for i in img_0.get_iter_z(t=0, c=0)]
channel_list = [i for i in img_0.get_iter_c(t=0, z=0)]

# grab all z slices of a single channel
z_list_Ch01 = [i for i in img_0.get_iter_z(t=0, c=0)]
z_list_Ch02 = [i for i in img_0.get_iter_z(t=0, c=1)]
z_list_Ch03 = [i for i in img_0.get_iter_z(t=0, c=2)]

# take a look at the first slice

BFPchannel = z_list_Ch01[0]
GFPchannel = z_list_Ch02[0]
RFPchannel = z_list_Ch03[0]

plt.subplot(311)
plt.imshow(BFPchannel,'Blues',vmax=np.max(BFPchannel)/5)
plt.subplot(312)
plt.imshow(GFPchannel,'Greens',vmax=np.max(GFPchannel)/5)
plt.subplot(313)
plt.imshow(RFPchannel,'Reds',vmax=np.max(RFPchannel)/5)

# max project and then take a look

maxCh01 = np.max(np.stack(z_list_Ch01),0)
maxCh02 = np.max(np.stack(z_list_Ch02),0)
maxCh03 = np.max(np.stack(z_list_Ch03),0)

plt.subplot(311)
plt.imshow(maxCh01,'Blues',vmax=np.max(maxCh01)/5)
plt.subplot(312)
plt.imshow(maxCh02,'Greens',vmax=np.max(maxCh02)/5)
plt.subplot(313)
plt.imshow(maxCh03,'Reds',vmax=np.max(maxCh03)/5)

# save as numpy arrays
outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/09192023/Max_projections'
prefix = 'OD01_plant1_inf1'
fileName = outputFolder + filesep + prefix + 'Ch01.npy'
np.save(fileName ,maxCh01)
# now load to confirm
A = np.load(fileName)
plt.imshow(A,'Blues',vmin=np.min(A)*5,vmax=np.max(A)/5)



#%% Try first for a single image: LSM files from a Zeiss microscope

# load the image
tif = tifffile.imread('/Volumes/JSALAMOS/lsm710/2023/9-19-23/OD05_plant6_inf7.lsm')
tif.shape # it's 5-dimensional: dummy, z, c, x, y

# max project
zdim = 1 # this is the position of the z dimension, it can be different from different scopes!
maxProj = tif.max(zdim) #z-project along dimension '1' which is the second one, z
print(maxProj.shape)

# get rid of the simpleton dimension
if 1 in maxProj.shape: #check if there's a singleton dimension that we'll get rid of
    singletonIdx = maxProj.shape.index(1) #find the position of the singleton dimension
    zproj = maxProj.max(singletonIdx) #this removes the singleton dimension.
    print(zproj.shape) #should be c,x,y
    
# look at the max projected images
maxCh01 = zproj[0, :, :]
maxCh02 = zproj[1, :, :]
maxCh03 = zproj[2, :, :]

plt.figure()
plt.subplot(311)
plt.imshow(maxCh01,'Blues',vmax=np.max(BFPchannel)/3)
plt.subplot(312)
plt.imshow(maxCh02,'Greens',vmax=np.max(GFPchannel)/7)
plt.subplot(313)
plt.imshow(maxCh03,'Reds',vmax=np.max(RFPchannel)/7)

# save as numpy arrays
outputFolder = '/Volumes/JSALAMOS/lsm710/2023/9-19-23/Max_projections'
prefix = 'OD05_plant6_inf7'
fileName = outputFolder + filesep + prefix + 'Ch01.npy'
np.save(fileName ,maxCh01)
# now load to confirm
A = np.load(fileName)
plt.figure()
plt.imshow(A,'Blues',vmin=np.min(A)*5,vmax=np.max(A)/2)


#%% now generate the max projections this in batch: LIF
date = '12112023'
#new = LifFile('/Volumes/JSALAMOS/LeicaDM6B/08282023/OD01_plant1_inf1.lif')
outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections'

isExist = os.path.exists(outputFolder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(outputFolder)
   print("The new output directory was created!")


filePathList = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/" + date + "/*.lif")
for filepath in filePathList:
    
    LIF = LifFile(filepath) # load the LIF file
    
    # the prefix identifying the image is whatever is between '.lif' and the last file separator
    prefixStart = filepath.rfind(filesep)
    prefixEnd = filepath.find('.lif')
    prefix = filepath[prefixStart+1:prefixEnd]
    
    # since each file has a single stack, we can just access the 'first' image
    img_0 = LIF.get_image(0)    
    
    # grab all z slices of each  channel. 0 is channel 1 = BFP, 1 is channel 2 = GFP, 2 is channel 3 = RFP
    z_list_Ch01 = [i for i in img_0.get_iter_z(t=0, c=0)]
    z_list_Ch02 = [i for i in img_0.get_iter_z(t=0, c=1)]
    z_list_Ch03 = [i for i in img_0.get_iter_z(t=0, c=2)]
    
    # max project along z (the first dimension)
    zdim = 0
    maxCh01 = np.max(np.stack(z_list_Ch01),zdim)
    maxCh02 = np.max(np.stack(z_list_Ch02),zdim)
    maxCh03 = np.max(np.stack(z_list_Ch03),zdim)
    
    # save the max projections
    fileNameCh01 = outputFolder + filesep + prefix + '_Ch01.npy'
    np.save(fileNameCh01 ,maxCh01)
    fileNameCh02 = outputFolder + filesep + prefix + '_Ch02.npy'
    np.save(fileNameCh02 ,maxCh02)
    fileNameCh03 = outputFolder + filesep + prefix + '_Ch03.npy'
    np.save(fileNameCh03 ,maxCh03)
    
#%% now generate the max projections this in batch: LSM

#new = LifFile('/Volumes/JSALAMOS/LeicaDM6B/08282023/OD01_plant1_inf1.lif')
#outputFolder = '/Volumes/JSALAMOS/lsm710/2023/9-19-23/Max_projections'
date = "2023/11-20-23"
outputFolder = '/Volumes/JSALAMOS/lsm710/' + date + '/Max_projections'

isExist = os.path.exists(outputFolder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(outputFolder)
   print("The new output directory is created!")


filePathList = glob.glob("/Volumes/JSALAMOS/lsm710/" + date + "/*.lsm")

for filepath in filePathList:
    
    tif = tifffile.imread(filepath) # load the TIF file
    
    # the prefix identifying the image is whatever is between '.lsm' and the last file separator
    prefixStart = filepath.rfind(filesep)
    prefixEnd = filepath.find('.lsm')
    prefix = filepath[prefixStart+1:prefixEnd]
    
    # max project
    zdim = 1 # this is the position of the z dimension, it can be different from different scopes!
    maxProj = tif.max(zdim) #z-project along dimension '1' which is the second one, z
    print(maxProj.shape)
    
    # get rid of the simpleton dimension
    if 1 in maxProj.shape: #check if there's a singleton dimension that we'll get rid of
        singletonIdx = maxProj.shape.index(1) #find the position of the singleton dimension
        zproj = maxProj.max(singletonIdx) #this removes the singleton dimension.
    
    # the max projected images
    maxCh01 = zproj[0, :, :]
    maxCh02 = zproj[1, :, :]
    maxCh03 = zproj[2, :, :]
    
    # save the max projections
    fileNameCh01 = outputFolder + filesep + prefix + '_Ch01.npy'
    np.save(fileNameCh01 ,maxCh01)
    fileNameCh02 = outputFolder + filesep + prefix + '_Ch02.npy'
    np.save(fileNameCh02 ,maxCh02)
    fileNameCh03 = outputFolder + filesep + prefix + '_Ch03.npy'
    np.save(fileNameCh03 ,maxCh03)
#%% Find and save Otsu segmentation thresholds for the DOG files
# now I want to pick a segmentation threshold for each channel in each plant
# I want to do it on a plant-wise basis because there are differences in BFP fluo between plants

#plantNames = ['OD01_plant5','OD01_plant6','OD01_plant7','OD05_plant4','OD05_plant5','OD05_plant6','OD2_plant4','OD2_plant5','OD2_plant6']
#plantnames = ['OD01_plant5']
plantNames = ['comp_plant1','comp_plant2','comp_plant3','comp_plant4','comp_plant5','comp_plant6','comp_plant7',
              'OD1_plant4','OD1_plant5','OD1_plant6','OD1_plant7','OD1_plant8','OD1_plant9']


#plantNames = ['OD05_BiBi654-514_plant6']
channels = ['Ch01','Ch02','Ch03']
palettes = ['Blues','Greens','Reds']


# grab the file names of all the max projections
# date = "2023/11-20-23"
# filePathList = glob.glob("/Volumes/JSALAMOS/lsm710/" + date + "/Max_projections/*.npy")
# DOGfilesOutput = "/Volumes/JSALAMOS/lsm710/" + date + "/DOGs"
# MaskFilesOutput = "/Volumes/JSALAMOS/lsm710/" + date + "/Masks"
# outputFolder = "/Volumes/JSALAMOS/lsm710/" + date + "/Max_projections"

date = '12112023'
filePathList = glob.glob('/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/*.npy')
DOGfilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/DOGs'
MaskFilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Masks'
outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections'

# Check whether the specified path exists or not
isExist = os.path.exists(DOGfilesOutput)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(DOGfilesOutput)
   print("The new DOGs directory is created!")
   
isExist = os.path.exists(MaskFilesOutput)
if not isExist:
   os.makedirs(MaskFilesOutput)
   print("The new Masks directory is created!")

sigma = 5 # for DOG filtering, 5 has been good for Leica experiments, I think 5 works for Zeiss too

for plant_counter, plantName in enumerate(plantNames): #loop over plants
    print(plantName)
    # find the file path names corresponding to this plant
    thisPlantPathNames = [i for i in filePathList if plantName in i]  
    # create dataframe to store the thresholds and such
    thresholds = pd.DataFrame([], columns=['filename','channel', 'threshold'])
    thresholds['filename'] = thisPlantPathNames
        
    for Channel_counter, channel in enumerate(channels): #loop over channels
        palette = palettes[Channel_counter]
        thisPlantThisChannelNames = [i for i in thisPlantPathNames if channel in i]
        
        for image_Counter, filename in enumerate(thisPlantThisChannelNames): #loop over infiltrations
                
            # open the image 
            Im = np.load(filename)
            # show the image
            # plt.imshow(Im,palette,vmin=np.min(Im)*3,vmax=np.max(Im)/6)           
            # apply DOG filter and show result
            DOGim = skimage.filters.difference_of_gaussians(Im,sigma)            
            # threshold the nuclear mask using the Otsu agorithm 
            if channel == 'Ch01':
                OTSUthresh = skimage.filters.threshold_otsu(DOGim)
            else:
                OTSUthresh = skimage.filters.threshold_otsu(DOGim)/1.5
            # fig = plt.figure()
            # # threshold the nuclear mask using the Li agorithm 
            # LIthresh = skimage.filters.threshold_li(DOGim)
            # # show results side by side
            # fig.set_size_inches(4.5, 4.5)
            # plt.subplot(1,3,1)
            # plt.imshow(DOGim>OTSUthresh,cmap=palette)
            # plt.title('Otsu thresholding')
            # plt.subplot(1,3,2)
            # plt.imshow(DOGim,cmap=palette,vmin=np.min(DOGim),vmax=np.max(DOGim)/20)
            # plt.title('DOG')
            # plt.subplot(1,3,3)
            # plt.imshow(Im,palette,vmin=np.min(Im)*3,vmax=np.max(Im)/4)
            # plt.show()            
            # save threshold found into dataframe
            print(OTSUthresh)
            thresholds.loc[thresholds['filename']==filename,'threshold'] = OTSUthresh
            thresholds.loc[thresholds['filename']==filename,'channel'] = channel 
            
            fig = plt.figure()
            plt.imshow(DOGim>OTSUthresh,palette)
            plt.title('OTSU Thresholded DOG \n' + filename)
            plt.show()  
            
            # save DOG
            DOGfileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
            np.save(DOGfilesOutput + filesep + DOGfileName + '.npy' ,DOGim)
            
            #save Mask
            ImMask = DOGim>OTSUthresh
            MaskFileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
            np.save(MaskFilesOutput + filesep + MaskFileName + '.npy' ,ImMask)
            
        
        # save dataframe with thresholds to pc
        thresholdsDFname = plantName + '_Otsu_thresholds'
        thresholdsDFPath = outputFolder + filesep + thresholdsDFname
        thresholds.to_csv(thresholdsDFPath + '.csv')
        
        
#%% Manually fix problematic Ch03 DOGs
# I know they are bad because way too many nuclei were found
badDOGs = ['OD05_BiBi656-614_plant7_inf2_Ch03','OD05_BiBi656-614_plant6_inf8_Ch03']
plantNames = ['OD1_plant1','OD1_plant2','OD1_plant3'
plantNames = ['OD05_BiBi654-514_plant1','OD05_BiBi654-514_plant2','OD05_BiBi654-514_plant3','OD05_BiBi654-514_plant4',
              'OD005_plant1','OD005_plant2','OD005_plant3','OD005_plant4',
              'OD05_sep654-514_plant1','OD05_sep654-514_plant2','OD05_sep654-514_plant3','OD05_sep654-514_plant4']

date = '12042023'
DOGpath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/DOGs'
MaskPath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/Masks'
filePathList = glob.glob('/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/*.npy')
channels = ['Ch01','Ch02','Ch03']
palettes = ['PuBu','BuGn','PuRd']

# loop over the files and show raw and dog side by side

for plant_counter, plantName in enumerate(plantNames): #loop over plants
    print(plantName)
    # find the file path names corresponding to this plant
    thisPlantPathNames = [i for i in filePathList if plantName in i]          
    for Channel_counter, channel in enumerate(channels): #loop over channels
        palette = palettes[Channel_counter]
        thisPlantThisChannelNames = [i for i in thisPlantPathNames if channel in i] 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for image_Counter, filename in enumerate(thisPlantThisChannelNames): #loop over infiltrations               
            # open the raw max projection image 
            Im = np.load(filename)
            # apply DOG filter
            DOGim = skimage.filters.difference_of_gaussians(Im,sigma)            
            # segment using the Otsu agorithm 
            OTSUthresh = skimage.filters.threshold_otsu(DOGim)
            Im2 = DOGim>OTSUthresh
            #show the images side by side
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(Im,palette,vmin=np.min(Im),vmax=np.max(Im)/4) 
            #plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)
            ax2.imshow(Im2,palette,vmax=np.max(Im2)/5)
            fig.suptitle(filename)
            plt.show()  


#%%  fix masks bymanually changing the otsu thresholds
date = '12112023'
DOGpath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/DOGs'
MaskPath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/Masks'
sigma = 5
palette = 'Greys'#'PuRd' #'YlGn'#
sample = 'comp_plant5_inf1_Ch03' 
# sample = 'comp_plant6_inf1_Ch03' 
# sample = 'comp_plant6_inf7_Ch03' 
filename = '/Volumes/JSALAMOS/LeicaDM6B/'+ date +'/Max_projections/'+ sample +'.npy'
maskname = '/Volumes/JSALAMOS/LeicaDM6B/'+ date +'/Masks/'+ sample +'.npy'

Im1 = np.load(filename)
Im2 = np.load(maskname)

DOGim = skimage.filters.difference_of_gaussians(Im1,sigma)            
  # threshold the nuclear mask using the Otsu agorithm 
OTSUthresh = skimage.filters.threshold_otsu(DOGim)
print(OTSUthresh)

#show the image
fig  = plt.figure()
plt.imshow(Im1,'inferno',vmin=np.min(Im1),vmax=np.max(Im1)/1) 
#plt.imshow(np.log2(Im1),'magma_r',vmax=np.max(np.log2(Im1))/1.005)
plt.title('Max projection')

#plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)

fig  = plt.figure()
plt.imshow(Im2,palette,vmin=np.min(Im2),vmax=np.max(Im2)/4) 
plt.title('original Mask')
#plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)

# Im3 = DOGim>OTSUthresh
# fig  = plt.figure()
# plt.imshow(Im3,palette,vmax=np.max(Im2)/5)
# plt.title('Otsu thresholded DOG')
# plt.show()  


OTSUthresh2 = OTSUthresh*3
fig = plt.figure()
ImMask = DOGim>OTSUthresh2
plt.imshow(ImMask,palette,vmax=np.max(ImMask)/5)
plt.title('lowered Otsu Thresholded DOG')
plt.show() 

#save DOG to  corresponding folder
DOGfileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
np.save(DOGpath + filesep + DOGfileName + '.npy' ,DOGim)

#save Mask to corresponding folder
#save Mask
np.save(MaskPath + filesep + sample + '.npy' ,ImMask)       
            
#%% Now do the counting in each channel separately, NOT applying the blue mask to all
# minArea = 65 # for filtering objects by size
# maxArea = 700
# sigma = 5 #for DOGs
# ODdict = {"inf1": 0.002, "inf2": 0.005, "inf3": 0.01, "inf4": 0.02, "inf5": 0.05, "inf6": 0.1, "inf7": 0.2, "inf8": 0.5, "inf9": 1}
# ODInfdict = {"inf1": 0.002, "inf2": 0.005, "inf3": 0.01, "inf4": 0.02, "inf5": 0.05, "inf6": 0.1,"inf7": 0.2, "inf8": 0.5, "inf9": 1}
# ODtotdict = {"OD01":0.1,"OD05":0.5,"OD2":2}
# originalLIFFiles = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/09192023/*.lif")
# maxProjectedFiles = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/09192023/Max_projections/*.npy")

# AllData2 = pd.DataFrame([], columns=['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','meanFluoGFP','SDFluoGFP','meanFluoRFP','SDFluoRFP',
#                                      'meanSumFluoGFP','SDSumFluoGFP','meanSumFluoRFP','SDSumFluoRFP','meanAreaBFP','SDAreaBFP','meanAreaGFP','SDAreaGFP',
#                                      'meanAreaRFP','SDAreaRFP'])
# AllData2['filename'] = originalLIFFiles
# AllData2OutputPath = '/Volumes/JSALAMOS/LeicaDM6B/09192023/Max_projections/AllData2.csv'
# outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/09192023/Max_projections'
# # populate with what we already know
# for idx, filename in enumerate(originalLIFFiles):
#     # get total OD
#     ODtot = ODtotdict[filename[filename.find('OD'):filename.find('plant')-1]]
#     # get titration OD
#     infOD = ODdict[filename[filename.find('inf'):filename.find('inf')+4]]
#     # get the plant ID
#     plantID = filename[filename.find('plant')+5]
#     AllData2.loc[AllData2['filename'].str.fullmatch(filename),['plant','ODtot','OD']] = [plantID,ODtot,infOD]

# #maxProjectedFiles = maxProjectedFiles[0:2]
# for idx, filename in enumerate(maxProjectedFiles): # loop over all files
#     print(str(100*np.round(idx/len(maxProjectedFiles),3))+ ' % of images processed')  
#     # get total OD
#     ODtot = ODtotdict[filename[filename.find('OD'):filename.find('plant')-1]]
#     # get titration OD
#     infOD = ODdict[filename[filename.find('inf'):filename.find('inf')+4]]
#     # get the plant ID
#     plantID = filename[filename.find('plant')+5]
#     # get the channel
#     channelID = filename[filename.find('Ch')+3]
#     # get the original LIF file name
#     LIFName = filename[filename.find('OD'):filename.find('Ch0')-1]
#     # get the DOG segmentation threshold
#     plantName = LIFName[0:-5]
#     Thresholds = pd.read_csv(outputFolder + filesep + plantName + '_Otsu_thresholds.csv')

    
#     #*** now do the actual counting! ***
    
#     #load the image
#     Im = np.load(filename)
#     # load the previously generated Otsu threshold of this image
#     threshVal = Thresholds.loc[Thresholds['filename']==filename,'threshold'].values[0]
#     threshVal = threshVal
    
#     # create DOG filtered image and then segment using Otsu threshold
#     DOGim = skimage.filters.difference_of_gaussians(Im,sigma)
#     # show
#     fig = plt.figure()
#     fig.set_size_inches(4.5, 4.5)
#     plt.imshow(DOGim,cmap='cubehelix_r',vmax=np.max(DOGim)/5)
#     plt.title('DOG-filtered nuclear label channel')
#     plt.show() 
#     # apply Otsu threshold to DOG filtered image
#     imMask = DOGim > threshVal
    
#     # apply watershed algorithm to separate joined nuclei
#     distance = ndimage.distance_transform_edt(imMask)
#     local_maxi = peak_local_max(
#     distance, indices=False, footprint=np.ones((10, 10)), labels=imMask)
#     markers = measure.label(local_maxi)
#     labels_ws = watershed(-distance, markers, mask=imMask)
#     # label each nucleus
#     num_cells = np.max(labels_ws)                                   
#     # Print out how many we identified.
#     print("Number of objects found: %s" %num_cells)
#     new_cmap = rand_cmap(num_cells, type='bright', first_color_black=True, last_color_black=False, verbose=True)
#     # # show
#     # plt.figure()
#     # plt.imshow(labels_ws, cmap=new_cmap)
#     # plt.show() 
    
#     # separate joined nuclei using watershed and then filter by area
#     [areaFiltMask,labels_ws2,num_cells2,meanArea,sdArea] = AreaFilter(imMask,minArea,maxArea)

#     # apply the nuclear mask to the image
#     maskedIm = Im * areaFiltMask
    
#     # calculate fluorescence intensities
#     MeanFluos = np.zeros(num_cells2) # to store values
#     IntFluos = np.zeros(num_cells2) # to store values

#     for i in range(num_cells2): # Loop through each nucleus. Remember that we have to start indexing at 1 in this case!
#         #print(str(100*np.round(i/num_cells2,3))+ ' % of nuclei processed')    
#         # select the binary mask corresponding to this nucleus   
#         singleNucleusMask = labels_ws2 == i + 1   
#         # apply the mask to each channel by multiplying the binary mask by the max projected image
#         singleNucleus = maskedIm * singleNucleusMask
        
#         # select only the pixels corresponding to the mask in each channel
#         nucleusPix = singleNucleus[singleNucleus>0]
#         bottom = np.percentile(nucleusPix,10) # the fluo value of the 25% percetile
#         top = np.percentile(nucleusPix,90) # the fluo value of the 25% percetile
#         MeanFluos[i] = np.mean(nucleusPix[np.where((nucleusPix >= bottom) & (nucleusPix <=top))]) # integrate the fluo of the middle 70% pixels
#         IntFluos[i] = np.sum(nucleusPix[np.where((nucleusPix >= bottom) & (nucleusPix <=top))]) # integrate the fluo of the middle 70% pixels
 
    
#     # # calculate positions
#     # props = skimage.measure.regionprops(labels_ws2, intensity_image=Im)
#     # regionprops_rowPos = np.zeros(num_cells2) #to store position values
#     # regionprops_colPos = np.zeros(num_cells2)   
#     # # Loop through each object and extract the properties. 
#     # for i in range(len(props)):
#     #     # Extract the positions
#     #     regionprops_rowPos[i] = props[i].centroid[0]
#     #     regionprops_colPos[i] = props[i].centroid[1]
    
#     # add the data to the corresponding row in the AllData dataframe
#     trueRule = (AllData2['filename'].str.contains(LIFName)) & (AllData2['plant']==plantID) & (AllData2['ODtot']==ODtot) & (AllData2['OD']==infOD)
    
#     if channelID == '1':
#         AllData2.loc[trueRule,'NBFP'] = num_cells2
#         AllData2.loc[trueRule,'meanAreaBFP'] = meanArea
#         AllData2.loc[trueRule,'SDAreaBFP'] = sdArea   
        
#     elif channelID == '2':
#         AllData2.loc[trueRule,'NGFP'] = num_cells2
#         AllData2.loc[trueRule,'meanFluoGFP'] = np.mean(MeanFluos)
#         AllData2.loc[trueRule,'SDFluoGFP'] = np.std(MeanFluos)
#         AllData2.loc[trueRule,'meanSumFluoGFP'] = np.mean(IntFluos)
#         AllData2.loc[trueRule,'SDSumFluoGFP'] = np.std(IntFluos)
#         AllData2.loc[trueRule,'meanAreaGFP'] = meanArea
#         AllData2.loc[trueRule,'SDAreaGFP'] = sdArea        
        
#     elif channelID == '3':
#         AllData2.loc[trueRule,'NRFP'] = num_cells2
#         AllData2.loc[trueRule,'meanFluoRFP'] = np.mean(MeanFluos)
#         AllData2.loc[trueRule,'SDFluoRFP'] = np.std(MeanFluos)
#         AllData2.loc[trueRule,'meanSumFluoRFP'] = np.mean(IntFluos)
#         AllData2.loc[trueRule,'SDSumFluoRFP'] = np.std(IntFluos)
#         AllData2.loc[trueRule,'meanAreaRFP'] = meanArea
#         AllData2.loc[trueRule,'SDAreaRFP'] = sdArea   
        
#     #save
#     AllData2.to_csv(AllData2OutputPath)

    
# #%% COUNT DOBLE TRANSFORMATIONS

# # we'll just use the masks here. After filtering by area, we ask how many overlap.

# minArea = 65 # for filtering objects by size
# maxArea = 700

# ODdict = {"inf1": 0.002, "inf2": 0.005, "inf3": 0.01, "inf4": 0.02, "inf5": 0.05, "inf6": 0.1, "inf7": 0.2, "inf8": 0.5, "inf9": 1}
# ODInfdict = {"inf1": 0.002, "inf2": 0.005, "inf3": 0.01, "inf4": 0.02, "inf5": 0.05, "inf6": 0.1,"inf7": 0.2, "inf8": 0.5, "inf9": 1}
# ODtotdict = {"OD01":0.1,"OD05":0.5,"OD2":2}
# originalLIFFiles = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/08282023/*.lif")
# maxProjectedFiles = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/08282023/Max_projections/*.npy")

# plantNames = ['OD01_plant1','OD01_plant2','OD01_plant3','OD01_plant4','OD05_plant1','OD05_plant2','OD05_plant3',
#               'OD2_plant1','OD2_plant2','OD2_plant3']

# #plantNames = ['OD01_plant1']
# channels = ['Ch01','Ch02','Ch03']
# palettes = ['Blues','Greens','Reds']

# # grab the file names of all the max projections
# filePathList = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/08282023/Max_projections/*.npy")
# DOGfilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/08282023/DOGs'
# MaskFilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/08282023/Masks'
# outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/08282023/Max_projections'
# multiplicityFolder = '/Volumes/JSALAMOS/LeicaDM6B/08282023/multiplicityData.csv'


# #initialize a dataframe to store values
# cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','ObsPBoth','ExpPBoth']
# multiplicityData = pd.DataFrame([], columns=cols)

# # we loop over infiltrations

# for plant, plantName in enumerate(plantNames):
#     print('plant: ' + str(plant/len(plantNames)))
#     if 'OD01' in plantName:
#         ODdict2 = dict(list(ODdict.items())[0:7]) # because OD01 doesn't have infiltrations 8 and 9
#     else:
#         ODdict2 = ODdict  
#     for inf, infName in enumerate(list(ODdict2.keys())):
#         print('infiltration: ' + str(inf/len(list(ODdict.keys()))))
#         filename = plantName + '_' + infName
#         filePrefix = MaskFilesOutput + filesep + filename
#         ODtot = ODtotdict[plantName[plantName.find('OD'):plantName.find('plant')-1]] #total OD
#         infOD = ODdict[infName] #titration OD
        
#         # load the blue channel
#         BFPMask = np.load(filePrefix + '_Ch01.npy')
#         # apply area filter and count number of nuclei
#         [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea)
#         # load the green channel
#         GFPMask = np.load(filePrefix + '_Ch02.npy')        
#         # apply the area filter and count number of nuclei
#         [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(GFPMask,minArea,maxArea)
#         # get the positions of green nuclei
#         [rowPosGFP,colPosGFP] = getPositions(labels_ws2GFP,NGFP)
#         # load the red channel
#         RFPMask = np.load(filePrefix + '_Ch03.npy') 
#         # apply the area filter and count number of nuclei
#         [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(RFPMask,minArea,maxArea)
#         # get the positions of red nuclei
#         [rowPosRFP,colPosRFP] = getPositions(labels_ws2RFP,NRFP)
#         # count how many greens overlap with reds
#         distanceThreshold = 10 # pixels
#         NBoth = countOverlaps(rowPosGFP,colPosGFP,rowPosRFP,colPosRFP,distanceThreshold)[0]
#         PBoth = NBoth/NBFP
        
#         pGreen = NGFP/NBFP
#         pRed = NRFP/NBFP
#         PExpectedBoth = pGreen*pRed
#         NExpectedBoth = PExpectedBoth*NBFP
        
#         data = [filename,plantName,ODtot,infOD,NBFP,NGFP,NRFP,PBoth,PExpectedBoth]
#         multiplicityData.loc[len(multiplicityData)+1, cols] = data
        
#         multiplicityData.to_csv(multiplicityFolder)
        
#%% Here we do the analysis but imposing the blue nuclear mask onto the green and red masks


#date = '10-16-23'
minArea = 60 # for filtering objects by area, 75 for Leica, 60 for Zeiss
maxArea = 850 # 750 for Leica, 850 for Zeiss
distanceThreshold = 10 # pixels, max distance between centroids in two channels for counting as a double transformation

ODdict= {"inf1": 0.002, "inf2": 0.005, "inf3": 0.01, "inf4": 0.02, "inf5": 0.05, "inf6": 0.1,"inf7": 0.2, "inf8": 0.5}
#ODdict= {"inf1": 0.005, "inf2": 0.01, "inf3": 0.05, "inf4": 0.1, "inf5": 0.5, "inf6": 1,"inf7": 2, "inf8": 3}
ODtotdict = {"OD005":0.05,"OD01":0.1,"OD05":0.5,"OD1":1,"OD2":2,"OD3":3}


plantNames = ['','','','','','','']


# grab the file names of all the max projections
date = '12042023'
filePathList = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/"  + date + "/Max_projections/*.npy")
DOGfilesOutput = "/Volumes/JSALAMOS/LeicaDM6B/" + date + "/DOGs"
MaskFilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Masks'
outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/AllData3.csv'
maxProjectionPath = "/Volumes/JSALAMOS/LeicaDM6B/" + date + "/Max_projections/"

# # grab the file names of all the max projections
# date = '2023/11-20-23'
# filePathList = glob.glob('/Volumes/JSALAMOS/lsm710/'  + date + '/Max_projections/*.npy')
# DOGfilesOutput = '/Volumes/JSALAMOS/lsm710/' + date + '/DOGs'
# MaskFilesOutput = '/Volumes/JSALAMOS/lsm710/' + date + '/Masks'
# outputFolder = '/Volumes/JSALAMOS/lsm710/' + date + '/Max_projections/AllData3.csv'
# maxProjectionPath = '/Volumes/JSALAMOS/lsm710/' + date + '/Max_projections/'


#initialize a dataframe to store values
cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','ObsPBoth','ExpPBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']

AllData3 = pd.DataFrame([], columns=cols)

# # this is to append to incomplete, previously generated  resutls
AllData3 = pd.read_csv('/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/AllData3.csv')
#plantNames = plantNames[7:]

for plant, plantName in enumerate(plantNames): #loop over plants
    print('plant: ' + str(plant/len(plantNames)))
    if 'OD05_BiBi656-614' in plantName:
        ODdict2 = dict(list(ODdict.items())[0:6]) # we need this because this experiment doesn't have infiltrations 7-9
    else:
        ODdict2 = ODdict  
    for inf, infName in enumerate(list(ODdict2.keys())): # loop over infiltration IDs
        print('infiltration: ' + str(inf/len(list(ODdict.keys())))) # for keeping track of progress
        filename = plantName + '_' + infName
        filePrefix = MaskFilesOutput + filesep + filename
        ODtot = ODtotdict[plantName[plantName.find('OD'):plantName.find('_')]] #total OD
        infOD = ODdict[infName] #titration OD
        
        # load the masks of the BFP channel corresponding to this one image
        # remember, a mask is a binary file generated by thresholding the DOG image by the OTSU threshold
        my_file = Path(filePrefix + '_Ch01.npy')
        if my_file.is_file(): # if the max projections exist
            BFPMask = np.load(filePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
             # and load the the other masks...
            GFPMask = np.load(filePrefix + '_Ch02.npy')
            RFPMask = np.load(filePrefix + '_Ch03.npy')
            # apply the area-filtered BFP mask to the GFP and RFP masks
            newGFPMask = areaFiltMaskBFP * GFPMask
            newRFPMask = areaFiltMaskBFP * RFPMask
            # now apply area filter and count number of nuclei in these GFP and RFP masks as well
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            
            # now let's figure out how many are double transformations
            [rowPosGFP,colPosGFP] = getPositions(labels_ws2GFP,NGFP)# get the positions of green nuclei...
            [rowPosRFP,colPosRFP] = getPositions(labels_ws2RFP,NRFP)# and of red nuclei...
            # count how many greens overlap with reds
            NBoth = countOverlaps(rowPosGFP,colPosGFP,rowPosRFP,colPosRFP,distanceThreshold)[0]
            PBoth = NBoth/NBFP
            pGreen = NGFP/NBFP
            pRed = NRFP/NBFP
            PExpectedBoth = pGreen*pRed
            NExpectedBoth = PExpectedBoth*NBFP # if independent, the probability of two random events is their multiplication
            
            # Now we deal with the fluorescence intensity
            GFPmaxProj = np.load(maxProjectionPath + filename + '_Ch02.npy') # load the maximum projection of GFP...
            RFPmaxProj = np.load(maxProjectionPath + filename + '_Ch03.npy') #...and RFP corresponding to this image
            
            # calculate fluorescence intensities
            [meanAvgFluoGFP,sdAvgFluoGFP,meanIntFluoGFP,sdIntFluoGFP] = calculateNucleiFluos(labels_ws2GFP,GFPmaxProj,NGFP)  
            [meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoRFP,sdIntFluoRFP] = calculateNucleiFluos(labels_ws2RFP,RFPmaxProj,NRFP)
            
            
            data = [filename,plantName,ODtot,infOD,NBFP,NGFP,NRFP,NBoth,PBoth,PExpectedBoth,meanAvgFluoGFP,sdAvgFluoGFP,
                    meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoGFP,sdIntFluoGFP,meanIntFluoRFP,sdIntFluoRFP]
            
            AllData3.loc[len(AllData3)+1, cols] = data
        
            AllData3.to_csv(outputFolder)
        




#%% IF A DIFFERENT EXPERIMENT WAS RUN ON THE SAME DAY

ODdict = {"inf1": 0.005, "inf2": 0.01, "inf3": 0.05, "inf4": 0.1, "inf5": 0.5, "inf6": 1,"inf7": 2, "inf8": 3}
ODtotdict = {"OD01":0.1,"OD05":0.5,"OD2":2,"OD3":3}

plantNames = ['OD3_plant1','OD3_plant2','OD3_plant3','OD3_plant4','OD3_plant5','OD3_plant6']

# this is to append to incomplete, previously generated  resutls
AllData3 = pd.read_csv('/Volumes/JSALAMOS/LeicaDM6B/'+date+'/Max_projections/AllData3.csv')
# plantNames = plantNames[5:]

for plant, plantName in enumerate(plantNames): #loop over plants
    print('plant: ' + str(plant/len(plantNames)))
    if 'OD01' in plantName:
        ODdict2 = dict(list(ODdict.items())[0:7]) # we need this because OD01 doesn't have infiltrations 8 and 9
    else:
        ODdict2 = ODdict  
    for inf, infName in enumerate(list(ODdict2.keys())): # loop over infiltration IDs
        print('infiltration: ' + str(inf/len(list(ODdict.keys())))) # for keeping track of progress
        filename = plantName + '_' + infName
        filePrefix = MaskFilesOutput + filesep + filename
        ODtot = ODtotdict[plantName[plantName.find('OD'):plantName.find('_')]] #total OD
        infOD = ODdict[infName] #titration OD
        
        # load the masks of the BFP channel corresponding to this one image
        # remember, a mask is a binary file generated by thresholding the DOG image by the OTSU threshold
        my_file = Path(filePrefix + '_Ch01.npy')
        if my_file.is_file(): # if the max projections exist
            BFPMask = np.load(filePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
             # and load the the other masks...
            GFPMask = np.load(filePrefix + '_Ch02.npy')
            RFPMask = np.load(filePrefix + '_Ch03.npy')
            # apply the area-filtered BFP mask to the GFP and RFP masks
            newGFPMask = areaFiltMaskBFP * GFPMask
            newRFPMask = areaFiltMaskBFP * RFPMask
            # now apply area filter and count number of nuclei in these GFP and RFP masks as well
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            
            # now let's figure out how many are double transformations
            [rowPosGFP,colPosGFP] = getPositions(labels_ws2GFP,NGFP)# get the positions of green nuclei...
            [rowPosRFP,colPosRFP] = getPositions(labels_ws2RFP,NRFP)# and of red nuclei...
            # count how many greens overlap with reds
            NBoth = countOverlaps(rowPosGFP,colPosGFP,rowPosRFP,colPosRFP,distanceThreshold)[0]
            PBoth = NBoth/NBFP
            pGreen = NGFP/NBFP
            pRed = NRFP/NBFP
            PExpectedBoth = pGreen*pRed
            NExpectedBoth = PExpectedBoth*NBFP # if independent, the probability of two random events is their multiplication
            
            # Now we deal with the fluorescence intensity
            GFPmaxProj = np.load(maxProjectionPath + filename + '_Ch02.npy') # load the maximum projection of GFP...
            RFPmaxProj = np.load(maxProjectionPath + filename + '_Ch03.npy') #...and RFP corresponding to this image
            
            # calculate fluorescence intensities
            [meanAvgFluoGFP,sdAvgFluoGFP,meanIntFluoGFP,sdIntFluoGFP] = calculateNucleiFluos(labels_ws2GFP,GFPmaxProj,NGFP)  
            [meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoRFP,sdIntFluoRFP] = calculateNucleiFluos(labels_ws2RFP,RFPmaxProj,NRFP)
            
            
            data = [filename,plantName,ODtot,infOD,NBFP,NGFP,NRFP,NBoth,PBoth,PExpectedBoth,meanAvgFluoGFP,sdAvgFluoGFP,
                    meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoGFP,sdIntFluoGFP,meanIntFluoRFP,sdIntFluoRFP]
            
            AllData3.loc[len(AllData3)+1, cols] = data
        
            AllData3.to_csv(outputFolder)




#%% COMPETITION EXPERIMENT 12/11/2023: Here we do the analysis but imposing the blue nuclear mask onto the green and red masks

date = '12112023'
minArea = 75 # for filtering objects by area, 75 for Leica, 60 for Zeiss
maxArea = 750 # 750 for Leica, 850 for Zeiss
distanceThreshold = 10 # pixels, max distance between centroids in two channels for counting as a double transformation

ODdict= {"inf1":'GV3101_EV_OD01', "inf2": 'GV3101_EV_OD2', "inf3": 'C58C1_OD01', "inf4": 'C58C1_OD2', "inf5": 'deltaVir1-2_OD01', "inf6": 'deltaVir1-2_OD2',"inf7": 'buffer_OD01'}
#ODdict= {"inf1": 0.005, "inf2": 0.01, "inf3": 0.05, "inf4": 0.1, "inf5": 0.5, "inf6": 1,"inf7": 2, "inf8": 3}
ODtotdict = {"OD005":0.05,"OD01":0.1,"OD05":0.5,"OD1":1,"OD2":2,"OD3":3}


plantNames = ['comp_plant1','comp_plant2','comp_plant3','comp_plant4','comp_plant5','comp_plant6','comp_plant7']


# grab the file names of all the max projections
filePathList = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/"  + date + "/Max_projections/*.npy")
DOGfilesOutput = "/Volumes/JSALAMOS/LeicaDM6B/" + date + "/DOGs"
MaskFilesOutput = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Masks'
outputFolder = '/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/CompData.csv'
maxProjectionPath = "/Volumes/JSALAMOS/LeicaDM6B/" + date + "/Max_projections/"

#initialize a dataframe to store values
cols = ['filename','plant','infName','NBFP','NGFP','NRFP','NBoth','ObsPBoth','ExpPBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']

CompData = pd.DataFrame([], columns=cols)

# # # this is to append to incomplete, previously generated  resutls
# CompData = pd.read_csv('/Volumes/JSALAMOS/LeicaDM6B/' + date + '/Max_projections/CompData.csv')
# #plantNames = plantNames[7:]

for plant, plantName in enumerate(plantNames): #loop over plants
    print('plant: ' + str(plant/len(plantNames)))
    if 'OD05_BiBi656-614' in plantName:
        ODdict2 = dict(list(ODdict.items())[0:6]) # we need this because this experiment doesn't have infiltrations 7-9
    else:
        ODdict2 = ODdict  
    for inf, infName in enumerate(list(ODdict2.keys())): # loop over infiltration IDs
        print('infiltration: ' + str(inf/len(list(ODdict.keys())))) # for keeping track of progress
        filename = plantName + '_' + infName
        filePrefix = MaskFilesOutput + filesep + filename
        #ODtot = ODtotdict[plantName[plantName.find('OD'):plantName.find('_')]] #total OD
        infName = ODdict[infName] #titration OD
        
        # load the masks of the BFP channel corresponding to this one image
        # remember, a mask is a binary file generated by thresholding the DOG image by the OTSU threshold
        my_file = Path(filePrefix + '_Ch01.npy')
        if my_file.is_file(): # if the max projections exist
            BFPMask = np.load(filePrefix + '_Ch01.npy') # load the blue channel mask...
            # apply area filter and count number of nuclei in BFP
            [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
             # and load the the other masks...
            GFPMask = np.load(filePrefix + '_Ch02.npy')
            RFPMask = np.load(filePrefix + '_Ch03.npy')
            # apply the area-filtered BFP mask to the GFP and RFP masks
            newGFPMask = areaFiltMaskBFP * GFPMask
            newRFPMask = areaFiltMaskBFP * RFPMask
            # now apply area filter and count number of nuclei in these GFP and RFP masks as well
            [areaFiltMaskGFP,labels_ws2GFP,NGFP,dummy,dummy] = AreaFilter(newGFPMask.astype(bool),minArea,maxArea,False)
            [areaFiltMaskRFP,labels_ws2RFP,NRFP,dummy,dummy] = AreaFilter(newRFPMask.astype(bool),minArea,maxArea,False)
            
            # now let's figure out how many are double transformations
            [rowPosGFP,colPosGFP] = getPositions(labels_ws2GFP,NGFP)# get the positions of green nuclei...
            [rowPosRFP,colPosRFP] = getPositions(labels_ws2RFP,NRFP)# and of red nuclei...
            # count how many greens overlap with reds
            NBoth = countOverlaps(rowPosGFP,colPosGFP,rowPosRFP,colPosRFP,distanceThreshold)[0]
            PBoth = NBoth/NBFP
            pGreen = NGFP/NBFP
            pRed = NRFP/NBFP
            PExpectedBoth = pGreen*pRed
            NExpectedBoth = PExpectedBoth*NBFP # if independent, the probability of two random events is their multiplication
            
            # Now we deal with the fluorescence intensity
            GFPmaxProj = np.load(maxProjectionPath + filename + '_Ch02.npy') # load the maximum projection of GFP...
            RFPmaxProj = np.load(maxProjectionPath + filename + '_Ch03.npy') #...and RFP corresponding to this image
            
            # calculate fluorescence intensities
            [meanAvgFluoGFP,sdAvgFluoGFP,meanIntFluoGFP,sdIntFluoGFP] = calculateNucleiFluos(labels_ws2GFP,GFPmaxProj,NGFP)  
            [meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoRFP,sdIntFluoRFP] = calculateNucleiFluos(labels_ws2RFP,RFPmaxProj,NRFP)
            
            
            data = [filename,plantName,infName,NBFP,NGFP,NRFP,NBoth,PBoth,PExpectedBoth,meanAvgFluoGFP,sdAvgFluoGFP,
                    meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoGFP,sdIntFluoGFP,meanIntFluoRFP,sdIntFluoRFP]
            
            CompData.loc[len(CompData)+1, cols] = data
        
            CompData.to_csv(outputFolder)
        








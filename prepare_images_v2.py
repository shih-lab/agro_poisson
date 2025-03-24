#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:17:30 2023

@author: simon_alamos
"""

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

# This is to let the user pick the folder where the raw data is stored without having to type the path
# I got it from https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

# for logging stuff
from datetime import datetime

import ast

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



def make_max_projections(rawdataInputFolder,maxProjOutputFolder,microscopeSystem,date):
    # microscopeSystem should be Leica_DM6B or Zeiss_LSM710

    # check if the output folder exists already
    isExist = os.path.exists(maxProjOutputFolder)
    if not isExist: # Create a new directory because it does not exist
       os.makedirs(maxProjOutputFolder)
       print("The output directory for max projections was created!")
 
    if microscopeSystem ==  'LeicaDM6B':
         # list all the raw image files in the folder 
        filePathList = glob.glob("/Volumes/JSALAMOS/LeicaDM6B/" + date + "/*.lif")
    
        # loop over the images to generate max projections
        for filepath in filePathList:
            print(filepath)
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
            fileNameCh01 = maxProjOutputFolder + filesep + prefix + '_Ch01.npy'
            np.save(fileNameCh01 ,maxCh01)
            fileNameCh02 = maxProjOutputFolder + filesep + prefix + '_Ch02.npy'
            np.save(fileNameCh02 ,maxCh02)
            fileNameCh03 = maxProjOutputFolder + filesep + prefix + '_Ch03.npy'
            np.save(fileNameCh03 ,maxCh03)
            
    elif microscopeSystem ==  'lsm710':
        # list all the raw image files in the folder 
        filePathList = glob.glob("/Volumes/JSALAMOS/lsm710/2024" + filesep + date + "/*.lsm")
        for filepath in filePathList:
            print(filepath)
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
            fileNameCh01 = maxProjOutputFolder + filesep + prefix + '_Ch01.npy'
            np.save(fileNameCh01 ,maxCh01)
            fileNameCh02 = maxProjOutputFolder + filesep + prefix + '_Ch02.npy'
            np.save(fileNameCh02 ,maxCh02)
            fileNameCh03 = maxProjOutputFolder + filesep + prefix + '_Ch03.npy'
            np.save(fileNameCh03 ,maxCh03)


    
    
def make_DOGs(experiment_database,expID,maxProjectionsFolder,DOGOutputFolder,MaskOutputFolder,Date,sigma,
              Otsu_scalingCh02,Otsu_scalingCh03,maskonly=False):
    channels = ['Ch01','Ch02','Ch03']
    palettes = ['Blues','YlGn','RdPu']
    # check if the oDOG utput folder exists already
    isExist = os.path.exists(DOGOutputFolder)
    if not isExist: # Create a new directory because it does not exist
        os.makedirs(DOGOutputFolder)
        print("The output directory for DOGs was created!")  
        
    # check if the oDOG utput folder exists already
    isExist = os.path.exists(MaskOutputFolder)
    if not isExist: # Create a new directory because it does not exist
        os.makedirs(MaskOutputFolder)
        print("The output directory for masks was created!")  
    
    plantNames = experiment_database[experiment_database['Experiment_ID']==exp]['plantNames']
    plantNames = plantNames.values[0].split(',')
    filePathList = glob.glob(maxProjectionsFolder + filesep + '*.npy')
    
    for plant_counter, plantName in enumerate(plantNames): #loop over plants
        print(plantName)
        # find the file path names corresponding to this plant
        thisPlantPathNames = [i for i in filePathList if plantName in i]  
            
        for Channel_counter, channel in enumerate(channels): #loop over channels
            palette = palettes[Channel_counter]
            thisPlantThisChannelNames = [i for i in thisPlantPathNames if channel in i]
            
            for image_Counter, filename in enumerate(thisPlantThisChannelNames): #loop over infiltrations (images)               
                # open the max projection image 
                MaxProIm = np.load(filename) 
                # apply DOG filter
                DOGim = skimage.filters.difference_of_gaussians(MaxProIm,sigma)            
                # threshold the DOG-filtered image using the Otsu agorithm 
                if channel == 'Ch01':
                    OTSUthresh = skimage.filters.threshold_otsu(DOGim)
                elif channel == 'Ch02':
                    OTSUthresh = skimage.filters.threshold_otsu(DOGim)/Otsu_scalingCh02
                elif channel == 'Ch03':
                    print('doing channel 3')
                    OTSUthresh = skimage.filters.threshold_otsu(DOGim)/Otsu_scalingCh03
                
                if not maskonly:
                    # save DOG
                    DOGfileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
                    np.save(DOGOutputFolder + filesep + DOGfileName + '.npy' ,DOGim)
                    
                #save Mask
                ImMask = DOGim>OTSUthresh
                MaskFileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
                np.save(MaskOutputFolder + filesep + MaskFileName + '.npy' ,ImMask)
                
                # show raw image and segmentation side by side
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(np.log2(MaxProIm+0.999),palette,vmin = np.log2(np.mean(MaxProIm)/3),vmax=np.max(np.log2(MaxProIm))/1.1)
                #plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)
                ax2.imshow(ImMask,palette)
#                fig.subtitle(filename)
                plt.title(MaskFileName)
                plt.show()  
                # save figure
                print(filename)
                fig.savefig(DOGOutputFolder + filesep + plantName + '_' + channel + '_' + str(image_Counter+1) + '.pdf')   # save the figure to file
                plt.close(fig)
            
                

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


def countNuclei(experiment_database,experimentID,minArea=75,maxArea=900,distanceThreshold=10):
    ODtotdict = {"OD005":0.05,"OD01":0.1,"OD05":0.5,"OD1":1,"OD2":2,"OD3":3}
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
    
    nuclei_counts_dataframe_name = 'experiment_' + str(experimentID) + '_nuclei_counts'

    if system == 'LeicaDM6B':
        #MaxProjFileList = glob.glob("/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/*.npy")
        MaskFilesLocation = '/Volumes/JSALAMOS' + filesep + system + filesep + date + '/Masks'
        outputFolder = '/Volumes/JSALAMOS' + filesep + system + filesep +  date + filesep + nuclei_counts_dataframe_name + '.csv'
        maxProjectionPath = "/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/"
    elif system == 'lsm710':
        #MaxProjFileList = glob.glob("/Volumes/JSALAMOS" + filesep + system + filesep + date + "/Max_projections/*.npy")
        MaskFilesLocation = '/Volumes/JSALAMOS' + filesep + system + filesep + '2024/' + date + '/Masks'
        outputFolder = '/Volumes/JSALAMOS' + filesep + system + filesep + '2024/' +  date + filesep + nuclei_counts_dataframe_name + '.csv'
        maxProjectionPath = "/Volumes/JSALAMOS" + filesep + system + filesep + '2024/' + date + '/Max_projections/'
        
    #initialize a dataframe to store values
    cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
            'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
    Nuclei_counts = pd.DataFrame([], columns=cols)
    
    
    for plant, plantName in enumerate(plantNames): #loop over plants

        for inf, infName in enumerate(list(ODdict.keys())): # loop over infiltration IDs
            #print('infiltration: ' + str(inf/len(list(ODdict.keys())))) # for keeping track of progress
            filename = plantName + '_' + infName
            singleMaskFilePrefix = MaskFilesLocation + filesep + filename
            infOD = ODdict[infName] #titration OD
            if 'OD' in plantName:
                ODtot = ODtotdict[plantName[plantName.find('OD'):plantName.find('_')]] #total OD
            else:
                ODtot = infOD
            
            # load the binary mask of the BFP channel corresponding to this one image
            BFP_mask_file = Path(singleMaskFilePrefix + '_Ch01.npy')
            if BFP_mask_file.is_file(): # if the max projections exist
                print(singleMaskFilePrefix)
                BFPMask = np.load(singleMaskFilePrefix + '_Ch01.npy') # load the blue channel mask...
                # apply area filter and count number of nuclei in BFP
                [areaFiltMaskBFP,labels_ws2BFP,NBFP,dummy,dummy] = AreaFilter(BFPMask,minArea,maxArea,False)
                # and now load the the other masks...
                GFPMask = np.load(singleMaskFilePrefix + '_Ch02.npy')
                RFPMask = np.load(singleMaskFilePrefix + '_Ch03.npy')
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
                
                # Now we deal with the fluorescence intensity
                GFPmaxProj = np.load(maxProjectionPath + filename + '_Ch02.npy') # load the maximum projection of GFP...
                RFPmaxProj = np.load(maxProjectionPath + filename + '_Ch03.npy') #...and RFP corresponding to this image
                
                # calculate fluorescence intensities
                [meanAvgFluoGFP,sdAvgFluoGFP,meanIntFluoGFP,sdIntFluoGFP] = calculateNucleiFluos(labels_ws2GFP,GFPmaxProj,NGFP)  
                [meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoRFP,sdIntFluoRFP] = calculateNucleiFluos(labels_ws2RFP,RFPmaxProj,NRFP)
                
                
                data = [filename,plantName,ODtot,infOD,NBFP,NGFP,NRFP,NBoth,meanAvgFluoGFP,sdAvgFluoGFP,
                        meanAvgFluoRFP,sdAvgFluoRFP,meanIntFluoGFP,sdIntFluoGFP,meanIntFluoRFP,sdIntFluoRFP]
                
                Nuclei_counts.loc[len(Nuclei_counts)+1, cols] = data
            
                Nuclei_counts.to_csv(outputFolder)
        



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

#%% load the experiment database

print('navigate to the folder where the experiment database file is stored - then select any file')
file_path = filedialog.askopenfilename() # store the file path as a string
lastFileSep = file_path.rfind(filesep) # find the position of the last path separator
folderpath = file_path[0:lastFileSep] # get the part of the path corresponding to the folder where the chosen file was located
experiment_database_filePath = folderpath + filesep + 'experiment_database.csv'
experiment_database = pd.read_csv(experiment_database_filePath)


#%% generate max projections

experimentIDs = experiment_database['Experiment_ID']
#experimentIDs = [30,31]
rawDataParentFolder = '/Volumes/JSALAMOS'
for exp in experimentIDs[-1:]:
    
    Date = experiment_database[experiment_database['Experiment_ID']==exp]['Date']
    Date = str(Date.values[0])
    System = experiment_database[experiment_database['Experiment_ID']==exp]['System']
    System = System.values[0]
        
    if System == 'LeicaDM6B':
        rawDataFolder = rawDataParentFolder + filesep + 'LeicaDM6B' + filesep + Date
    elif System == 'lsm710':
        rawDataFolder = rawDataParentFolder + filesep + 'lsm710' + filesep + '2024' + filesep + Date
        
    maxProjOutputFolder = rawDataFolder + filesep + 'Max_projections'
    
    make_max_projections(rawDataFolder,maxProjOutputFolder,System,Date)
    
    # # log the date and time this part of the code was executed
    # runDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # experiment_database[experiment_database['Experiment_ID']==exp]['max_projections'] = runDate
    
    # # update the experiment_database spreadsheet 
    # experiment_database.to_csv(experiment_database_filePath)


#%% segment nuclei using a DOG filter and then Otsu segmentation

experimentIDs = experiment_database['Experiment_ID']
#experimentIDs = [30]
rawDataParentFolder = '/Volumes/JSALAMOS'

for exp in experimentIDs[-1:]:
    print(exp)
    Date = experiment_database[experiment_database['Experiment_ID']==exp]['Date']
    Date = str(Date.values[0])
    System = experiment_database[experiment_database['Experiment_ID']==exp]['System']
    System = System.values[0]
    plantNames = experiment_database[experiment_database['Experiment_ID']==exp]['plantNames']
    
    if System == 'LeicaDM6B':
        rawDataFolder = rawDataParentFolder + filesep + 'LeicaDM6B' + filesep + Date
    elif System == 'lsm710':
        rawDataFolder = rawDataParentFolder + filesep + 'lsm710'+ filesep + '2024' + filesep + Date
        
    maxProjOutputFolder = rawDataFolder + filesep + 'Max_projections'
    DOGOutputFolder = rawDataFolder + filesep + 'DOGs'
    MaskOutputFolder = rawDataFolder + filesep + 'Masks'
    
    Otsu_scalingCh02 = 3.5
    Otsu_scalingCh03 = 0.1
    # the automatic Otsu threshold gets divided by this number, 1.25 was used for Leica, 2 for Zeiss
    sigma = 6 # size of the filter kernel, used 5-7 for Leica and Zeiss
    make_DOGs(experiment_database,exp,maxProjOutputFolder,DOGOutputFolder,MaskOutputFolder,Date,sigma,Otsu_scalingCh02,Otsu_scalingCh03,maskonly=False)
    
    # # log the date and time this part of the code was executed
    # runDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    # experiment_database[experiment_database['Experiment_ID']==exp]['max_projections'] = runDate
    
    # update the experiment_database spreadsheet 
    #experiment_database.to_csv(experiment_database_filePath)

    
#%% bad masks Zeiss images
# exp 18

# # exp 19
# 'OD05_plant5_inf4_Ch03'
# 'OD05_plant5_inf2_Ch03'
# 'OD05_plant5_inf7_Ch02'
# 'OD05_plant5_inf4_Ch02'
# 'OD05_plant5_inf2_Ch02'

# # exp 20
# 'OD2_plant6_inf2_Ch01'
# 'OD2_plant6_inf2_Ch03'
# 'OD2_plant5_inf5_Ch03'
# 'OD2_plant5_inf2_Ch03'
# 'OD2_plant5_inf5_Ch02'
# 'OD2_plant5_inf5_Ch01'
# 'OD2_plant5_inf2_Ch01'
# 'OD2_plant4_inf2_Ch03'
# 'OD2_plant4_inf4_Ch02'

# # exp 21
# 'OD05_BiBi656-614_plant4_inf4_Ch03'
# 'OD05_BiBi656-614_plant4_inf3_Ch03'
# 'OD05_BiBi656-614_plant4_inf2_Ch03'
# 'OD05_BiBi656-614_plant4_inf1_Ch03'
# 'OD05_BiBi656-614_plant4_inf2_Ch02'
# 'OD05_BiBi656-614_plant4_inf1_Ch02'
# 'OD05_BiBi656-614_plant3_inf3_Ch03'
# 'OD05_BiBi656-614_plant3_inf1_Ch03'
# 'OD05_BiBi656-614_plant3_inf4_Ch02'
# 'OD05_BiBi656-614_plant3_inf3_Ch02'
# 'OD05_BiBi656-614_plant3_inf2_Ch02'
# 'OD05_BiBi656-614_plant3_inf1_Ch02'
# 'OD05_BiBi656-614_plant2_inf4_Ch03'
# 'OD05_BiBi656-614_plant2_inf1_Ch03'
# 'OD05_BiBi656-614_plant2_inf2_Ch02'
# 'OD05_BiBi656-614_plant1_inf4_Ch03'
# 'OD05_BiBi656-614_plant1_inf3_Ch03'
# 'OD05_BiBi656-614_plant1_inf1_Ch03'
# 'OD05_BiBi656-614_plant1_inf2_Ch02'
# 'OD05_BiBi656-614_plant1_inf4_Ch02'
# 'OD05_BiBi656-614_plant1_inf1_Ch01'

# exp 22
# need to check

# # exp 23
# 'OD05_BiBi654-514_plant8_inf3_Ch03'
# 'OD05_BiBi654-514_plant8_inf5_Ch02'
# 'OD05_BiBi654-514_plant8_inf3_Ch02'
# 'OD05_BiBi654-514_plant8_inf2_Ch02'
# 'OD05_BiBi654-514_plant8_inf1_Ch02'
# 'OD05_BiBi654-514_plant8_inf1_Ch02'
# 'OD05_BiBi654-514_plant7_inf5_Ch03'
# 'OD05_BiBi654-514_plant7_inf5_Ch02'
# 'OD05_BiBi654-514_plant6_inf3_Ch03'
# 'OD05_BiBi654-514_plant6_inf5_Ch02'
# 'OD05_BiBi654-514_plant6_inf3_Ch02'
# 'OD05_BiBi654-514_plant5_inf5_Ch03'


# # exp 24
# 'OD05_sep654-514_plant7_inf5_Ch03'
# 'OD05_sep654-514_plant7_inf3_Ch03'
# 'OD05_sep654-514_plant7_inf1_Ch03'
# 'OD05_sep654-514_plant7_inf5_Ch02'
# 'OD05_sep654-514_plant6_inf5_Ch03'
# 'OD05_sep654-514_plant6_inf3_Ch03'
# 'OD05_sep654-514_plant6_inf3_Ch02'
# 'OD05_sep654-514_plant6_inf1_Ch02'
# 'OD05_sep654-514_plant5_inf3_Ch03'
# 'OD05_sep654-514_plant5_inf2_Ch03'
# 'OD05_sep654-514_plant5_inf1_Ch03'
# 'OD05_sep654-514_plant5_inf5_Ch02'
# 'OD05_sep654-514_plant5_inf3_Ch02'


# exp30
# 'all inf6s'
# 'coop_plant7_inf6_Ch03'
# 'coop_plant7_inf5_Ch03'
# 'coop_plant6_inf1_Ch03'
# 'coop_plant6_inf5_Ch02'
# 'coop_plant4_inf5_Ch03'
# 'coop_plant4_inf1_Ch03'
# 'coop_plant3_inf5_Ch03'
# 'coop_plant3_inf3_Ch03'
# 'coop_plant2_inf4_Ch03'
# 'coop_plant2_inf3_Ch03'
# 'coop_plant2_inf2_Ch03'
# 'coop_plant2_inf1_Ch03'
# 'coop_plant2_inf3_Ch02'
# 'coop_plant2_inf1_Ch02'
# 'coop_plant1_inf5_Ch03'
# 'coop_plant1_inf3_Ch03'






#%%

# pickled data = 
#'OD05_sep656-654_plant3_inf4_Ch02'

parentFolder = '/Volumes/JSALAMOS/LeicaDM6B/10232024/Max_projections/'
filename = parentFolder + '' + '.npy'
channel = filename[-4:]

MaxProIm = np.load(filename) 
# apply DOG filter
DOGim = skimage.filters.difference_of_gaussians(MaxProIm,sigma)            
# threshold the DOG-filtered image using the Otsu agorithm 
if channel == 'Ch01':
    OTSUthresh = skimage.filters.threshold_otsu(DOGim)
else:
    OTSUthresh = skimage.filters.threshold_otsu(DOGim)/4

# save DOG
DOGfileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
np.save(DOGOutputFolder + filesep + DOGfileName + '.npy' ,DOGim)

#save Mask
ImMask = DOGim>OTSUthresh
MaskFileName = filename[filename.rfind(filesep)+1 : filename.find('.')]
np.save(MaskOutputFolder + filesep + MaskFileName + '.npy' ,ImMask)


#%% bad masks in Leica images

# exp 1

# # exp 3
# 'OD2_plant2_inf9_Ch03'
# 'OD2_plant2_inf3_Ch03'
# 'OD2_plant2_inf1_Ch03'
# 'OD2_plant1_inf9_Ch03'
# 'OD2_plant1_inf8_Ch03'
# 'OD2_plant1_inf1_Ch03'
# 'OD2_plant1_inf9_Ch02'

# #exp 4
# 'OD01_plant7_inf7_Ch03'
# 'OD01_plant7_inf6_Ch03'
# 'OD01_plant7_inf5_Ch03'
# 'OD01_plant7_inf5_Ch02'
# 'OD01_plant6_inf7_Ch03'
# 'OD01_plant6_inf6_Ch03'
# 'OD01_plant6_inf3_Ch02'

# #exp 5
# 'OD05_plant6_inf9_Ch03'
# 'OD05_plant6_inf8_Ch03'
# 'OD05_plant6_inf7_Ch03'
# 'OD05_plant6_inf6_Ch03'
# 'OD05_plant6_inf5_Ch03'
# 'OD05_plant6_inf9_Ch02'
# 'OD05_plant6_inf8_Ch02'
# 'OD05_plant6_inf7_Ch02'
# 'OD05_plant4_inf9_Ch03'
# 'OD05_plant4_inf8_Ch03'
# 'OD05_plant4_inf7_Ch03'
# 'OD05_plant4_inf6_Ch03'
# 'OD05_plant4_inf8_Ch01'
# 'OD05_plant4_inf7_Ch01'

#exp 6

# #exp 7
# 'OD05_BiBi656-614_plant4_inf8_Ch02'
# 'OD05_BiBi656-614_plant4_inf7_Ch02'
# 'OD05_BiBi656-614_plant4_inf6_Ch02'
# 'OD05_BiBi656-614_plant4_inf5_Ch02'
# 'OD05_BiBi656-614_plant3_inf8_Ch03'
# 'OD05_BiBi656-614_plant3_inf4_Ch03'
# 'OD05_BiBi656-614_plant3_inf3_Ch03'
# 'OD05_BiBi656-614_plant3_inf1_Ch03'
# 'OD05_BiBi656-614_plant3_inf8_Ch02'
# 'OD05_BiBi656-614_plant2_inf8_Ch03'
# 'OD05_BiBi656-614_plant2_inf7_Ch03'
# 'OD05_BiBi656-614_plant2_inf8_Ch02'
# 'OD05_BiBi656-614_plant2_inf7_Ch02'
# 'OD05_BiBi656-614_plant1_inf3_Ch03'
# 'OD05_BiBi656-614_plant1_inf2_Ch03'
# 'OD05_BiBi656-614_plant1_inf1_Ch03'

# #exp 8
# 'OD005_plant4_inf5_Ch03'
# 'OD005_plant4_inf6_Ch03'

#exp 9

# #exp 10
# 'OD05_sep654-514_plant1_inf4_Ch01'

#exp 11

# #exp 12
# 'OD3_plant6_inf1_Ch02'

# #exp 13
# 'OD05_BiBi654-514_plant10_inf1_Ch03'
# 'OD05_BiBi654-514_plant10_inf5_Ch02'
# 'OD05_BiBi654-514_plant10_inf4_Ch02'
# 'OD05_BiBi654-514_plant10_inf3_Ch02'
# 'OD05_BiBi654-514_plant10_inf2_Ch02'
# 'OD05_BiBi654-514_plant10_inf1_Ch02'
# 'OD05_BiBi654-514_plant9_inf1_Ch03'
# 'OD05_BiBi654-514_plant9_inf4_Ch02'
# 'OD05_BiBi654-514_plant9_inf3_Ch02'
# 'OD05_BiBi654-514_plant9_inf2_Ch02'
# 'OD05_BiBi654-514_plant9_inf1_Ch02'
# 'OD05_BiBi654-514_plant7_inf4_Ch03'
# 'OD05_BiBi654-514_plant7_inf3_Ch02'
# 'OD05_BiBi654-514_plant7_inf2_Ch02'
# 'OD05_BiBi654-514_plant7_inf1_Ch02'
# 'OD05_BiBi654-514_plant6_inf3_Ch03'
# 'OD05_BiBi654-514_plant6_inf2_Ch02'
# 'OD05_BiBi654-514_plant6_inf1_Ch02'
# 'OD05_BiBi654-514_plant5_inf8_Ch02'
# 'OD05_BiBi654-514_plant5_inf6_Ch02'
# 'OD05_BiBi654-514_plant5_inf3_Ch02'
# 'OD05_BiBi654-514_plant5_inf2_Ch02'
# 'OD05_BiBi654-514_plant5_inf1_Ch02'

# #exp 14
# 'OD1_plant3_inf7_Ch02'
# 'OD1_plant3_inf3_Ch02'
# 'OD1_plant3_inf2_Ch02'
# 'OD1_plant3_inf1_Ch02'
# 'OD1_plant2_inf4_Ch02'
# 'OD1_plant2_inf4_Ch02'
# 'OD1_plant1_inf1_Ch02'
# 'OD1_plant1_inf2_Ch02'
# 'OD1_plant1_inf3_Ch02'

# #exp 15
# 'OD05_BiBi656-614_plant7_inf7_Ch03'
# 'OD05_BiBi656-614_plant7_inf2_Ch03'
# 'OD05_BiBi656-614_plant7_inf8_Ch02'
# 'OD05_BiBi656-614_plant7_inf2_Ch02'
# 'OD05_BiBi656-614_plant6_inf8_Ch03'
# 'OD05_BiBi656-614_plant6_inf3_Ch03'
# 'OD05_BiBi656-614_plant6_inf8_Ch02'
# 'OD05_BiBi656-614_plant6_inf1_Ch02'
# 'OD05_BiBi656-614_plant5_inf8_Ch03'
# 'OD05_BiBi656-614_plant5_inf8_Ch03'
# 'OD05_BiBi656-614_plant5_inf2_Ch02'

# #exp 16
# 'OD05_sep654-514_plant7_inf5_Ch03'
# 'OD05_sep654-514_plant7_inf6_Ch02'
# 'OD05_sep654-514_plant7_inf5_Ch02'
# 'OD05_sep654-514_plant7_inf4_Ch02'
# 'OD05_sep654-514_plant7_inf2_Ch02'
# 'OD05_sep654-514_plant7_inf1_Ch02'
# 'OD05_sep654-514_plant6_inf5_Ch03'
# 'OD05_sep654-514_plant6_inf2_Ch02'
# 'OD05_sep654-514_plant5_inf8_Ch02'
# 'OD05_sep654-514_plant5_inf4_Ch02'
# 'OD05_sep654-514_plant5_inf3_Ch02'
# 'OD05_sep654-514_plant5_inf2_Ch02'
# 'OD05_sep654-514_plant5_inf1_Ch02'

# #exp 17
# 'OD1_plant9_inf8_Ch03'
# 'OD1_plant9_inf8_Ch02'
# 'OD1_plant9_inf7_Ch02'
# 'OD1_plant9_inf2_Ch02'
# 'OD1_plant8_inf8_Ch03'
# 'OD1_plant8_inf8_Ch02'
# 'OD1_plant7_inf8_Ch03'
# 'OD1_plant7_inf7_Ch03'
# 'OD1_plant7_inf6_Ch03'
# 'OD1_plant7_inf1_Ch03'
# 'OD1_plant7_inf8_Ch02'
# 'OD1_plant7_inf7_Ch02'
# 'OD1_plant7_inf1_Ch02'
# 'OD1_plant7_inf2_Ch02'
# 'OD1_plant7_inf3_Ch02'
# 'OD1_plant6_inf8_Ch03'
# 'OD1_plant6_inf7_Ch03'
# 'OD1_plant6_inf6_Ch03'
# 'OD1_plant6_inf8_Ch02'
# 'OD1_plant6_inf7_Ch02'
# 'OD1_plant6_inf6_Ch02'
# 'OD1_plant5_inf8_Ch03'
# 'OD1_plant5_inf7_Ch03'
# 'OD1_plant5_inf6_Ch03'
# 'OD1_plant5_inf5_Ch03'
# 'OD1_plant5_inf8_Ch02'
# 'OD1_plant5_inf7_Ch02'
# 'OD1_plant4_inf8_Ch03'
# 'OD1_plant4_inf7_Ch03'
# 'OD1_plant4_inf6_Ch03'
# 'OD1_plant4_inf8_Ch02'
# 'OD1_plant4_inf7_Ch02'
# 'OD1_plant4_inf6_Ch02'
# 'OD1_plant4_inf3_Ch02'
# 'OD1_plant4_inf2_Ch02'

# # exp 25
# 'OD01_C58C1_plant5_inf3_Ch03'
# 'OD01_C58C1_plant5_inf2_Ch03'
# 'OD01_C58C1_plant5_inf6_Ch02'
# 'OD01_C58C1_plant5_inf5_Ch02'
# 'OD01_C58C1_plant5_inf4_Ch02'
# 'OD01_C58C1_plant5_inf3_Ch02'
# 'OD01_C58C1_plant5_inf2_Ch02'
# 'OD01_C58C1_plant5_inf1_Ch02'
# 'OD01_C58C1_plant4_inf4_Ch03'
# 'OD01_C58C1_plant4_inf3_Ch03'
# 'OD01_C58C1_plant4_inf2_Ch03'
# 'OD01_C58C1_plant4_inf1_Ch03'
# 'OD01_C58C1_plant4_inf6_Ch02'
# 'OD01_C58C1_plant4_inf5_Ch02'
# 'OD01_C58C1_plant4_inf4_Ch02'
# 'OD01_C58C1_plant4_inf3_Ch02'
# 'OD01_C58C1_plant4_inf1_Ch02'
# 'OD01_C58C1_plant3_inf4_Ch02'
# 'OD01_C58C1_plant3_inf3_Ch02'
# 'OD01_C58C1_plant3_inf2_Ch02'
# 'OD01_C58C1_plant3_inf1_Ch02'
# 'OD01_C58C1_plant2_inf1_Ch03'
# 'OD01_C58C1_plant2_inf6_Ch02'
# 'OD01_C58C1_plant2_inf5_Ch02'
# 'OD01_C58C1_plant2_inf4_Ch02'
# 'OD01_C58C1_plant2_inf3_Ch02'
# 'OD01_C58C1_plant2_inf2_Ch02'
# 'OD01_C58C1_plant2_inf1_Ch02'
# 'OD01_C58C1_plant1_inf1_Ch03'
# 'OD01_C58C1_plant1_inf6_Ch02'
# 'OD01_C58C1_plant1_inf5_Ch02'
# 'OD01_C58C1_plant1_inf4_Ch02'
# 'OD01_C58C1_plant1_inf3_Ch02'
# 'OD01_C58C1_plant1_inf2_Ch02'
# 'OD01_C58C1_plant1_inf1_Ch02'

# # exp 26
# 'OD05_C58C1_plant5_inf1_Ch03'
# 'OD05_C58C1_plant5_inf5_Ch02'
# 'OD05_C58C1_plant5_inf4_Ch02'
# 'OD05_C58C1_plant5_inf3_Ch02'
# 'OD05_C58C1_plant5_inf2_Ch02'
# 'OD05_C58C1_plant5_inf1_Ch02'
# 'OD05_C58C1_plant4_inf7_Ch03'
# 'OD05_C58C1_plant4_inf6_Ch03'
# 'OD05_C58C1_plant4_inf5_Ch03'
# 'OD05_C58C1_plant4_inf4_Ch03'
# 'OD05_C58C1_plant4_inf3_Ch03'
# 'OD05_C58C1_plant4_inf2_Ch03'
# 'OD05_C58C1_plant4_inf1_Ch03'
# 'OD05_C58C1_plant4_inf7_Ch02'
# 'OD05_C58C1_plant4_inf6_Ch02'
# 'OD05_C58C1_plant4_inf5_Ch02'
# 'OD05_C58C1_plant4_inf4_Ch02'
# 'OD05_C58C1_plant4_inf2_Ch02'
# 'OD05_C58C1_plant4_inf1_Ch02'
# 'OD05_C58C1_plant3_inf4_Ch03'
# 'OD05_C58C1_plant3_inf3_Ch03'
# 'OD05_C58C1_plant3_inf2_Ch03'
# 'OD05_C58C1_plant3_inf6_Ch02'
# 'OD05_C58C1_plant3_inf5_Ch02'
# 'OD05_C58C1_plant3_inf4_Ch02'
# 'OD05_C58C1_plant3_inf3_Ch02'
# 'OD05_C58C1_plant3_inf2_Ch02'
# 'OD05_C58C1_plant2_inf7_Ch03'
# 'OD05_C58C1_plant2_inf1_Ch03'
# 'OD05_C58C1_plant2_inf6_Ch02'
# 'OD05_C58C1_plant2_inf5_Ch02'
# 'OD05_C58C1_plant2_inf4_Ch02'
# 'OD05_C58C1_plant2_inf3_Ch02'
# 'OD05_C58C1_plant2_inf2_Ch02'
# 'OD05_C58C1_plant2_inf1_Ch02'
# 'OD05_C58C1_plant1_inf4_Ch03'
# 'OD05_C58C1_plant1_inf2_Ch03'
# 'OD05_C58C1_plant1_inf1_Ch03'
# 'OD05_C58C1_plant1_inf7_Ch02'
# 'OD05_C58C1_plant1_inf6_Ch02'
# 'OD05_C58C1_plant1_inf5_Ch02'
# 'OD05_C58C1_plant1_inf4_Ch02'
# 'OD05_C58C1_plant1_inf2_Ch02'

# # exp 27
# 'OD2_C58C1_plant5_inf6_Ch02'
# 'OD2_C58C1_plant5_inf5_Ch02'
# 'OD2_C58C1_plant5_inf4_Ch02'
# 'OD2_C58C1_plant5_inf3_Ch02'
# 'OD2_C58C1_plant5_inf2_Ch02'
# 'OD2_C58C1_plant5_inf1_Ch02'
# 'OD2_C58C1_plant4_inf4_Ch03'
# 'OD2_C58C1_plant4_inf3_Ch03'
# 'OD2_C58C1_plant4_inf3_Ch03'
# 'OD2_C58C1_plant4_inf2_Ch03'
# 'OD2_C58C1_plant4_inf1_Ch03'
# 'OD2_C58C1_plant4_inf6_Ch02'
# 'OD2_C58C1_plant4_inf5_Ch02'
# 'OD2_C58C1_plant4_inf4_Ch02'
# 'OD2_C58C1_plant4_inf3_Ch02'
# 'OD2_C58C1_plant4_inf8_Ch02'
# 'OD2_C58C1_plant4_inf7_Ch02'
# 'OD2_C58C1_plant4_inf2_Ch02'
# 'OD2_C58C1_plant3_inf5_Ch03'
# 'OD2_C58C1_plant3_inf4_Ch03'
# 'OD2_C58C1_plant3_inf3_Ch03'
# 'OD2_C58C1_plant3_inf8_Ch03'
# 'OD2_C58C1_plant3_inf2_Ch03'
# 'OD2_C58C1_plant3_inf1_Ch03'
# 'OD2_C58C1_plant3_inf6_Ch02'
# 'OD2_C58C1_plant3_inf5_Ch02'
# 'OD2_C58C1_plant3_inf4_Ch02'
# 'OD2_C58C1_plant3_inf8_Ch02'
# 'OD2_C58C1_plant3_inf7_Ch02'
# 'OD2_C58C1_plant2_inf2_Ch03'
# 'OD2_C58C1_plant2_inf1_Ch03'
# 'OD2_C58C1_plant2_inf6_Ch02'
# 'OD2_C58C1_plant2_inf5_Ch02'
# 'OD2_C58C1_plant2_inf4_Ch02'
# 'OD2_C58C1_plant2_inf3_Ch02'
# 'OD2_C58C1_plant2_inf8_Ch02'
# 'OD2_C58C1_plant2_inf2_Ch02'
# 'OD2_C58C1_plant2_inf1_Ch02'
# 'OD2_C58C1_plant1_inf5_Ch03'
# 'OD2_C58C1_plant1_inf2_Ch03'
# 'OD2_C58C1_plant1_inf1_Ch03'
# 'OD2_C58C1_plant1_inf7_Ch02'
# 'OD2_C58C1_plant1_inf6_Ch02'
# 'OD2_C58C1_plant1_inf5_Ch02'
# 'OD2_C58C1_plant1_inf4_Ch02'
# 'OD2_C58C1_plant1_inf3_Ch02'

# # exp 28
# 'noComp_plant4_inf4_Ch03'
# 'noComp_plant4_inf4_Ch03'
# 'noComp_plant4_inf7_Ch03'
# 'noComp_plant4_inf6_Ch03'
# 'noComp_plant4_inf5_Ch03'
# 'noComp_plant4_inf4_Ch03'
# 'noComp_plant4_inf3_Ch03'
# 'noComp_plant4_inf2_Ch03'
# 'noComp_plant4_inf1_Ch03'
# 'noComp_plant3_inf2_Ch03'
# 'noComp_plant3_inf6_Ch02'
# 'noComp_plant3_inf5_Ch02'
# 'noComp_plant3_inf4_Ch02'
# 'noComp_plant3_inf3_Ch02'
# 'noComp_plant3_inf2_Ch02'
# 'noComp_plant3_inf1_Ch02'
# 'noComp_plant2_inf1_Ch02'
# 'noComp_plant2_inf6_Ch02'
# 'noComp_plant2_inf4_Ch02'
# 'noComp_plant2_inf3_Ch02'
# 'noComp_plant2_inf2_Ch02'
# 'noComp_plant1_inf1_Ch02'
# 'noComp_plant1_inf5_Ch02'
# 'noComp_plant1_inf3_Ch02'
# 'noComp_plant1_inf2_Ch02'

# exp 30 and 31
# 'control_plant15_Ch01'
# 'control_plant11_Ch01'
# 'control_plant13_Ch01'
# 'control_plant5_Ch01'
# 'control_plant1_Ch01'
# 'cooperation_plant19_Ch03'
# 'cooperation_plant17_Ch03'
# 'cooperation_plant15_Ch03'
# 'cooperation_plant15_Ch02'
# 'cooperation_plant13_Ch02'
# 'cooperation_plant12_Ch03'
# 'cooperation_plant11_Ch03'
# 'cooperation_plant11_Ch02'
# 'cooperation_plant9_Ch03'
# 'cooperation_plant8_Ch03'
# 'cooperation_plant7_Ch03'
# 'cooperation_plant5_Ch03'
# 'cooperation_plant4_Ch01'
# 'cooperation_plant3_Ch03'
# 'cooperation_plant10_Ch03'
# 'cooperation_plant1_Ch03'


# # exp 36 and 37
# 'OD05_sep656-654_plant1_inf2_Ch02'
# 'OD05_sep656-654_plant1_inf3_Ch02'
# 'OD05_sep656-654_plant1_inf4_Ch02'
# 'OD05_sep656-654_plant1_inf2_Ch03'
# 'OD05_sep656-654_plant2_inf1_Ch02'
# 'OD05_sep656-654_plant2_inf2_Ch02'
# 'OD05_sep656-654_plant2_inf3_Ch02'
# 'OD05_sep656-654_plant2_inf4_Ch02'
# 'OD05_sep656-654_plant2_inf3_Ch03'
# 'OD05_sep656-654_plant3_inf1_Ch02'
# 'OD05_sep656-654_plant3_inf2_Ch02'
# 'OD05_sep656-654_plant4_inf1_Ch02'
# 'OD05_sep656-654_plant4_inf3_Ch02'
# 'OD05_sep656-654_plant4_inf4_Ch02'
# 'OD05_sep656-654_plant5_inf1_Ch02'
# 'OD05_sep656-654_plant5_inf2_Ch02'
# 'OD05_sep656-654_plant5_inf3_Ch02'
# 'OD05_sep656-654_plant5_inf1_Ch03'
# 'OD05_sep656-654_plant5_inf2_Ch03'
# 'OD05_sep656-654_plant6_inf2_Ch02'
# 'OD05_sep656-654_plant6_inf3_Ch02'
# 'OD05_sep656-654_plant6_inf4_Ch02'
# 'OD05_sep656-654_plant6_inf1_Ch03'
# 'OD05_sep656-654_plant7_inf1_Ch02'
# 'OD05_sep656-654_plant7_inf2_Ch02'
# 'OD05_sep656-654_plant7_inf3_Ch02'
# 'OD05_sep656-654_plant7_inf1_Ch03'
# 'OD05_sep656-654_plant7_inf2_Ch03'



# exp 38



#%%  fix masks by manually changing the otsu thresholds

# exp 36 and 3
date = '10232024'
DOGpath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/DOGs'
MaskPath = '/Volumes/JSALAMOS/LeicaDM6B/'+ date + '/Masks'
# DOGpath = '/Volumes/JSALAMOS/lsm710/2024/' + date + '/DOGs'
# MaskPath = '/Volumes/JSALAMOS/lsm710/2024/' + date + '/Masks'
sigma = 6
palette = 'Greys'#'PuRd' #'YlGn'#
sample = 'vircoop_plant5_inf4-17_Ch03'

filename = '/Volumes/JSALAMOS/LeicaDM6B/'+ date +'/Max_projections/'+ sample +'.npy'
maskname = '/Volumes/JSALAMOS/LeicaDM6B/'+ date +'/Masks/'+ sample +'.npy'
# filename = '/Volumes/JSALAMOS/lsm710/2024/' + date +'/Max_projections/'+ sample +'.npy'
# maskname = '/Volumes/JSALAMOS/lsm710/2024/' + date +'/Masks/'+ sample +'.npy'

Im1 = np.load(filename)
haveIm2 = False
# try:
Im2 = np.load(maskname)
# except ValueError:
#     haveIm2 = False
#     print("Oops!  That didn't work.  Try again...")

DOGim = skimage.filters.difference_of_gaussians(Im1,sigma)            
  # threshold the nuclear mask using the Otsu agorithm 
OTSUthresh = skimage.filters.threshold_otsu(DOGim)
print(OTSUthresh)

#show the image
fig  = plt.figure()
#plt.imshow(Im1,'inferno',vmin=np.min(Im1))#,vmax=np.max(Im1)/4) 
plt.imshow(Im1,'magma_r',vmax=np.max(Im1)/2)
#plt.imshow(Im1,'magma_r')
plt.title('Max projection')

#plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)

if haveIm2:
    fig  = plt.figure()
    plt.imshow(Im2,palette,vmin=np.min(Im2),vmax=np.max(Im2)/5) 
    plt.imshow(Im2,palette)
    plt.title('original Mask')
    #plt.imshow(np.log(Im),palette,vmax=np.max(np.log(Im))/1.15)

# Im3 = DOGim>OTSUthresh
# fig  = plt.figure()
# plt.imshow(Im3,palette,vmax=np.max(Im2)/5)
# plt.title('Otsu thresholded DOG')
# plt.show()  


OTSUthresh2 = OTSUthresh/1
#OTSUthresh2 = 0.04
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
 

#%%
experimentIDs = experiment_database['Experiment_ID']
#experimentIDs = [30]
rawDataParentFolder = '/Volumes/JSALAMOS'

for exp in experimentIDs[-1:]:
    print(exp)
    countNuclei(experiment_database,exp,minArea=40,maxArea=900,distanceThreshold=5)

#%% loop over masks to find pickles
folder = '/Volumes/JSALAMOS/LeicaDM6B/10232024/Masks'
filesList = glob.glob(folder +  "/*.npy")

for i, filename in enumerate(filesList):
    print(filename)
    np.load(filename)



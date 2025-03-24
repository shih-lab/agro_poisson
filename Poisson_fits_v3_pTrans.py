

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

# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

#%% Fitting stuff: define the objective function

# a useful function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# define the objective function
def fractionNotTransformed(dilutionOD,  alpha): 
    # dilutionOD = the OD of the culture that we're diluting, the one with the nuclear tag
    # NotTransf = our Y values, the fraction of cells that are not transformed (in log scale)
    # alpha = a sclaing factor to go from OD to poisson probability   
    NotTransfP = np.exp(-alpha * dilutionOD) # Poisson probability of zero successes
    return NotTransfP

# define the objective function
def fractionTransformed(dilutionOD,  alpha): 
    # dilutionOD = the OD of the culture that we're diluting, the one with the nuclear tag
    # NotTransf = our Y values, the fraction of cells that are not transformed (in log scale)
    # alpha = a sclaing factor to go from OD to poisson probability   
    TransfP = 1 - np.exp(-alpha * dilutionOD) # Poisson probability of zero successes
    return TransfP

def fractionTransformed2(dilutionOD,  alpha, fractionTransformable): 
    # dilutionOD = the OD of the culture that we're diluting, the one with the nuclear tag
    # alpha = a sclaing factor to go from OD to poisson probability   
    # fractionTransformable = fraction of cells that can get transformed
    TransfP = fractionTransformable * (1 - np.exp(-alpha * dilutionOD)) # Poisson probability of zero successes
    return TransfP

def fractionTransformed3(dilutionOD,  alpha, pTransfer): 
    # dilutionOD = the OD of the culture that we're diluting, the one with the nuclear tag
    # alpha = a sclaing factor to go from OD to poisson probability   
    # pTransfer = probability of transferring pVS1 given a contact
    TransfP = pTransfer * (1 - np.exp(-alpha * dilutionOD)) # Poisson probability of zero successes
    return TransfP

def getRsquared(ydata, fittedY):
    # can get the residual sum of squares 
    residuals = ydata - fittedY
    ss_res = np.sum(residuals**2)
    #  get the total sum of squares 
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    # get the R squared value
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared
    
    

#%% grab all the data

# load the experiment_database spreadsheet
print('navigate to the folder where the experiment database file is stored - then select any file')
file_path = filedialog.askopenfilename() # store the file path as a string
lastFileSep = file_path.rfind(filesep) # find the position of the last path separator
folderpath = file_path[0:lastFileSep] # get the part of the path corresponding to the folder where the chosen file was located
experiment_database_filePath = folderpath + filesep + 'experiment_database.csv'
experiment_database = pd.read_csv(experiment_database_filePath)

# select which experiments to aggregate. Refer to 'experiment_database' spreadsheet for more info
experimentIDs = ['1','2','3','4','5','6','9','12','14','17']
experiment_database["Experiment_ID"]=experiment_database["Experiment_ID"].values.astype(str)
allIDs = experiment_database['Experiment_ID'].values
condition = [x in experimentIDs for x in allIDs]
ODdilution_exp_database = experiment_database[condition]
#initialize a dataframe to store values
cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
ODdata = pd.DataFrame([], columns=cols)
# open the nuclei_counts results of each of the experiments we're interested in
commonPath = '/Volumes/JSALAMOS/'
for expID in ODdilution_exp_database['Experiment_ID'].values:
    thisExperiment = ODdilution_exp_database[ODdilution_exp_database['Experiment_ID']==expID]
    microscopeSystem = thisExperiment['System'].values[0]
    date = str(thisExperiment['Date'].values[0])
    resultsSuffix = 'experiment_' + expID + '_nuclei_counts.csv'
    resultsPath = commonPath + filesep + microscopeSystem + filesep + date + filesep + resultsSuffix
    expCountsData = pd.read_csv(resultsPath)
    ODdata = pd.concat([ODdata,expCountsData])
# convert the counts to int64
ODdata = ODdata.astype({"NBFP": int, "NGFP": int, "NRFP": int,"NBoth": int})


# select C58C1 experiments to aggregate. Refer to 'experiment_database' spreadsheet for more info
experimentIDs = ['25','26','27']
experiment_database["Experiment_ID"]=experiment_database["Experiment_ID"].values.astype(str)
allIDs = experiment_database['Experiment_ID'].values
condition = [x in experimentIDs for x in allIDs]
C58ODdilution_exp_database = experiment_database[condition]
#initialize a dataframe to store values
cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
C58ODdata = pd.DataFrame([], columns=cols)
# open the nuclei_counts results of each of the experiments we're interested in
commonPath = '/Volumes/JSALAMOS/'
for expID in C58ODdilution_exp_database['Experiment_ID'].values:
    thisExperiment = C58ODdilution_exp_database[C58ODdilution_exp_database['Experiment_ID']==expID]
    microscopeSystem = thisExperiment['System'].values[0]
    date = str(thisExperiment['Date'].values[0])
    resultsSuffix = 'experiment_' + expID + '_nuclei_counts.csv'
    resultsPath = commonPath + filesep + microscopeSystem + filesep + date + filesep + resultsSuffix
    expCountsData = pd.read_csv(resultsPath)
    C58ODdata = pd.concat([C58ODdata,expCountsData])
# convert the counts to int64
C58ODdata = C58ODdata.astype({"NBFP": int, "NGFP": int, "NRFP": int,"NBoth": int})


# select non-competititive OD titration experiments to aggregate. Refer to 'experiment_database' spreadsheet for more info
experimentIDs = ['28']
experiment_database["Experiment_ID"]=experiment_database["Experiment_ID"].values.astype(str)
allIDs = experiment_database['Experiment_ID'].values
condition = [x in experimentIDs for x in allIDs]
NoComp_exp_database = experiment_database[condition]
#initialize a dataframe to store values
cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
NoCompODdata = pd.DataFrame([], columns=cols)
# open the nuclei_counts results of each of the experiments we're interested in
commonPath = '/Volumes/JSALAMOS/'
for expID in NoComp_exp_database['Experiment_ID'].values:
    thisExperiment = NoComp_exp_database[NoComp_exp_database['Experiment_ID']==expID]
    microscopeSystem = thisExperiment['System'].values[0]
    date = str(thisExperiment['Date'].values[0])
    resultsSuffix = 'experiment_' + expID + '_nuclei_counts.csv'
    resultsPath = commonPath + filesep + microscopeSystem + filesep + date + filesep + resultsSuffix
    expCountsData = pd.read_csv(resultsPath)
    NoCompODdata = pd.concat([NoCompODdata,expCountsData])
# convert the counts to int64
NoCompODdata = NoCompODdata.astype({"NBFP": int, "NGFP": int, "NRFP": int,"NBoth": int})
NoCompODdata['ODtot'] = 0.05


#%% plot the data for each of the total ODs

totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
# totalODs = [0.1, 0.5, 2]
#ODdata = NoCompODdata

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain

ODdata['fracGFP'] = ODdata['NGFP']/ODdata['NBFP']
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/ODdata['NBFP']
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1


for totOD in totalODs:
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1]
        
    # calculate the means and erors for plotting
    MeanPerDilution = thisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerDilution = thisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerDilution['fracGFP']
    RFPMeanPerOD = MeanPerDilution['fracRFP']
    GFPSDPerOD = SDPerDilution['fracGFP']
    RFPSDPerOD = SDPerDilution['fracRFP']
    ODvals = MeanPerDilution.index
    
    # plot stuff
    #plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color="k",mfc='limegreen',mec='black', ms=5)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',color='limegreen',alpha = 0.5,s=30)
    #plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="^", color="k",mfc='orchid',mec='black', ms=6)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracRFP",marker='o',color='orchid',alpha = 0.5,s=30)

    plt.ylabel('fraction of detected nuclei')
    plt.title('Total OD = ' + str(totOD),color='k')
    
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle='-')
    
    plt.xticks(rotation=0)
    plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
    plt.xlim(-3.5,0.5)
    plt.grid()
    plt.xlabel('log10(OD) labeled strain')
    plt.legend(['GFP','RFP'],title = 'mean '+ r"$\pm$" + ' SEM' ,loc='upper right',bbox_to_anchor =(2.65, 1.04))
    plt.show()

#%% plot the data for each of the total ODs, incorporate 'fraction transformable'

totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
#totalODs = [0.1]
fractionTransformable = 0.44

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain

ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1


for totOD in totalODs:
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1]       
    
    # calculate the means and erors for plotting
    MeanPerDilution = thisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerDilution = thisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerDilution['fracGFP']
    RFPMeanPerOD = MeanPerDilution['fracRFP']
    GFPSDPerOD = SDPerDilution['fracGFP']
    RFPSDPerOD = SDPerDilution['fracRFP']
    ODvals = MeanPerDilution.index
    
    # plot stuff
    plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color="k",mfc='limegreen',mec='black', ms=5)
    #sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',color='limegreen',alpha = 0.5,s=40)
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="^", color="k",mfc='orchid',mec='black', ms=6)
    #sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracRFP",marker='o',color='orchid',alpha = 0.5,s=40)
    plt.ylabel('fraction of untransformed \n transformable cells')
    plt.title('Total OD = ' + str(totOD),color='k')
    
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    #ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    
    plt.grid('major')
    #plt.xticks(rotation=)
    plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
    plt.xlim(-3.3,0.5)

    plt.xlabel('log10(OD) labeled strain')
    plt.legend(['GFP','RFP'],title = 'mean '+ r"$\pm$" + ' SEM' ,loc='upper right',bbox_to_anchor =(1.65, 1.04))
    
# # add the no competition experiment on top
  
# NoCompODdata['ODoneStrain'] = NoCompODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
# NoCompODdata['fracGFP'] = NoCompODdata['NGFP']/(NoCompODdata['NBFP']*fractionTransformable)
# NoCompODdata['fracGFP'].loc[NoCompODdata['fracGFP']>1]=1
# NoCompODdata['fracRFP'] = NoCompODdata['NRFP']/(NoCompODdata['NBFP']*fractionTransformable)
# NoCompODdata['fracRFP'].loc[NoCompODdata['fracRFP']>1]=1
# NoCompODdata['NotGFP'] = 1 - NoCompODdata['fracGFP']
# NoCompODdata['NotRFP'] = 1 - NoCompODdata['fracRFP']
# # calculate the means and erors for plotting
# MeanPerDilution = NoCompODdata.groupby(by=["ODoneStrain"]).mean()
# SDPerDilution = NoCompODdata.groupby(by=["ODoneStrain"]).sem()
# NotGFPMeanPerOD = MeanPerDilution['NotGFP']
# NotRFPMeanPerOD = MeanPerDilution['NotRFP']
# NotGFPSDPerOD = SDPerDilution['NotGFP']
# NotRFPSDPerOD = SDPerDilution['NotRFP']
# ODvals = MeanPerDilution.index
# plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD, fmt="o", color="limegreen",mfc='w',mec='limegreen', ms=5)
# plt.errorbar(np.log10(ODvals),NotRFPMeanPerOD,NotRFPSDPerOD, fmt="^", color="orchid",mfc='w',mec='orchid', ms=6)
    

    
    
plt.show()
  
    
  
#%% plot the data for each of the total ODs, incorporate 'fraction transformable' and Poisson fit

totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
#totalODs = [0.1, 0.5, 2]
fractionTransformable = 0.44

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])


ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1

# to store fit results
GFPFittedAlphas = np.zeros(len(totalODs))
GFPFittedAlphaErrors = np.zeros(len(totalODs))
GFPFittedAlphaRsqrd = np.zeros(len(totalODs))
RFPFittedAlphas = np.zeros(len(totalODs))
RFPFittedAlphaErrors = np.zeros(len(totalODs))
RFPFittedAlphaRsqrd = np.zeros(len(totalODs))


for i, totOD in enumerate(totalODs):
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1] 
    
    # calculate the means and erors for plotting data
    MeanPerDilution = thisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerDilution = thisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerDilution['fracGFP']
    RFPMeanPerOD = MeanPerDilution['fracRFP']
    GFPSDPerOD = SDPerDilution['fracGFP']
    RFPSDPerOD = SDPerDilution['fracRFP']
    ODvals = MeanPerDilution.index
    
    # now do the fitting to the Poisson prediction
    FitX = thisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = thisODdata['fracGFP']
    FitYR = thisODdata['fracRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    # first, fit GFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
    fitYG = fractionTransformed(FitX, fit_alphaG)
    fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)    
    # now, fit  RFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
    fitYR = fractionTransformed(FitX, fit_alphaR)
    fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
    GFPRsqrd = getRsquared(fitYG, FitYG)
    RFPRsqrd = getRsquared(fitYR, FitYR)


    # plot fits
    plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
    plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)
    # store fit results
    GFPFittedAlphas[i] = fit_alphaG
    GFPFittedAlphaErrors[i] = alphaG_error
    RFPFittedAlphas[i] = fit_alphaR
    RFPFittedAlphaErrors[i] = alphaR_error
    GFPFittedAlphaRsqrd[i] = GFPRsqrd
    RFPFittedAlphaRsqrd[i] = RFPRsqrd
    
    # plot data
    plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color="limegreen",mfc='limegreen',mec='black', ms=5)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',color='limegreen',alpha = 0.4,s=40)
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="^", color="orchid",mfc='orchid',mec='black', ms=6)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracRFP",marker='^',color='orchid',alpha = 0.4,s=40)
    
    plt.ylabel('fraction of transformed \n transformable cells')
    plt.title('Total OD = ' + str(totOD),color='k')
    
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    #ax.grid(which='minor', color='#CCCCCC', linestyle='-')
    
    plt.grid()
    plt.xticks(rotation=0)
    plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
    plt.xlim(-3.3,0.5)
    plt.grid()
    plt.xlabel('log10(OD) labeled strain')
    plt.legend(['GFP Poisson fit','RFP Poisson fit','GFP data','RFP data'],title = 'mean '+ r"$\pm$" + ' SEM' ,
               loc='upper right',bbox_to_anchor =(2.3, 1.04))
    plt.show()



#%% plot the data for each of the total ODs, fit 'fraction transformable' and Poisson 

totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
#totalODs = [0.1, 0.5, 2]

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])


ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP'])
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP'])
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1

# to store fit results
GFPFittedAlphas = np.zeros(len(totalODs))
GFPFittedAlphaErrors = np.zeros(len(totalODs))
GFPFittedAlphaRsqrd = np.zeros(len(totalODs))
GFPFittedFTrans = np.zeros(len(totalODs))
GFPFittedFTransError = np.zeros(len(totalODs))
RFPFittedAlphas = np.zeros(len(totalODs))
RFPFittedAlphaErrors = np.zeros(len(totalODs))
RFPFittedAlphaRsqrd = np.zeros(len(totalODs))
RFPFittedFTrans = np.zeros(len(totalODs))
RFPFittedFTransError = np.zeros(len(totalODs))



for i, totOD in enumerate(totalODs):
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1] 
    
    # calculate the means and erors for plotting data
    MeanPerDilution = thisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerDilution = thisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerDilution['fracGFP']
    RFPMeanPerOD = MeanPerDilution['fracRFP']
    GFPSDPerOD = SDPerDilution['fracGFP']
    RFPSDPerOD = SDPerDilution['fracRFP']
    ODvals = MeanPerDilution.index
    
    # now do the fitting to the Poisson prediction
    FitX = thisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = thisODdata['fracGFP']
    FitYR = thisODdata['fracRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    # first, fit GFP
    FitBounds = ((0,0.4),(200,0.5)) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionTransformed2, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    fit_fractionG = poptG[1]
    
    alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
    FracTransG_error = np.round(np.sqrt(np.diag(pcov))[1],2)
    
    fitYG = fractionTransformed2(FitX, fit_alphaG,fit_fractionG)
    fitY_contG = fractionTransformed2(xForFit_cont, fit_alphaG,fit_fractionG)
    
    # now, fit  RFP
    FitBounds = ((0,0.4),(200,0.5)) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionTransformed2, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    fit_fractionR = poptR[1]
    alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
    FracTransR_error = np.round(np.sqrt(np.diag(pcov))[1],2)
    fitYR = fractionTransformed2(FitX, fit_alphaR,fit_fractionR)
    fitY_contR = fractionTransformed2(xForFit_cont, fit_alphaR,fit_fractionR) 


    # plot fits
    plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
    plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)
    # store fit results
    GFPFittedAlphas[i] = fit_alphaG
    GFPFittedAlphaErrors[i] = alphaG_error
    GFPFittedFTrans[i] = fit_fractionG
    GFPFittedFTransError[i] = FracTransG_error
    RFPFittedAlphas[i] = fit_alphaR
    RFPFittedAlphaErrors[i] = alphaR_error
    GFPFittedAlphaRsqrd[i] = GFPRsqrd
    RFPFittedAlphaRsqrd[i] = RFPRsqrd
    RFPFittedFTrans[i] = fit_fractionR
    RFPFittedFTransError[i] = FracTransR_error
    
    # plot data
    plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color="limegreen",mfc='limegreen',mec='black', ms=5)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',color='limegreen',alpha = 0.4,s=40)
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="^", color="orchid",mfc='orchid',mec='black', ms=6)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracRFP",marker='^',color='orchid',alpha = 0.4,s=40)
    
    plt.ylabel('fraction of transformed \n transformable cells')
    plt.title('Total OD = ' + str(totOD),color='k')
    
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    #ax.grid(which='minor', color='#CCCCCC', linestyle='-')
    
    plt.grid()
    plt.xticks(rotation=0)
    plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
    plt.xlim(-3.3,0.5)
    plt.grid()
    plt.xlabel('log10(OD) labeled strain')
    plt.legend(['GFP Poisson fit','RFP Poisson fit','GFP data','RFP data'],title = 'mean '+ r"$\pm$" + ' SEM' ,
               loc='upper right',bbox_to_anchor =(2.3, 1.04))
    plt.show()



#%% plot the data for each of the total ODs, fit 'fraction transformable' and Poisson on a per plant basis
# whithout pooling together data from different plants

totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
#totalODs=[0.1]
#totalODs = [0.1, 0.5, 2]

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])


ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP'])
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP'])
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1
ODdata['alphaG'] = ''
ODdata['alphaErrorsG'] = ''
ODdata['FtransformableG'] = ''
ODdata['FtransformableErrorsG'] = ''
ODdata['alphaR'] = ''
ODdata['alphaErrorsR'] = ''
ODdata['FtransformableR'] = ''
ODdata['FtransformableErrorsR'] = ''
ODdata['plantID'] = [x[-1] for x in ODdata['plant']]

# to store fit results
GFPFittedAlphas = np.zeros(len(totalODs))
GFPFittedAlphaErrors = np.zeros(len(totalODs))
GFPFittedAlphaRsqrd = np.zeros(len(totalODs))
GFPFittedFTrans = np.zeros(len(totalODs))
GFPFittedFTransError = np.zeros(len(totalODs))
RFPFittedAlphas = np.zeros(len(totalODs))
RFPFittedAlphaErrors = np.zeros(len(totalODs))
RFPFittedAlphaRsqrd = np.zeros(len(totalODs))
RFPFittedFTrans = np.zeros(len(totalODs))
RFPFittedFTransError = np.zeros(len(totalODs))


for i, totOD in enumerate(totalODs):
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1] 
    
    uniquePlants = thisODdata['plantID'].unique()
    
    for plant in uniquePlants:
        
        thisplantData = thisODdata[thisODdata['plantID']==plant]
        
        # calculate the means and erors for plotting data
        MeanPerDilution = thisplantData.groupby(by=["ODoneStrain"]).mean()
        SDPerDilution = thisplantData.groupby(by=["ODoneStrain"]).sem()
        GFPMeanPerOD = MeanPerDilution['fracGFP']
        RFPMeanPerOD = MeanPerDilution['fracRFP']
        GFPSDPerOD = SDPerDilution['fracGFP']
        RFPSDPerOD = SDPerDilution['fracRFP']
        ODvals = MeanPerDilution.index
        
        # now do the fitting to the Poisson prediction
        FitX = thisplantData['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
        FitYG = thisplantData['fracGFP']
        FitYR = thisplantData['fracRFP']
        xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
        # first, fit GFP
        FitBounds = ((0,0),(200,1)) # lower and upper bounds for alpha
        poptG, pcov = scipy.optimize.curve_fit(fractionTransformed2, FitX, FitYG, bounds = FitBounds)
        fit_alphaG = poptG[0]
        fit_fractionG = poptG[1]
        
        alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
        FracTransG_error = np.round(np.sqrt(np.diag(pcov))[1],2)
        
        fitYG = fractionTransformed2(FitX, fit_alphaG,fit_fractionG)
        fitY_contG = fractionTransformed2(xForFit_cont, fit_alphaG,fit_fractionG)
        
        # now, fit  RFP
        FitBounds = ((0,0),(200,1)) # lower and upper bounds for alpha
        poptR, pcov = scipy.optimize.curve_fit(fractionTransformed2, FitX, FitYR, bounds = FitBounds)
        fit_alphaR = poptR[0]
        fit_fractionR = poptR[1]
        alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
        FracTransR_error = np.round(np.sqrt(np.diag(pcov))[1],2)
        fitYR = fractionTransformed2(FitX, fit_alphaR,fit_fractionR)
        fitY_contR = fractionTransformed2(xForFit_cont, fit_alphaR,fit_fractionR) 
    
    
        # plot fits
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)
        plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
        sns.scatterplot(data=thisplantData, x="log10_ODoneStrain", y="fracGFP",marker='o',color='limegreen',alpha = 0.4,s=40)
        plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)
        sns.scatterplot(data=thisplantData, x="log10_ODoneStrain", y="fracRFP",marker='o',color='orchid',alpha = 0.4,s=40)
        plt.title('ODtot '+ str(totOD) + ' - plant ' + plant + '\n alphaG = ' + str(np.round(fit_alphaG,2)) + 
                  '\n alphaR = ' + str(np.round(fit_alphaR,2)))
        plt.ylim(0,1)
        
        # store fit results
        thisODdata.loc[thisODdata['plantID']==plant, 'alphaG'] = fit_alphaG
        thisODdata.loc[thisODdata['plantID']==plant,'alphaErrorsG'] = alphaG_error
        thisODdata.loc[thisODdata['plantID']==plant,'FtransformableG'] = fit_fractionG
        thisODdata.loc[thisODdata['plantID']==plant,'FtransformableErrorsG'] = FracTransG_error
        thisODdata.loc[thisODdata['plantID']==plant,'alphaR'] = fit_alphaR
        thisODdata.loc[thisODdata['plantID']==plant,'alphaErrorsR'] = alphaR_error
        thisODdata.loc[thisODdata['plantID']==plant,'FtransformableR'] = fit_fractionR
        thisODdata.loc[thisODdata['plantID']==plant,'FtransformableErrorsR'] = FracTransR_error
     
    thisODfileNames = thisODdata['filename']
    allFileNames = ODdata['filename']
    condition = allFileNames.isin(thisODfileNames)
    ODdata.loc[condition,:] = thisODdata
    
# # plot results
        
#sns.barplot(data =ODdata, x = 'ODtot', y ='FtransformableG', hue='plantID') 
ODdata = ODdata.apply(pd.to_numeric, errors='ignore')
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
L1 = (ODdata['FtransformableG'] + ODdata['FtransformableR'])/2
plt.hist(L1[~np.isnan(L1[:])],bins=30)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.hist(ODdata['FtransformableG'],color='limegreen')
plt.xlim(0,1)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
L2 = ODdata['FtransformableR']
plt.hist(L2[~np.isnan(L2[:])],bins=10,color='orchid')
plt.xlim(0,1)


MeanPerODtot = ODdata.groupby(by=["ODtot"]).mean()
SDPerODtot = ODdata.groupby(by=["ODtot"]).std()

GFPFittedAlphas = MeanPerODtot['alphaG']
RFPFittedAlphas = MeanPerODtot['alphaR']
GFPFittedAlphaErrors = SDPerODtot['alphaG']
RFPFittedAlphaErrors = SDPerODtot['alphaR']

GFPFittedFracTrans = MeanPerODtot['FtransformableG']
RFPFittedFracTrans = MeanPerODtot['FtransformableR']
GFPFittedFracTransErrors = SDPerODtot['FtransformableG']
RFPFittedFracTransErrors = SDPerODtot['FtransformableR']


# plot the fitted fraction transformable as a function of total OD
ODtots = [0.05,0.1, 0.5, 1, 2, 3]
#ODtots = [0.1, 0.5, 2]

CarsonODs = [0.6]
Carsonalphas = [12.1] #YFP leaf8, RFP leaf 9, YFP leaf 10
CarsonSD = [3.5]

# fit alpha fit means to a line
xForFit = ODtots
yforFitG = np.log(GFPFittedFracTrans)
yforFitR = np.log(RFPFittedFracTrans)
[GreenSlope, GreenYintercept] = np.polyfit(xForFit,yforFitG,1)
[RedSlope, RedYintercept] = np.polyfit(xForFit,yforFitR,1)
xToPlotFit = np.arange(0,3.5,0.01)
yToPlotGreenFit = GreenYintercept + (GreenSlope * xToPlotFit)
yToPlotRedFit = RedYintercept + (RedSlope * xToPlotFit)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.errorbar(ODtots,GFPFittedFracTrans,GFPFittedFracTransErrors, fmt="o", color="k",mfc='limegreen',mec='black', ms=5)
plt.errorbar(ODtots,RFPFittedFracTrans,RFPFittedFracTransErrors, fmt="^", color="k",mfc='orchid',mec='black', ms=6)
# plt.errorbar(CarsonODs[0],Carsonalphas[0],CarsonSD[0], marker='s', fmt="o", color="k",mfc='gold',mec='black', ms=5)
# plt.errorbar(CarsonODs[1],Carsonalphas[1],CarsonSD[1], marker='v',fmt="o", color="k",mfc='orangered',mec='black', ms=6)
# plt.plot(xToPlotFit,np.exp(yToPlotGreenFit),color='limegreen',lw=1)
# plt.plot(xToPlotFit,np.exp(yToPlotRedFit),color='orchid',lw=1)

# Change major ticks to show every 
ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=45)
plt.xlabel('total OD')
plt.ylabel('Fitted Poisson probability ' + r'$\alpha$' + '\n mean number of infection events \n per plant cell per unit OD')
# plt.xscale('log')
#plt.yscale('log')
plt.legend(['exponential fit GFP','exponential fit RFP','GFP fitted '+ r"$\alpha$",'RFP fitted '+ r"$\alpha$",'Carlson et al. 2023'],
           loc='upper right',bbox_to_anchor =(2.1, 1.04))
# plt.xlim(-0.2,3.5)
plt.ylim(0,1)





#%% plot the GFP data for each of the total ODs, incorporate 'fraction transformable' and Poisson fit and raw data
palette =['khaki','limegreen','mediumturquoise','cornflowerblue','mediumorchid','firebrick']


totalODs = [0.05, 0.1, 0.5, 1, 2, 3]
#totalODs = [0.1, 0.5, 2]
fractionTransformable = 0.44

ODdata['ODoneStrain'] = ODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain

ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1


# to store fit reslts
GFPFittedAlphas = np.zeros(len(totalODs))
GFPFittedAlphaErrors = np.zeros(len(totalODs))
RFPFittedAlphas = np.zeros(len(totalODs))
RFPFittedAlphaErrors = np.zeros(len(totalODs))

for i, totOD in enumerate(totalODs):
    thiscolor = palette[i]
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5, 1.5)
    
    thisODdata = ODdata[ODdata['ODtot']==totOD] 
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        thisODdata = thisODdata[thisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        thisODdata = thisODdata[thisODdata.OD != 1] 
    
    # calculate the means and erors for plotting data
    MeanPerDilution = thisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerDilution = thisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerDilution['fracGFP']
    RFPMeanPerOD = MeanPerDilution['fracRFP']
    GFPSDPerOD = SDPerDilution['fracGFP']
    RFPSDPerOD = SDPerDilution['fracRFP']
    ODvals = MeanPerDilution.index
    
    # now do the fitting to the Poisson prediction
    FitX = thisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = thisODdata['fracGFP']
    FitYR = thisODdata['fracRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    # first, fit GFP
    FitBounds = (-0,200) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    alphaG_error = np.round(np.sqrt(np.diag(poptG))[0][0],1)
    fitYG = fractionTransformed(FitX, fit_alphaG)
    fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)    
    # now, fit  RFP
    FitBounds = (-0,200) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    alphaR_error = np.round(np.sqrt(np.diag(poptR))[0][0],1)
    fitYR = fractionTransformed(FitX, fit_alphaR)
    fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
    # plot fits
    plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color=thiscolor,lw=1.5)
    #plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)
    # store fit results
    GFPFittedAlphas[i] = fit_alphaG
    GFPFittedAlphaErrors[i] = alphaG_error
    RFPFittedAlphas[i] = fit_alphaR
    RFPFittedAlphaErrors[i] = alphaR_error
    
    # plot data
    plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color="k",mfc=thiscolor,mec='black', ms=5)
    sns.scatterplot(data=thisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',edgecolor = 'w',color=thiscolor,alpha = 0.4,s=25)
    #plt.errorbar(np.log10(ODvals),NotRFPMeanPerOD,NotRFPSDPerOD, fmt="^", color="k",mfc='orchid',mec='black', ms=6)
    plt.ylabel('fraction of transformed \n transformable cells')
    plt.title('Total OD = ' + str(totOD),color='k')
    
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    #ax.grid(which='minor', color='#CCCCCC', linestyle='-')
    
    plt.grid('major')
    
    plt.xticks(rotation=45)
    plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
    plt.xlim(-3.3,0.5)

    plt.xlabel('log10(OD) labeled strain')
    plt.legend(['GFP Poisson fit','RFP Poisson fit','GFP data','RFP data'],title = 'mean '+ r"$\pm$" + ' SEM' ,
               loc='upper right',bbox_to_anchor =(2.5, 1.04))
    plt.show()



#%% plot the fitted alphas as a function of the total OD

ODtots = [0.05,0.1, 0.5, 1, 2, 3]
#ODtots = [0.1, 0.5, 2]

CarsonODs = [0.6]
Carsonalphas = [12.1] #YFP leaf8, RFP leaf 9, YFP leaf 10
CarsonSD = [3.5]

# fit alpha fit means to a line
xForFit = ODtots
yforFitG = np.log(GFPFittedAlphas)
yforFitR = np.log(RFPFittedAlphas)
[GreenSlope, GreenYintercept] = np.polyfit(xForFit,yforFitG,1)
[RedSlope, RedYintercept] = np.polyfit(xForFit,yforFitR,1)
xToPlotFit = np.arange(0,3.5,0.01)
yToPlotGreenFit = GreenYintercept + (GreenSlope * xToPlotFit)
yToPlotRedFit = RedYintercept + (RedSlope * xToPlotFit)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.errorbar(ODtots,GFPFittedAlphas,GFPFittedAlphaErrors, fmt="o", color="k",mfc='limegreen',mec='black', ms=5)
plt.errorbar(ODtots,RFPFittedAlphas,RFPFittedAlphaErrors, fmt="^", color="k",mfc='orchid',mec='black', ms=6)
# plt.errorbar(CarsonODs[0],Carsonalphas[0],CarsonSD[0], marker='s', fmt="o", color="k",mfc='gold',mec='black', ms=5)
# plt.errorbar(CarsonODs[1],Carsonalphas[1],CarsonSD[1], marker='v',fmt="o", color="k",mfc='orangered',mec='black', ms=6)
plt.plot(xToPlotFit,np.exp(yToPlotGreenFit),color='limegreen',lw=1)
plt.plot(xToPlotFit,np.exp(yToPlotRedFit),color='orchid',lw=1)

# Change major ticks to show every 
ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=45)
plt.xlabel('total OD')
plt.ylabel('Fitted Poisson probability ' + r'$\alpha$' + '\n mean number of infection events \n per plant cell per unit OD')
# plt.xscale('log')
plt.yscale('log')
plt.legend(['exponential fit GFP','exponential fit RFP','GFP fitted '+ r"$\alpha$",'RFP fitted '+ r"$\alpha$",'Carlson et al. 2023'],
           loc='upper right',bbox_to_anchor =(2.1, 1.04))
plt.xlim(0,3.5)
plt.ylim(3,220)


# #%% plot the fitted alphas as a function of the total OD in log

# ODtots = [0.05,0.1, 0.5, 1, 2, 3]

# CarsonODs = [0.6]
# Carsonalphas = [12.1] #YFP leaf8, RFP leaf 9, YFP leaf 10
# CarsonSD = [3.5]

# # fit alpha fit means to a line
# xForFit = ODtots
# yforFitG = np.log(GFPFittedAlphas)
# yforFitR = np.log(RFPFittedAlphas)
# [GreenSlope, GreenYintercept] = np.polyfit(xForFit,yforFitG,1)
# [RedSlope, RedYintercept] = np.polyfit(xForFit,yforFitR,1)
# xToPlotFit = np.arange(0,3.5,0.01)
# yToPlotGreenFit = GreenYintercept + (GreenSlope * xToPlotFit)
# yToPlotRedFit = RedYintercept + (RedSlope * xToPlotFit)

# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# plt.errorbar(ODtots,np.log(GFPFittedAlphas),np.log(GFPFittedAlphaErrors), fmt="o", color="k",mfc='limegreen',mec='black', ms=5)
# plt.errorbar(ODtots,np.log(RFPFittedAlphas),np.log(RFPFittedAlphaErrors), fmt="^", color="k",mfc='orchid',mec='black', ms=6)
# # plt.errorbar(CarsonODs[0],Carsonalphas[0],CarsonSD[0], marker='s', fmt="o", color="k",mfc='gold',mec='black', ms=5)
# # plt.errorbar(CarsonODs[1],Carsonalphas[1],CarsonSD[1], marker='v',fmt="o", color="k",mfc='orangered',mec='black', ms=6)
# plt.plot(xToPlotFit,yToPlotGreenFit,color='limegreen',lw=1)
# plt.plot(xToPlotFit,yToPlotRedFit,color='orchid',lw=1)

# # Change major ticks to show every 
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# #ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# #ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xticks(rotation=45)
# plt.xlabel('total OD')
# plt.ylabel('Fitted Poisson probability ' + r'$\alpha$' + '\n mean number of infection events \n per plant cell per unit OD')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.legend(['exponential fit GFP','exponential fit RFP','GFP fitted '+ r"$\alpha$",'RFP fitted '+ r"$\alpha$",'Carlson et al. 2023'],
#            loc='upper right',bbox_to_anchor =(2.1, 1.04))
# plt.xlim(-0.2,3.5)


#%% plot the fraction transformed for 1 color (GFP in this case) and all total ODs, in the same plot


ODtots = [0.05,0.1, 0.5, 1, 2, 3]
#ODtots = [0.05]
#ODdata = NoCompODdata
palette1 = cm.YlGn(np.linspace(0, 1, len(ODtots)))
palette2 = cm.RdPu(np.linspace(0, 1, len(ODtots)))
palette =['khaki','lightgreen','mediumturquoise','cornflowerblue','mediumorchid','firebrick']
#palette =['limegreen','mediumturquoise','mediumorchid']

markers = ['o','s','^','d','v','<']
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)

for idx, ODtot in enumerate(ODtots):
    thisColor = palette[idx]
    thisMarker = markers[idx]
    ThisODdata = ODdata[ODdata['ODtot']==ODtot]
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if ODtot == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if ODtot == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1] 
    
    ThisODdata['ODoneStrain'] = ThisODdata['OD']/2
    ThisODdata['log10_ODoneStrain'] = np.log10(ThisODdata['ODoneStrain'])
    FitX = ThisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = ThisODdata['fracGFP']
    FitYR = ThisODdata['fracRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    
    #perform the Poisson fit for GFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    alphaG_error = np.round(np.sqrt(np.diag(poptG))[0][0],1)
    fitYG = fractionTransformed(FitX, fit_alphaG)
    fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)
    
    #perform the Poisson fit for RFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    alphaR_error = np.round(np.sqrt(np.diag(poptR))[0][0],1)
    fitYR = fractionTransformed(FitX, fit_alphaR)
    fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
    
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerOD = ThisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerOD['fracGFP']
    RFPMeanPerOD = MeanPerOD['fracRFP']
    GFPSDPerOD = SDPerOD['fracGFP']
    RFPSDPerOD = SDPerOD['fracRFP']
    ODvals = MeanPerOD.index
    
    # now do the plotting itself
    #plt.plot(np.log10(xForFit_cont),fitY_contG,'-',lw=1.5,color=thisColor)
    #sns.scatterplot(data=ThisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',edgecolor = 'w',color=thisColor,alpha = 0.3,s=75)
    #plt.plot(np.log10(xForFit_cont),fitY_contG,'-',lw=1.5,color=thisColor, label='_nolegend_')
    #plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD, fmt="o", color="k",mfc = thisColor, mec='black', ms=5)
    plt.errorbar(np.log10(ODvals),GFPMeanPerOD,GFPSDPerOD, fmt="o", color=palette1[idx],mfc = palette1[idx], mec='black',marker=thisMarker,ms=5,mew=0.5)
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="o", color=palette2[idx],mfc = palette2[idx], mec='black',marker=thisMarker,ms=5,mew=0.5)

plt.plot([],[],'k-')
plt.ylabel('fraction of transformed \n transformable cells')
# plt.title('Poisson fit, total OD = ' + str(ODtot) + '\n'+ r'$\alpha$ GFP = ' + str(np.round(fit_alphaG,2)) + '$\pm$'+str(alphaG_error)+ 
#           '\n'+ r'$\alpha$ RFP = ' + str(np.round(fit_alphaR,2)) + '$\pm$'+str(alphaR_error)+ '\nfraction transformable = '+ str(fractionTransformable),
#           color='k')

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=0)
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.4,0.3)
plt.grid('major')
plt.legend(['Poisson fits','0.05','0.1','0.5','1','2','3'],
            title = 'total OD',loc='upper right',bbox_to_anchor =(1.8, 1.04))
# plt.legend(['Poisson fits','0.1','0.5','2'],
#            title = 'total OD',loc='upper right',bbox_to_anchor =(1.6, 1.04))
plt.xlabel(r"$log_{10}$"  + 'OD GFP strain')
plt.title('GFP')
#plt.yscale('log')
plt.show()

#%% plot the fraction not transformed for 1 color (GFP in this case) and all total ODs, in the same plot WITH RAW DATA


ODtots = [0.05,0.1, 0.5, 1, 2, 3]
palette = cm.YlGn(np.linspace(0, 1, len(ODtots)))
#palette =['khaki','limegreen','mediumturquoise','cornflowerblue','mediumorchid','firebrick']
#palette =['limegreen','mediumturquoise','mediumorchid']

markers = ['o','s','^','d','v','<']
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)

for idx, ODtot in enumerate(ODtots):
    thisColor = palette[idx]
    thisMarker = markers[idx]
    ThisODdata = ODdata[ODdata['ODtot']==ODtot]
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if ODtot == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if ODtot == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1] 
    
    ThisODdata['NotGFP'] = 1 - ThisODdata['fracGFP']
    ThisODdata['NotRFP'] = 1 - ThisODdata['fracRFP']
    ThisODdata['ODoneStrain'] = ThisODdata['OD']/2
    ODdata['log10_ODoneStrain'] = np.log10(ODdata['ODoneStrain'])
    FitX = ThisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = ThisODdata['NotGFP']
    FitYR = ThisODdata['NotRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    
    #perform the Poisson fit for GFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionNotTransformed, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    alphaG_error = np.round(np.sqrt(np.diag(poptG))[0][0],1)
    fitYG = fractionNotTransformed(FitX, fit_alphaG)
    fitY_contG = fractionNotTransformed(xForFit_cont, fit_alphaG)
    
    #perform the Poisson fit for RFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionNotTransformed, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    alphaR_error = np.round(np.sqrt(np.diag(poptR))[0][0],1)
    fitYR = fractionNotTransformed(FitX, fit_alphaR)
    fitY_contR = fractionNotTransformed(xForFit_cont, fit_alphaR)
    
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerOD = ThisODdata.groupby(by=["ODoneStrain"]).sem()
    NotGFPMeanPerOD = MeanPerOD['NotGFP']
    NotRFPMeanPerOD = MeanPerOD['NotRFP']
    NotGFPSDPerOD = SDPerOD['NotGFP']
    NotRFPSDPerOD = SDPerOD['NotRFP']
    ODvals = MeanPerOD.index
    
    # now do the plotting itself
    #plt.plot(np.log10(xForFit_cont),fitY_contG,'-',lw=1.5,color=thisColor)
    plt.plot(np.log10(xForFit_cont),fitY_contR,'-',lw=1.5,color=thisColor, label='_nolegend_')
    sns.scatterplot(data=ThisODdata, x="log10_ODoneStrain", y="NotRFP",marker='o',edgecolor = 'none',color=thisColor,alpha = 0.4,s=35)
    #plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD,marker=thisMarker, color="k",mfc=thiscolor,mec='black', ms=6)
    #plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD, fmt="o", color="k",mfc = thisColor, mec='black', ms=5)
    #plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD, fmt="o", color="k",mfc = thisColor, mec='black',marker=thisMarker,ms=5.5)

plt.plot([],[],'k-')
plt.ylabel('fraction of untransformed \n transformable cells')
# plt.title('Poisson fit, total OD = ' + str(ODtot) + '\n'+ r'$\alpha$ GFP = ' + str(np.round(fit_alphaG,2)) + '$\pm$'+str(alphaG_error)+ 
#           '\n'+ r'$\alpha$ RFP = ' + str(np.round(fit_alphaR,2)) + '$\pm$'+str(alphaR_error)+ '\nfraction transformable = '+ str(fractionTransformable),
#           color='k')

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.grid()
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(-4,-0.5)
# plt.legend(['Poisson fits','0.05','0.1','0.5','1','2','3'],
#            title = 'total OD',loc='upper right',bbox_to_anchor =(1.6, 1.04))
plt.legend(['Poisson fits','0.1','0.5','2'],
           title = 'total OD',loc='upper right',bbox_to_anchor =(1.9, 1.04))
plt.xlabel(r"$log_{10}$"  + 'OD GFP strain')
plt.title('RFP')
#plt.yscale('log')
plt.show()





#%% plot the fraction not transformed for 1 color (RFP in this case) and all total ODs, in the same plot


ODtots = [0.05,0.1, 0.5, 1, 2, 3]
# ODtots = [0.1, 0.5, 2]
# ODdata = C58ODdata
palette = cm.RdPu_r(np.linspace(0, 1, len(ODtots)))
#palette =['khaki','lightgreen','mediumturquoise','cornflowerblue','mediumorchid','firebrick']
#palette =['limegreen','mediumturquoise','mediumorchid']

markers = ['o','s','^','d','v','<']
fig, ax = plt.subplots()
fig.set_size_inches(2.5, 2)

for idx, ODtot in enumerate(ODtots):
    thisColor = palette[idx]
    thisMarker = markers[idx]
    ThisODdata = ODdata[ODdata['ODtot']==ODtot]
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if ODtot == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if ODtot == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1] 
    
    ThisODdata['ODoneStrain'] = ThisODdata['OD']/2
    ThisODdata['log10_ODoneStrain'] = np.log10(ThisODdata['ODoneStrain'])
    FitX = ThisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = ThisODdata['fracGFP']
    FitYR = ThisODdata['fracRFP']
    xForFit_cont = np.logspace(-3.5,-0.2,100) # dense OD values for plotting purposes
    
    #perform the Poisson fit for GFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
    fit_alphaG = poptG[0]
    alphaG_error = np.round(np.sqrt(np.diag(poptG))[0][0],1)
    fitYG = fractionTransformed(FitX, fit_alphaG)
    fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)
    
    #perform the Poisson fit for RFP
    FitBounds = (-100,100) # lower and upper bounds for alpha
    poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
    fit_alphaR = poptR[0]
    alphaR_error = np.round(np.sqrt(np.diag(poptR))[0][0],1)
    fitYR = fractionTransformed(FitX, fit_alphaR)
    fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
    
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerOD = ThisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerOD['fracGFP']
    RFPMeanPerOD = MeanPerOD['fracRFP']
    GFPSDPerOD = SDPerOD['fracGFP']
    RFPSDPerOD = SDPerOD['fracRFP']
    ODvals = MeanPerOD.index
    
    # now do the plotting itself
    #plt.plot(np.log10(xForFit_cont),fitY_contG,'-',lw=1.5,color=thisColor)
    #sns.scatterplot(data=ThisODdata, x="log10_ODoneStrain", y="fracGFP",marker='o',edgecolor = 'w',color=thisColor,alpha = 0.3,s=75)
    plt.plot(np.log10(xForFit_cont),fitY_contR,'-',lw=1.5,color=thisColor, label='_nolegend_')
    #plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD, fmt="o", color="k",mfc = thisColor, mec='black', ms=5)
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="o", color=thisColor,mfc = thisColor, mec='black',marker=thisMarker,ms=5.5)

plt.plot([],[],'k-')
plt.ylabel('fraction of transformed \n transformable cells')
# plt.title('Poisson fit, total OD = ' + str(ODtot) + '\n'+ r'$\alpha$ GFP = ' + str(np.round(fit_alphaG,2)) + '$\pm$'+str(alphaG_error)+ 
#           '\n'+ r'$\alpha$ RFP = ' + str(np.round(fit_alphaR,2)) + '$\pm$'+str(alphaR_error)+ '\nfraction transformable = '+ str(fractionTransformable),
#           color='k')

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=0)
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(-4,-0.5)
plt.grid('major')
plt.legend(['Poisson fits','0.05','0.1','0.5','1','2','3'],
            title = 'total OD',loc='upper right',bbox_to_anchor =(1.6, 1.04))
# plt.legend(['Poisson fits','0.1','0.5','2'],
#            title = 'total OD',loc='upper right',bbox_to_anchor =(1.6, 1.04))
plt.xlabel(r"$log_{10}$"  + 'OD RFP strain')
plt.title('RFP')
#plt.yscale('log')
plt.show()

#%% plot the fraction not transformed for 1 color (GFP in this case) and all total ODs, in the same plot
# use the exponential scaling of alphas to collapse the data

ODtot = np.array([0.05, 0.1, 0.5, 1, 2, 3])
palette =['khaki','limegreen','mediumturquoise','cornflowerblue','mediumorchid','firebrick']
#palette =['limegreen','mediumturquoise','mediumorchid']
markers = ['o','s','^','d','v','<']
YintG = GreenYintercept
YintR = RedYintercept
Mg = GreenSlope
Mr = RedSlope
# Yint = RedYintercept
# M = RedSlope
a = np.exp(YintG+Mg*ODtot)
b = np.exp(YintG)
GreenScalingFactor = a/b

a = np.exp(YintR+Mr*ODtot)
b = np.exp(YintR)
RedScalingFactor = a/b

ODtots = [0.05,0.1, 0.5, 1, 2, 3]
#ODtots=[0.05]
palette = cm.YlGn(np.linspace(0, 1, len(ODtots)))
palette1 = cm.YlGn(np.linspace(0, 1, len(ODtots)))
palette2 = cm.RdPu(np.linspace(0, 1, len(ODtots)))
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)

for idx, ODtot in enumerate(ODtots):
    GreenScaling = GreenScalingFactor[idx]
    RedScaling = RedScalingFactor[idx]    
    thisMarker = markers[idx]
    thisColor = palette[idx]
    thisColor1 = palette1[idx]
    thisColor2 = palette2[idx]
    ThisODdata = ODdata[ODdata['ODtot']==ODtot]
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1] 
    

    ThisODdata['ODoneStrain'] = ThisODdata['OD']/2
    FitX = ThisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = ThisODdata['fracGFP']
    FitYR = ThisODdata['fracRFP']
    
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerOD = ThisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerOD['fracGFP']
    RFPMeanPerOD = MeanPerOD['fracRFP']
    GFPSDPerOD = SDPerOD['fracGFP']
    RFPSDPerOD = SDPerOD['fracRFP']
    ODvals_GFP = GreenScaling*MeanPerOD.index
    ODvals_RFP = RedScaling*MeanPerOD.index
    
    # now do the plotting itself
    plt.errorbar(np.log10(ODvals_GFP),GFPMeanPerOD,GFPSDPerOD, fmt="o", color=thisColor1,mfc = thisColor1, mec='black',marker=thisMarker, ms=5,mew=0.5)
    plt.errorbar(np.log10(ODvals_GFP),RFPMeanPerOD,RFPSDPerOD, fmt="o", color=thisColor2,mfc = thisColor2, mec='black',marker=thisMarker, ms=5,mew=0.5)


#perform the Poisson fit for GFP
scaledAlpha = np.exp(GreenYintercept)
xForFit_cont = np.logspace(-4,-0.2,100) # dense OD values for plotting purposes
fitY_contG = fractionTransformed(xForFit_cont, scaledAlpha)

plt.plot([],[],'k-')
plt.plot(np.log10(xForFit_cont),fitY_contG,'k-',lw=1.5, label='_nolegend_')

plt.ylabel('fraction of untransformed \n transformable cells')
# plt.title('Poisson fit, total OD = ' + str(ODtot) + '\n'+ r'$\alpha$ GFP = ' + str(np.round(fit_alphaG,2)) + '$\pm$'+str(alphaG_error)+ 
#           '\n'+ r'$\alpha$ RFP = ' + str(np.round(fit_alphaR,2)) + '$\pm$'+str(alphaR_error)+ '\nfraction transformable = '+ str(fractionTransformable),
#           color='k')

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=0)
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-4,-0.5)
plt.grid('major')
plt.legend(['Poisson fit','0.05','0.1','0.5','1','2','3'],
           title = 'total OD',loc='upper right',bbox_to_anchor =(1.8, 1.04))
plt.xlabel(r"$log_{10}$"  + 'OD GFP strain (rescaled)')
plt.title('GFP')
plt.show()

#%% plot the fraction not transformed for 1 color (RFP in this case) and all total ODs, in the same plot
# use the exponential scaling of alphas to collapse the data

ODtot = np.array([0.05, 0.1, 0.5, 1, 2, 3])
markers = ['o','s','^','d','v','<']
Yint = RedYintercept
M = RedSlope
# Yint = RedYintercept
# M = RedSlope
a = np.exp(Yint+M*ODtot)
b = np.exp(Yint)
scalingFactor = a/b


ODtots = [0.05,0.1, 0.5, 1, 2, 3]
palette = cm.RdPu(np.linspace(0, 1, len(ODtots)))
fig, ax = plt.subplots()
fig.set_size_inches(2.75, 2)

for idx, ODtot in enumerate(ODtots):
    scaling = scalingFactor[idx]
    thisMarker = markers[idx]
    thisColor = palette[idx,:]
    ThisODdata = ODdata[ODdata['ODtot']==ODtot]
    
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if totOD == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]       
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if totOD == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1] 
    
    ThisODdata['ODoneStrain'] = ThisODdata['OD']/2
    FitX = ThisODdata['ODoneStrain'] # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
    FitYG = ThisODdata['fracGFP']
    FitYR = ThisODdata['fracRFP']
    
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["ODoneStrain"]).mean()
    SDPerOD = ThisODdata.groupby(by=["ODoneStrain"]).sem()
    GFPMeanPerOD = MeanPerOD['fracGFP']
    RFPMeanPerOD = MeanPerOD['fracRFP']
    GFPSDPerOD = SDPerOD['fracGFP']
    RFPSDPerOD = SDPerOD['fracRFP']
    ODvals = scaling*MeanPerOD.index
    
    # now do the plotting itself
    plt.errorbar(np.log10(ODvals),RFPMeanPerOD,RFPSDPerOD, fmt="o", color=thisColor,mfc = thisColor, mec='black',marker=thisMarker, ms=5)


#perform the Poisson fit for GFP
scaledAlpha = np.exp(RedYintercept)
xForFit_cont = np.logspace(-4,-0.2,100) # dense OD values for plotting purposes
fitY_contR = fractionTransformed(xForFit_cont, scaledAlpha)

plt.plot([],[],'k-')
plt.plot(np.log10(xForFit_cont),fitY_contR,'k-',lw=1.5, label='_nolegend_')

plt.ylabel('fraction of untransformed \n transformable cells')
# plt.title('Poisson fit, total OD = ' + str(ODtot) + '\n'+ r'$\alpha$ GFP = ' + str(np.round(fit_alphaG,2)) + '$\pm$'+str(alphaG_error)+ 
#           '\n'+ r'$\alpha$ RFP = ' + str(np.round(fit_alphaR,2)) + '$\pm$'+str(alphaR_error)+ '\nfraction transformable = '+ str(fractionTransformable),
#           color='k')

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=45)
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(-4,-0.5)
plt.grid()
plt.legend(['Poisson fit','0.05','0.1','0.5','1','2','3'],
           title = 'total OD',loc='upper right',bbox_to_anchor =(1.6, 1.04))
plt.xlabel(r"$log_{10}$"  + 'OD RFP strain (rescaled)')
plt.title('RFP')
plt.show()


#%% compare no-competition prediction and observaton
fractionTransformable = 0.45

NoCompODdata['ODoneStrain'] = NoCompODdata['OD']/2 # this is the OD of 1/2 of the labeled bacteria.
# since the labeled GFP and RFP strains are infiltrated at the same OD, this is the OD of each labeled strain

NoCompODdata['fracGFP'] = NoCompODdata['NGFP']/(NoCompODdata['NBFP']*fractionTransformable)
NoCompODdata['fracGFP'].loc[NoCompODdata['fracGFP']>1]=1
NoCompODdata['fracRFP'] = NoCompODdata['NRFP']/(NoCompODdata['NBFP']*fractionTransformable)
NoCompODdata['fracRFP'].loc[NoCompODdata['fracRFP']>1]=1
NoCompODdata['NotGFP'] = 1 - NoCompODdata['fracGFP']
NoCompODdata['NotRFP'] = 1 - NoCompODdata['fracRFP']

MeanPerOD = NoCompODdata.groupby(by=["ODoneStrain"]).mean()
SDPerOD = NoCompODdata.groupby(by=["ODoneStrain"]).sem()
NotGFPMeanPerOD = MeanPerOD['NotGFP']
NotRFPMeanPerOD = MeanPerOD['NotRFP']
NotGFPSDPerOD = SDPerOD['NotGFP']
NotRFPSDPerOD = SDPerOD['NotRFP']
ODvals = MeanPerOD.index



# prediction for GFP
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
alphaG = 110
mG = -0.87
ODs = np.logspace(-3.2,-0.5)

pNots_GFP = np.exp(-alphaG*ODs*np.exp(mG*ODs))
plt.plot(np.log10(ODs),pNots_GFP,'k--')

#plt.plot(ODs,pNots_GFP,'g')
#plt.xscale('log')


# prediction for RFP
alphaR = 110
mR = -0.91
pNots_GFP = np.exp(-alphaR*ODs*np.exp(mR*ODs))
# plt.plot(np.log10(ODs),pNots_GFP,'b--')
# ODs = np.logspace(-3.2,-0.5)
# pNots = np.zeros(len(ODs))

# for i,od in enumerate(ODs):    
#     pNots[i] = np.exp(-alphaR * od * np.exp(mR*od))
    
plt.errorbar(np.log10(ODvals),NotRFPMeanPerOD,NotRFPSDPerOD,fmt="o", color="k",mfc = 'mediumorchid', mec='black',marker='^', ms=6)
plt.errorbar(np.log10(ODvals),NotGFPMeanPerOD,NotGFPSDPerOD,fmt="o", color="k",mfc = 'limegreen', mec='black',marker='o', ms=5)


# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.xticks(rotation=45)
plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(-4,-0.5)
plt.grid()
plt.ylabel('fraction of cells not transformed')
plt.xlabel('log$_{10}$OD of each reporter strain')
plt.legend(['prediction','GFP','RFP'],
           title = 'total OD',loc='upper right',bbox_to_anchor =(1.7, 1.04))
#plt.xscale('log')
plt.show()

#%%

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
plt.errorbar(ODvals,NotGFPMeanPerOD,NotGFPSDPerOD,fmt="o", color="k",mfc = 'limegreen', mec='black',marker='o', ms=5)
plt.errorbar(ODvals,NotRFPMeanPerOD,NotRFPSDPerOD,fmt="o", color="k",mfc = 'mediumorchid', mec='black',marker='s', ms=5)

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every x
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')


plt.xscale('log')















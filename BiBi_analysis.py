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
from scipy import optimize
from scipy.stats import iqr

# this is to set up the figure style
plt.style.use('default')
# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size']= 9

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

#%% functions 

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
    # alpha = a sclaing factor to go from OD to poisson probability   
    TransfP = 1 - np.exp(-alpha * dilutionOD) # Poisson probability of zero successes
    return TransfP

def fractionTransformed2(dilutionOD,  alpha, fractionTransformable): 
    # dilutionOD = the OD of the culture that we're diluting, the one with the nuclear tag
    # alpha = a sclaing factor to go from OD to poisson probability   
    # fractionTransformable = fraction of cells that can get transformed
    TransfP = fractionTransformable * (1 - np.exp(-alpha * dilutionOD)) # Poisson probability of zero successes
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

#%% load the data

# load the experiment_database spreadsheet
print('navigate to the folder where the experiment database file is stored - then select any file')
file_path = filedialog.askopenfilename() # store the file path as a string
lastFileSep = file_path.rfind(filesep) # find the position of the last path separator
folderpath = file_path[0:lastFileSep] # get the part of the path corresponding to the folder where the chosen file was located
experiment_database_filePath = folderpath + filesep + 'experiment_database.csv'
experiment_database = pd.read_csv(experiment_database_filePath)

# select which experiments to aggregate. Refer to 'experiment_database' spreadsheet for more info
experimentIDs = ['7','8','10','11','13','15','16','36','37']
experiment_database["Experiment_ID"]=experiment_database["Experiment_ID"].values.astype(str)
allIDs = experiment_database['Experiment_ID'].values
condition = [x in experimentIDs for x in allIDs]
BiBi_exp_database = experiment_database[condition]

#initialize a dataframe to store values
cols = ['filename','plant','ODtot','OD','NBFP','NGFP','NRFP','NBoth','meanAvgFluoGFP','sdAvgFluoGFP',
        'meanAvgFluoRFP','sdAvgFluoRFP','meanIntFluoGFP','sdIntFluoGFP','meanIntFluoRFP','sdIntFluoRFP']  
BiBidata = pd.DataFrame([], columns=cols)

# open the nuclei_counts results of each of the experiments we're interested in
commonPath = '/Volumes/JSALAMOS/'
for expID in BiBi_exp_database['Experiment_ID'].values:
    thisExperiment = BiBi_exp_database[BiBi_exp_database['Experiment_ID']==expID]
    microscopeSystem = thisExperiment['System'].values[0]
    date = str(thisExperiment['Date'].values[0])
    resultsSuffix = 'experiment_' + expID + '_nuclei_counts.csv'
    resultsPath = commonPath + filesep + microscopeSystem + filesep + date + filesep + resultsSuffix
    expCountsData = pd.read_csv(resultsPath)
    BiBidata = pd.concat([BiBidata,expCountsData])

# convert the counts to int64
BiBidata = BiBidata.astype({"NBFP": int, "NGFP": int, "NRFP": int,"NBoth": int})



#%% now select the 'non-BiBi' experiments in case we want to compare them to BiBi data
fractionTransformable = 0.44 # fraction of all nuclei that can get transformed
experimentIDs = ['2','5']
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
ODdata['ODoneStrain'] = ODdata['OD']/2
ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1
ODdata['NotGFP'] = 1 - ODdata['fracGFP']
ODdata['NotRFP'] = 1 - ODdata['fracRFP']
ODdata['ObsPBoth'] = ODdata['NBoth']/(ODdata['NBFP']*fractionTransformable)
ODdata['ObsPBoth'].loc[ODdata['ObsPBoth']>1]=1
ODdata['RedGivenGreen'] = ODdata['NBoth']/ODdata['NGFP'] # what fraction of those that express GFP, also express RFP
ODdata['GreenGivenRed'] = ODdata['NBoth']/ODdata['NRFP'] # what fraction of those that express RFP, also express GFP
ODdata['expBoth'] = ODdata['fracGFP'] * ODdata['fracRFP']



# calculate the means and erors for plotting
MeanPerDilution = ODdata.groupby(by=["ODoneStrain"]).mean()
SDPerDilution = ODdata.groupby(by=["ODoneStrain"]).sem()
NotGFPMeanPerOD = MeanPerDilution['NotGFP']
NotRFPMeanPerOD = MeanPerDilution['NotRFP']
NotGFPSDPerOD = SDPerDilution['NotGFP']
NotRFPSDPerOD = SDPerDilution['NotRFP']
ODvals = MeanPerDilution.index

#%% calculate stuff for plotting

fractionTransformable = 0.44 # fraction of all nuclei that can get transformed

BiBidata['fracGFP'] = BiBidata['NGFP']/(BiBidata['NBFP']*fractionTransformable)
BiBidata['fracGFP'].loc[BiBidata['fracGFP']>1]=1
BiBidata['fracRFP'] = BiBidata['NRFP']/(BiBidata['NBFP']*fractionTransformable)
BiBidata['fracRFP'].loc[BiBidata['fracRFP']>1]=1
BiBidata['ObsPBoth'] = BiBidata['NBoth']/(BiBidata['NBFP']*fractionTransformable)
BiBidata['ObsPBoth'].loc[BiBidata['ObsPBoth']>1]=1

BiBidata['fracEither'] = (BiBidata['fracRFP'] + BiBidata['fracGFP']) - BiBidata['ObsPBoth']
BiBidata['NotGFP'] = 1 - BiBidata['fracGFP']
BiBidata['NotRFP'] = 1 - BiBidata['fracRFP']
BiBidata['NotBoth'] = 1 - BiBidata['ObsPBoth']
BiBidata['fracGFPOnly'] = BiBidata['fracGFP'] - BiBidata['ObsPBoth']
BiBidata['fracRFPOnly'] = BiBidata['fracRFP'] - BiBidata['ObsPBoth']
BiBidata['fracGFP2'] = BiBidata['fracGFP']/BiBidata['fracEither'] # prob of FFP given that cell was transformed
BiBidata['fracRFP2'] = BiBidata['fracRFP']/BiBidata['fracEither'] # prob of RFP given that cell was transformed

BiBidata['RedGivenGreen'] = BiBidata['NBoth']/BiBidata['NGFP'] # what fraction of those that express GFP, also express RFP
BiBidata['GreenGivenRed'] = BiBidata['NBoth']/BiBidata['NRFP'] # what fraction of those that express RFP, also express GFP
BiBidata['expBoth'] = BiBidata['fracGFP'] * BiBidata['fracRFP']


BiBi_656_614_data = BiBidata[BiBidata['filename'].str.contains('BiBi656')]
BiBi_656_614_data['ODoneStrain'] = BiBi_656_614_data['OD']

Mix_656_614_data = BiBidata[(BiBidata['filename'].str.contains('sep656') & ~BiBidata['filename'].str.contains('654'))]
Mix_656_614_data['ODoneStrain'] = Mix_656_614_data['OD']/2

BiBi_654_514_data = BiBidata[BiBidata['filename'].str.contains('BiBi654')]
BiBi_654_514_data['ODoneStrain'] = BiBi_654_514_data['OD']

Mix_654_514_data = BiBidata[BiBidata['filename'].str.contains('sep654')]
Mix_654_514_data['ODoneStrain'] = Mix_654_514_data['OD']/2

Mix_654_656_data = BiBidata[(BiBidata['filename'].str.contains('654') & BiBidata['filename'].str.contains('sep656'))]
Mix_654_656_data['ODoneStrain'] = Mix_654_656_data['OD']/2

meanBiBi_656_614 = BiBi_656_614_data.groupby('ODoneStrain').mean()
errorBiBi_656_614 = BiBi_656_614_data.groupby('ODoneStrain').sem()

meanMix_656_614 = Mix_656_614_data.groupby('ODoneStrain').mean()
errorMix_656_614 = Mix_656_614_data.groupby('ODoneStrain').sem()

meanBiBi_654_514 = BiBi_654_514_data.groupby('ODoneStrain').mean()
errorBiBi_654_514 = BiBi_654_514_data.groupby('ODoneStrain').sem()

meanMix_654_514 = Mix_654_514_data.groupby('ODoneStrain').mean()
errorMix_654_514 = Mix_654_514_data.groupby('ODoneStrain').sem()

meanMix_654_656 = Mix_654_656_data.groupby('ODoneStrain').mean()
errorMix_654_656 = Mix_654_656_data.groupby('ODoneStrain').sem()

#%% plots for GFP pVS1 Kan (614) ; RFP BBR1 Spec (656) (656_614)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
ODvals = meanBiBi_656_614.index
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanBiBi_656_614['fracGFP'], errorBiBi_656_614['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5)
#sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='^',color='orchid',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanBiBi_656_614['fracRFP'], errorBiBi_656_614['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6)


# now do the fitting to the Poisson prediction
FitX = ODvals # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG = meanBiBi_656_614['fracGFP']
FitYR = meanBiBi_656_614['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
# first, fit GFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
fit_alphaG = np.round(poptG[0],2)
alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)    
# now, fit  RFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
fit_alphaR = np.round(poptR[0],2)
alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)



# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD BiBi strain')
plt.ylabel('fraction of cells transformed ')
plt.legend(['GFP (pVS1 Kan) ' + str(fit_alphaG) + '$\pm$' + str(alphaG_error),
            'RFP (BBR1 Spec) '+ str(fit_alphaR) + '$\pm$' + str(alphaR_error)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
plt.title('BiBi strain \n total OD constant 0.5')
plt.show()

#%% plots for RFP pVS1 Kan (514) ; GFP BBR1 Spec (654) (656_614)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
ODvals = meanBiBi_656_614.index
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanBiBi_654_514['fracGFP'], errorBiBi_654_514['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5)
#sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='^',color='orchid',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanBiBi_654_514['fracRFP'], errorBiBi_654_514['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6)

# now do the fitting to the Poisson prediction
FitX = ODvals # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG = meanBiBi_654_514['fracGFP']
FitYR = meanBiBi_654_514['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
# first, fit GFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
fit_alphaG = np.round(poptG[0],2)
alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)    
# now, fit  RFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
fit_alphaR = np.round(poptR[0],2)
alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD BiBi strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['GFP (BBR1 Spec) ' + str(fit_alphaG) + '$\pm$' + str(alphaG_error),
            'RFP (pVS1 Kan) '+ str(fit_alphaR) + '$\pm$' + str(alphaR_error)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
plt.title('BiBi strain \n total OD constant 0.5')
plt.show()



#%% plots for coinfiltration of RFP BBR1 Kan (654) and GFP BBR1 Spec (656)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
ODvals = meanMix_654_656.index
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanMix_654_656['fracGFP'], errorMix_654_656['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5)
#sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='^',color='orchid',alpha = 0.6,s=60)
plt.errorbar(np.log10(ODvals), meanMix_654_656['fracRFP'], errorMix_654_656['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6)

# now do the fitting to the Poisson prediction
FitX = ODvals # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG = meanMix_654_656['fracGFP']
FitYR = meanMix_654_656['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
# first, fit GFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
fit_alphaG = np.round(poptG[0],2)
alphaG_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG)    
# now, fit  RFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
fit_alphaR = np.round(poptR[0],2)
alphaR_error = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD BiBi strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['GFP (BBR1 Spec) ' + str(fit_alphaG) + '$\pm$' + str(alphaG_error),
            'RFP (BBR1 Spec) '+ str(fit_alphaR) + '$\pm$' + str(alphaR_error)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
plt.title('coinfiltation of two BBR1 strains \n total OD constant 0.5')
plt.show()
#%% plots for RFP pVS1 Kan (514) ; GFP BBR1 Spec (654) (656_614)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
ODvals = meanBiBi_656_614.index
plt.errorbar(np.log10(ODvals), meanBiBi_654_514['fracGFP'], errorBiBi_654_514['fracGFP'],ls='none',marker='D',mfc='limegreen',mec='k',color='k', ms=5)
plt.errorbar(np.log10(ODvals), meanBiBi_654_514['fracRFP'], errorBiBi_654_514['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6)
plt.errorbar(np.log10(ODvals), meanBiBi_656_614['fracGFP'], errorBiBi_656_614['fracGFP'],ls='none',marker='v',mfc='limegreen',mec='k',color='k', ms=6)
plt.errorbar(np.log10(ODvals), meanBiBi_656_614['fracRFP'], errorBiBi_656_614['fracRFP'],ls='none',marker='s',mfc='orchid',mec='k',color='k', ms=5)

# now do the fitting to the Poisson prediction for the 654_514 BiBi
FitX = ODvals # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG1 = meanBiBi_654_514['fracGFP']
FitYR1 = meanBiBi_654_514['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
# first, fit GFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG1, bounds = FitBounds)
fit_alphaG1 = np.round(poptG[0],2)
alphaG_error1 = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG1)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG1)    
# now, fit  RFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR1, bounds = FitBounds)
fit_alphaR1 = np.round(poptR[0],2)
alphaR_error1 = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR1)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR1)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)
# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)

# now do the fitting to the Poisson prediction for the 656_614 BiBi
FitX = ODvals # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG2 = meanBiBi_656_614['fracGFP']
FitYR2 = meanBiBi_656_614['fracRFP']
# first, fit GFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG2, bounds = FitBounds)
fit_alphaG2 = np.round(poptG[0],2)
alphaG_error2 = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG2)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG2)    
# now, fit  RFP
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR2, bounds = FitBounds)
fit_alphaR2 = np.round(poptR[0],2)
alphaR_error2 = np.round(np.sqrt(np.diag(pcov))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR2)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR2)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)
# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='limegreen',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR,'-',color='orchid',lw=1.5)



# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD BiBi strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['GFP (BBR1 Spec) ' + str(fit_alphaG1) + '$\pm$' + str(alphaG_error1),
            'RFP (pVS1 Kan) '+ str(fit_alphaR1) + '$\pm$' + str(alphaR_error1),
           'GFP (pVS1 Kan) ' + str(fit_alphaG2) + '$\pm$' + str(alphaG_error2),
           'RFP (BBR1 Spec) '+ str(fit_alphaR2) + '$\pm$' + str(alphaR_error2)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
plt.title('BiBi strain \n total OD constant 0.5')
plt.show()


#%% plot GFP pVS1 Kan from BiBi (614) vs GFP pVS1 Kan single binary vector strain (614 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['fracGFP'], errorBiBi_656_614['fracGFP'],ls='none',marker='s',mfc='g',mec='k',color='k', ms=5)
plt.errorbar(np.log10(MeanPerDilution.index), MeanPerDilution['fracGFP'], SDPerDilution['fracGFP'],ls='none',marker='D',mfc='turquoise',mec='k',color='k', ms=5)


# now do the fitting to the Poisson prediction for 656_614 GFP
FitX1 = meanBiBi_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG1 = meanBiBi_656_614['fracGFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX1, FitYG1, bounds = FitBounds)
fit_alphaG1 = np.round(poptG1[0],2)
alphaG_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYG1 = fractionTransformed(FitX, fit_alphaG1)
fitY_contG1 = fractionTransformed(xForFit_cont, fit_alphaG1)    
# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = MeanPerDilution.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG2 = MeanPerDilution['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYG2, bounds = FitBounds)
fit_alphaG2 = np.round(poptG2[0],2)
alphaG_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYG2 = fractionTransformed(FitX2, fit_alphaG2)
fitY_contG2 = fractionTransformed(xForFit_cont, fit_alphaG2)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG1,'-', color='g',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG2,'-',color='turquoise',lw=1.5)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
#plt.legend(['GFP from BiBi (pVS1 Kan)','GFP from Bi (pVS1 Kan)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.legend(['GFP from BiBi(pVS1 Kan) ' + str(fit_alphaG1) + '$\pm$' + str(alphaG_error1),
            'GFP from Bi (pVS1 Kan) '+ str(fit_alphaG2) + '$\pm$' + str(alphaG_error2)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
plt.title('GFP (pVS1 Kan) \n launched from different strains \n total OD constant 0.5')
plt.show()


#%% plot RFP pVS1 Kan from BiBi (514) vs RFP pVS1 Kan single binary vector strain (514 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_654_514['fracRFP'], errorBiBi_654_514['fracRFP'],ls='none',marker='s',mfc='salmon',mec='k',color='k', ms=5)
plt.errorbar(np.log10(MeanPerDilution.index), MeanPerDilution['fracRFP'], SDPerDilution['fracRFP'],ls='none',marker='D',mfc='mediumorchid',mec='k',color='k', ms=5)


# now do the fitting to the Poisson prediction for 656_614 GFP
FitX1 = meanBiBi_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR1 = meanBiBi_654_514['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR1, bounds = FitBounds)
fit_alphaR1 = np.round(poptR1[0],2)
alphaR_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYR1 = fractionTransformed(FitX, fit_alphaR1)
fitY_contR1 = fractionTransformed(xForFit_cont, fit_alphaR1)    
# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = MeanPerDilution.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR2 = MeanPerDilution['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYR2, bounds = FitBounds)
fit_alphaR2 = np.round(poptR2[0],2)
alphaR_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYR2 = fractionTransformed(FitX2, fit_alphaR2)
fitY_contR2 = fractionTransformed(xForFit_cont, fit_alphaR2)
# GFPRsqrd = getRsquared(fitYG, FitYG)
# RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contR1,'-', color='salmon',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR2,'-',color='mediumorchid',lw=1.5)

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['RFP from BiBi(pVS1 Kan) ' + str(fit_alphaR1) + '$\pm$' + str(alphaR_error1),
            'RFP from Bi (pVS1 Kan) '+ str(fit_alphaR2) + '$\pm$' + str(alphaR_error2)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
#plt.legend(['RFP from BiBi (pVS1 Kan)','RFP from Bi (pVS1 Kan)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.title('RFP (pVS1 Kan) \n launched from different strains \n total OD constant 0.5')
plt.show()



#%% plot GFP BBR1 Kan from BiBi (654) vs GFP BBR1 Kan single binary vector strain (654 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['fracGFP'], errorBiBi_654_514['fracGFP'],ls='none',marker='s',mfc='g',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index), meanMix_654_514['fracGFP'], errorMix_654_514['fracGFP'],ls='none',marker='D',mfc='turquoise',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_654_656.index), meanMix_654_656['fracGFP'], errorMix_654_656['fracGFP'],ls='none',marker='^',mfc='greenyellow',mec='k',color='k', ms=5)


# now do the fitting to the Poisson prediction for 656_614 GFP
FitX1 = meanBiBi_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG1 = meanBiBi_654_514['fracGFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX1, FitYG1, bounds = FitBounds)
fit_alphaG1 = np.round(poptG1[0],2)
alphaG_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYG1 = fractionTransformed(FitX1, fit_alphaG1)
fitY_contG1 = fractionTransformed(xForFit_cont, fit_alphaG1) 
   
# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = meanMix_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG2 = meanMix_654_514['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYG2, bounds = FitBounds)
fit_alphaG2 = np.round(poptG2[0],2)
alphaG_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYG2 = fractionTransformed(FitX2, fit_alphaG2)
fitY_contG2 = fractionTransformed(xForFit_cont, fit_alphaG2)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

FitX3 = meanMix_654_656.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG3 = meanMix_654_656['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG3, pcov3 = scipy.optimize.curve_fit(fractionTransformed, FitX3, FitYG3, bounds = FitBounds)
fit_alphaG3 = np.round(poptG3[0],2)
alphaG_error3 = np.round(np.sqrt(np.diag(pcov3))[0],1)
fitYG3 = fractionTransformed(FitX3, fit_alphaG3)
fitY_contG3 = fractionTransformed(xForFit_cont, fit_alphaG3)
GFPRsqrd = getRsquared(fitYG, FitYG)
RFPRsqrd = getRsquared(fitYR, FitYR)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG1,'-', color='g',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG2,'-',color='turquoise',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG3,'-',color='greenyellow',lw=1.5)

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['GFP from BiBi(BBR1 Spec) ' + str(fit_alphaG1) + '$\pm$' + str(alphaG_error1),
            'GFP from co (BBR1 GFP Spec ; pVS1 RFP Kan) '+ str(fit_alphaG2) + '$\pm$' + str(alphaG_error2),
            'GFP from co (BBR1 GFP Spec ; BBR1 RFP Spec) '+ str(fit_alphaG3) + '$\pm$' + str(alphaG_error3)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
#plt.legend(['GFP from BiBi (BBR1 Spec)','GFP from Bi (BBR1 Spec)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.title('GFP (BBR1 Spec) \n launched from different strains \n total OD constant 0.5')
plt.show()


#%% plot RFP BBR1 Kan from BiBi (656) vs RFP BBR1 Kan single binary vector strain (656 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['fracRFP'], errorBiBi_656_614['fracRFP'],ls='none',marker='s',mfc='salmon',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index), meanMix_656_614['fracRFP'], errorMix_656_614['fracRFP'],ls='none',marker='D',mfc='mediumorchid',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_654_656.index), meanMix_654_656['fracRFP'], errorMix_654_656['fracRFP'],ls='none',marker='^',mfc='red',mec='k',color='k', ms=5)

# now do the fitting to the Poisson prediction for 656_614 GFP
FitX1 = meanBiBi_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR1 = meanBiBi_656_614['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR1, bounds = FitBounds)
fit_alphaR1 = np.round(poptR1[0],2)
alphaR_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYR1 = fractionTransformed(FitX, fit_alphaR1)
fitY_contR1 = fractionTransformed(xForFit_cont, fit_alphaR1)    
# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = meanMix_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR2 = meanMix_656_614['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYR2, bounds = FitBounds)
fit_alphaR2 = np.round(poptR2[0],2)
alphaR_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYR2 = fractionTransformed(FitX2, fit_alphaR2)
fitY_contR2 = fractionTransformed(xForFit_cont, fit_alphaR2)
# GFPRsqrd = getRsquared(fitYG, FitYG)
# RFPRsqrd = getRsquared(fitYR, FitYR)
FitX3 = meanMix_654_656.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR3 = meanMix_654_656['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR3, pcov3 = scipy.optimize.curve_fit(fractionTransformed, FitX3, FitYR3, bounds = FitBounds)
fit_alphaR3 = np.round(poptR3[0],2)
alphaR_error3 = np.round(np.sqrt(np.diag(pcov3))[0],1)
fitYR3 = fractionTransformed(FitX3, fit_alphaR3)
fitY_contR3 = fractionTransformed(xForFit_cont, fit_alphaR3)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contR1,'-', color='salmon',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR2,'-',color='mediumorchid',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contR3,'-',color='red',lw=1.5)

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['RFP from BiBi(BBR1 Spec) ' + str(fit_alphaR1) + '$\pm$' + str(alphaR_error1),
            'RFP from co (BBR1 RFP Spec ; pVS1 GFP Kan) '+ str(fit_alphaR2) + '$\pm$' + str(alphaR_error2),
            'RFP from co (BBR1 RFP Spec ; BBR1 GFP Spec) '+ str(fit_alphaR3) + '$\pm$' + str(alphaR_error3)],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))
#plt.legend(['RFP from BiBi (BBR1 Spec)','RFP from Bi (BBR1 Spec)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.title('RFP (BBR1 Spec) \n launched from different strains \n total OD constant 0.5')
plt.show()




#%% plot GFP pVS1 Kan from BiBi (614), GFP BBR1 Kan from BiBi (654),  
# GFP pVS1 Kan single binary vector strain (614 Bi), and GFP BBR1 Spec single binary vector strain (654 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2.75, 2.5)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(MeanPerDilution.index), MeanPerDilution['fracGFP'], SDPerDilution['fracGFP'],ls='none',marker='>',mfc='orange',mec='k',color='orange', ms=5)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['fracGFP'], errorBiBi_656_614['fracGFP'],ls='none',marker='s',mfc='b',mec='k',color='b', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index), meanMix_656_614['fracGFP'], errorMix_656_614['fracGFP'],ls='none',marker='^',mfc='khaki',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['fracGFP'], errorBiBi_654_514['fracGFP'],ls='none',marker='D',mfc='r',mec='k',color='r', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index), meanMix_654_514['fracGFP'], errorMix_654_514['fracGFP'],ls='none',marker='<',mfc='c',mec='k',color='c', ms=5)
plt.errorbar(np.log10(meanMix_654_656.index), meanMix_654_656['fracGFP'], errorMix_654_656['fracGFP'],ls='none',marker='^',mfc='greenyellow',mec='k',color='k', ms=5)




# now do the fitting to the Poisson prediction for 514 614 GFP
FitX = MeanPerDilution.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG = MeanPerDilution['fracGFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYG, bounds = FitBounds)
fit_alphaG = np.round(poptG[0],2)
alphaG_error = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYG = fractionTransformed(FitX, fit_alphaG)
fitY_contG = fractionTransformed(xForFit_cont, fit_alphaG) 

# now do the fitting to the Poisson prediction for 656_614 GFP
FitX1 = meanBiBi_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG1 = meanBiBi_656_614['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX1, FitYG1, bounds = FitBounds)
fit_alphaG1 = np.round(poptG1[0],2)
alphaG_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYG1 = fractionTransformed(FitX1, fit_alphaG1)
fitY_contG1 = fractionTransformed(xForFit_cont, fit_alphaG1) 
   
# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = meanMix_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG2 = meanMix_656_614['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYG2, bounds = FitBounds)
fit_alphaG2 = np.round(poptG2[0],2)
alphaG_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYG2 = fractionTransformed(FitX2, fit_alphaG2)
fitY_contG2 = fractionTransformed(xForFit_cont, fit_alphaG2)

# now do the fitting to the Poisson prediction for 654_514 GFP
FitX3 = meanBiBi_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG3 = meanBiBi_654_514['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG3, pcov3 = scipy.optimize.curve_fit(fractionTransformed, FitX3, FitYG3, bounds = FitBounds)
fit_alphaG3 = np.round(poptG3[0],2)
alphaG_error3 = np.round(np.sqrt(np.diag(pcov3))[0],1)
fitYG3 = fractionTransformed(FitX3, fit_alphaG3)
fitY_contG3 = fractionTransformed(xForFit_cont, fit_alphaG3)

# now, fit  GFP pVS1 Kan single binary vector strain (614 Bi)
FitX4 = meanMix_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG4 = meanMix_654_514['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG4, pcov4 = scipy.optimize.curve_fit(fractionTransformed, FitX4, FitYG4, bounds = FitBounds)
fit_alphaG4 = np.round(poptG4[0],2)
alphaG_error4 = np.round(np.sqrt(np.diag(pcov4))[0],1)
fitYG4 = fractionTransformed(FitX4, fit_alphaG4)
fitY_contG4 = fractionTransformed(xForFit_cont, fit_alphaG4)

FitX5 = meanMix_654_656.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG5 = meanMix_654_656['fracGFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG5, pcov5 = scipy.optimize.curve_fit(fractionTransformed, FitX5, FitYG5, bounds = FitBounds)
fit_alphaG5 = np.round(poptG5[0],2)
alphaG_error5 = np.round(np.sqrt(np.diag(pcov5))[0],1)
fitYG5 = fractionTransformed(FitX5, fit_alphaG5)
fitY_contG5 = fractionTransformed(xForFit_cont, fit_alphaG5)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='orange',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG1,'-', color='b',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG2,'-',color='khaki',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG3,'-',color='r',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG4,'-',color='c',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG5,'-',color='greenyellow',lw=1.5)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
#plt.xlim(0.001,0.5)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['GFP from co (pVS1 GFP Kan + pVS1 RFP Kan) '+ str(fit_alphaG) + '$\pm$' + str(alphaG_error),
            'GFP from BiBi (pVS1 GFP Kan ; BBR1 RFP Spec) ' + str(fit_alphaG1) + '$\pm$' + str(alphaG_error1),
            'GFP from co (pVS1 GFP Kan + BBR1 RFP Spec) '+ str(fit_alphaG2) + '$\pm$' + str(alphaG_error2),
           'GFP from BiBi (BBR1 GFP Spec ; pVS1 RFP Kan) ' + str(fit_alphaG3) + '$\pm$' + str(alphaG_error3),
           'GFP from co (BBR1 GFP Spec + pVS1 RFP Kan) '+ str(fit_alphaG4) + '$\pm$' + str(alphaG_error4),
           'GFP from co (BBR1 GFP Spec + BBR1 RFP Spec) '+ str(fit_alphaG5) + '$\pm$' + str(alphaG_error5),
           ],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))


#plt.legend(['GFP from BiBi (pVS1 Kan)','GFP from Bi (pVS1 Kan)','GFP from BiBi (BBR1 Spec)','GFP from Bi (BBR1 Spec)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.title('GFP \n launched from different strains/plasmids \n total OD constant 0.5')
plt.show()


#%% plot RFP pVS1 Kan from BiBi (514), RFP BBR1 Kan from BiBi (656),  
# RFP pVS1 Kan single binary vector strain (514 Bi), and RFP BBR1 Spec single binary vector strain (656 Bi)

fig, ax = plt.subplots()
fig.set_size_inches(2.75, 2.5)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(MeanPerDilution.index), MeanPerDilution['fracRFP'], SDPerDilution['fracRFP'],ls='none',marker='>',mfc='orange',mec='k',color='orange', ms=5)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['fracRFP'], errorBiBi_654_514['fracRFP'],ls='none',marker='D',mfc='b',mec='k',color='b', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index), meanMix_654_514['fracRFP'], errorMix_654_514['fracRFP'],ls='none',marker='<',mfc='khaki',mec='k',color='khaki', ms=5)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['fracRFP'], errorBiBi_656_614['fracRFP'],ls='none',marker='s',mfc='r',mec='k',color='r', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index), meanMix_656_614['fracRFP'], errorMix_656_614['fracRFP'],ls='none',marker='^',mfc='c',mec='k',color='c', ms=5)
plt.errorbar(np.log10(meanMix_654_656.index), meanMix_654_656['fracRFP'], errorMix_654_656['fracRFP'],ls='none',marker='^',mfc='greenyellow',mec='k',color='greenyellow', ms=5)


# now do the fitting to the Poisson prediction for 514 614 RFP
FitX = MeanPerDilution.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR = MeanPerDilution['fracRFP']
xForFit_cont = np.logspace(-3.15,-0.15,100) # dense OD values for plotting purposes
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX, FitYR, bounds = FitBounds)
fit_alphaR = np.round(poptR[0],2)
alphaR_error = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYR = fractionTransformed(FitX, fit_alphaR)
fitY_contR = fractionTransformed(xForFit_cont, fit_alphaR) 

# now do the fitting to the Poisson prediction for 656_614 RFP
FitX1 = meanBiBi_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR1 = meanBiBi_654_514['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR1, pcov1 = scipy.optimize.curve_fit(fractionTransformed, FitX1, FitYR1, bounds = FitBounds)
fit_alphaR1 = np.round(poptR1[0],2)
alphaR_error1 = np.round(np.sqrt(np.diag(pcov1))[0],1)
fitYR1 = fractionTransformed(FitX1, fit_alphaR1)
fitY_contR1 = fractionTransformed(xForFit_cont, fit_alphaR1) 
   
# now, fit  RFP pVS1 Kan single binary vector strain (614 Bi)
FitX2 = meanMix_654_514.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYR2 = meanMix_654_514['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptR2, pcov2 = scipy.optimize.curve_fit(fractionTransformed, FitX2, FitYR2, bounds = FitBounds)
fit_alphaR2 = np.round(poptR2[0],2)
alphaR_error2 = np.round(np.sqrt(np.diag(pcov2))[0],1)
fitYR2 = fractionTransformed(FitX2, fit_alphaR2)
fitY_contR2 = fractionTransformed(xForFit_cont, fit_alphaR2)

# now do the fitting to the Poisson prediction for 654_514 RFP
FitX3 = meanBiBi_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG3 = meanBiBi_656_614['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG3, pcov3 = scipy.optimize.curve_fit(fractionTransformed, FitX3, FitYG3, bounds = FitBounds)
fit_alphaR3 = np.round(poptG3[0],2)
alphaR_error3 = np.round(np.sqrt(np.diag(pcov3))[0],1)
fitYG3 = fractionTransformed(FitX3, fit_alphaR3)
fitY_contG3 = fractionTransformed(xForFit_cont, fit_alphaR3)

# now, fit  RFP pVS1 Kan single binary vector strain (614 Bi)
FitX4 = meanMix_656_614.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG4 = meanMix_656_614['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG4, pcov4 = scipy.optimize.curve_fit(fractionTransformed, FitX4, FitYG4, bounds = FitBounds)
fit_alphaR4 = np.round(poptG4[0],2)
alphaR_error4 = np.round(np.sqrt(np.diag(pcov4))[0],1)
fitYG4 = fractionTransformed(FitX4, fit_alphaG4)
fitY_contG4 = fractionTransformed(xForFit_cont, fit_alphaR4)

FitX5 = meanMix_654_656.index # ****  IMPORTANT!  ***** the OD in the dataframe is the mix of green and red strains
FitYG5 = meanMix_654_656['fracRFP']
FitBounds = (-100,100) # lower and upper bounds for alpha
poptG5, pcov5 = scipy.optimize.curve_fit(fractionTransformed, FitX5, FitYG5, bounds = FitBounds)
fit_alphaR5 = np.round(poptG5[0],2)
alphaR_error5 = np.round(np.sqrt(np.diag(pcov5))[0],1)
fitYG5 = fractionTransformed(FitX5, fit_alphaR5)
fitY_contG5 = fractionTransformed(xForFit_cont, fit_alphaR5)

# plot fits
plt.plot(np.log10(xForFit_cont),fitY_contG,'-', color='orange',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG1,'-', color='b',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG2,'-',color='khaki',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG3,'-',color='r',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG4,'-',color='c',lw=1.5)
plt.plot(np.log10(xForFit_cont),fitY_contG5,'-',color='greenyellow',lw=1.5)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.xlim(-3.15,-0.15)
#plt.xlim(0.001,0.5)
plt.grid('major')
#plt.xscale('log')
plt.xlabel ('$log_{10}$ OD of labeled strain')
plt.ylabel('fraction of cells transformed')
plt.legend(['RFP from co (pVS1 RFP Kan + pVS1 RFP Kan) '+ str(fit_alphaR) + '$\pm$' + str(alphaR_error),
            'RFP from BiBi (pVS1 RFP Kan ; BBR1 RFP Spec) ' + str(fit_alphaR1) + '$\pm$' + str(alphaR_error1),
            'RFP from co (pVS1 RFP Kan + BBR1 GFP Spec) '+ str(fit_alphaR2) + '$\pm$' + str(alphaR_error2),
           'RFP from BiBi (BBR1 RFP Spec ; pVS1 GFP Kan) ' + str(fit_alphaR3) + '$\pm$' + str(alphaR_error3),
           'RFP from co (BBR1 RFP Spec + pVS1 GFP Kan) '+ str(fit_alphaR4) + '$\pm$' + str(alphaR_error4),
           'RFP from co (BBR1 RFP Spec + BBR1 GFP Spec) '+ str(fit_alphaR5) + '$\pm$' + str(alphaR_error5),
           ],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1.05, 1))


#plt.legend(['RFP from BiBi (pVS1 Kan)','RFP from Bi (pVS1 Kan)','RFP from BiBi (BBR1 Spec)','RFP from Bi (BBR1 Spec)'],title = 'mean $\pm$ SEM',bbox_to_anchor =(2.35, 1))
plt.title('RFP \n launched from different strains/plasmids \n total OD constant 0.5')
plt.show()

#%%
meanGFPalphas = [fit_alphaG,fit_alphaG1,fit_alphaG2,fit_alphaG3,fit_alphaG4,fit_alphaG5]
errorGFPalphas = [alphaG_error,alphaG_error1,alphaG_error2,alphaG_error3,alphaG_error4,alphaG_error5]
meanRFPalphas = [fit_alphaR,fit_alphaR1,fit_alphaR2,fit_alphaR3,fit_alphaR4,fit_alphaR5]
errorRFPalphas = [alphaR_error,alphaR_error1,alphaR_error2,alphaR_error3,alphaR_error4,alphaR_error5]

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(20))

# Change minor ticks to show every 5. (20/4 = 5)
#ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

#plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(0.001,0.5)
plt.grid('major')

plt.errorbar(np.arange(len(meanGFPalphas)), meanGFPalphas, errorGFPalphas,ls='none',marker='o',mfc='limegreen',mec='k',color='limegreen', ms=6)
plt.errorbar(np.arange(len(meanGFPalphas))+0.25, meanRFPalphas, errorRFPalphas,ls='none',marker='^',mfc='orchid',mec='k',color='orchid', ms=7)
xtickNames = ['pVS1 coinfiltration','BiBi pVS1-RFP BBR1-GFP','coinfiltration pVS1-RFP BBR1-GFP','BiBi BBR1-RFP pVS1-GFP','coinfiltration BBR1-RFP pVS1-GFP','coinfiltration BBR1']
plt.xticks(np.arange(len(meanGFPalphas)),xtickNames, rotation = 90)
plt.ylabel('alpha')

#%% fraction expressing both green and red

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['ObsPBoth'], errorBiBi_656_614['ObsPBoth'],ls='none',marker='s',mfc='b',mec='k',color='b', ms=5)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['ObsPBoth'], errorBiBi_654_514['ObsPBoth'],ls='none',marker='D',mfc='r',mec='k',color='r', ms=5)
plt.errorbar(np.log10(MeanPerDilution.index*2), MeanPerDilution['ObsPBoth'], SDPerDilution['ObsPBoth'],ls='none',marker='^',mfc='khaki',mec='k',color='khaki', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index*2), meanMix_654_514['ObsPBoth'], errorMix_654_514['ObsPBoth'],ls='none',marker='<',mfc='palegreen',mec='k',color='palegreen', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_656_614['ObsPBoth'], errorMix_656_614['ObsPBoth'],ls='none',marker='>',mfc='lightpink',mec='k',color='lightpink', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_654_656['ObsPBoth'], errorMix_656_614['ObsPBoth'],ls='none',marker='v',mfc='lightsteelblue',mec='k',color='lightsteelblue', ms=5)


#plt.plot(ODvals,Probs_both_per_OD,'k') #this was generated by 'contransformation_model' script

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
#ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='-')

#plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(0.001,0.5)
plt.grid('major')

#plt.xscale('log')
plt.yscale('log')
plt.xlabel ('$log_{10}$ OD of labeled bacteria')
plt.ylabel('fraction of co-transformed \n transformable cells')
plt.legend(['BiBi (GFP pVS1; RFP BBR1)','BiBi (RFP pVS1; GFP BBR1)','GFP Bi (pVS1) + RFP Bi (pVS1)','RFP Bi (pVS1) + GFP Bi (BBR1)','GFP Bi (pVS1) + RFP Bi (BBR1)','GFP Bi (BBR1) + RFP Bi (BBR1)'],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(3, 1))
plt.title('cotransformation frequency \n total OD constant 0.5')
plt.xlim(-2.85,-0.1)
plt.show()



#%% how many greens also express red, and viceversa
# RedGivenGreen and GreenGivenRed

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['RedGivenGreen'], errorBiBi_656_614['RedGivenGreen'],ls='none',marker='s',mfc='b',mec='k',color='b', ms=5)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['RedGivenGreen'], errorBiBi_654_514['RedGivenGreen'],ls='none',marker='D',mfc='r',mec='k',color='r', ms=5)
plt.errorbar(np.log10(MeanPerDilution.index*2), MeanPerDilution['RedGivenGreen'], SDPerDilution['RedGivenGreen'],ls='none',marker='^',mfc='khaki',mec='k',color='khaki', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index*2), meanMix_654_514['RedGivenGreen'], errorMix_654_514['RedGivenGreen'],ls='none',marker='<',mfc='palegreen',mec='k',color='palegreen', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_656_614['RedGivenGreen'], errorMix_656_614['RedGivenGreen'],ls='none',marker='>',mfc='lightpink',mec='k',color='lightpink', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_654_656['RedGivenGreen'], errorMix_656_614['RedGivenGreen'],ls='none',marker='v',mfc='lightsteelblue',mec='k',color='lightsteelblue', ms=5)


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

#plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
#plt.xlim(0.001,0.5)
plt.grid()
#plt.xscale('log')
plt.yscale('log')
plt.xlabel ('$log_{10}$ OD of labeled bacteria')
plt.ylabel('fraction of GFP cells')
plt.legend(['BiBi (GFP pVS1; RFP BBR1)','BiBi (RFP pVS1; GFP BBR1)','GFP Bi (pVS1) + RFP Bi (pVS1)','RFP Bi (pVS1) + GFP Bi (BBR1)','GFP Bi (pVS1) + RFP Bi (BBR1)','GFP Bi (BBR1) + RFP Bi (BBR1)'],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(3, 1))
plt.title('fraction of GFP cells that are also RFP \n total OD constant 0.5')
plt.show()

#%% how many greens also express red, and viceversa
# RedGivenGreen and GreenGivenRed

fig, ax = plt.subplots()
fig.set_size_inches(2, 2)
# plot all the data points
#sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['GreenGivenRed'], errorBiBi_656_614['RedGivenGreen'],ls='none',marker='s',mfc='cornflowerblue',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['GreenGivenRed'], errorBiBi_654_514['RedGivenGreen'],ls='none',marker='^',mfc='mediumorchid',mec='k',color='k', ms=5)
plt.errorbar(np.log10(MeanPerDilution.index*2), MeanPerDilution['GreenGivenRed'], SDPerDilution['RedGivenGreen'],ls='none',marker='o',mfc='lightsalmon',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_654_514.index*2), meanMix_654_514['GreenGivenRed'], errorMix_654_514['RedGivenGreen'],ls='none',marker='<',mfc='limegreen',mec='k',color='k', ms=5)
plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_656_614['GreenGivenRed'], errorMix_656_614['RedGivenGreen'],ls='none',marker='>',mfc='gold',mec='k',color='k', ms=5)


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

#plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
plt.ylim(0.01,1.2)
plt.grid()
#plt.xscale('log')
plt.yscale('log')
plt.xlabel ('$log_{10}$ OD of labeled bacteria')
plt.ylabel('fraction of RFP cells')
plt.legend(['BiBi (GFP pVS1; RFP BBR1)','BiBi (RFP pVS1; GFP BBR1)','GFP Bi (pVS1) + RFP Bi (pVS1)','RFP Bi (pVS1) + GFP Bi (BBR1)','GFP Bi (pVS1) + RFP Bi (BBR1)'],
           title = 'mean $\pm$ SEM',bbox_to_anchor =(1, 1))
plt.title('fraction of RFP cells that are also GFP \n total OD constant 0.5')
plt.show()

#%%

# plt.errorbar(np.log10(meanBiBi_656_614.index), meanBiBi_656_614['ObsPBoth'], errorBiBi_656_614['ObsPBoth'],ls='none',marker='s',mfc='b',mec='k',color='k', ms=5)
# plt.errorbar(np.log10(meanBiBi_654_514.index), meanBiBi_654_514['ObsPBoth'], errorBiBi_654_514['ObsPBoth'],ls='none',marker='D',mfc='r',mec='k',color='k', ms=5)
# plt.errorbar(np.log10(MeanPerDilution.index*2), MeanPerDilution['ObsPBoth'], SDPerDilution['ObsPBoth'],ls='none',marker='^',mfc='khaki',mec='k',color='k', ms=5)
# plt.errorbar(np.log10(meanMix_654_514.index*2), meanMix_654_514['ObsPBoth'], errorMix_654_514['ObsPBoth'],ls='none',marker='<',mfc='palegreen',mec='k',color='k', ms=5)
# plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_656_614['ObsPBoth'], errorMix_656_614['ObsPBoth'],ls='none',marker='>',mfc='lightpink',mec='k',color='k', ms=5)
# plt.errorbar(np.log10(meanMix_656_614.index*2), meanMix_654_656['ObsPBoth'], errorMix_656_614['ObsPBoth'],ls='none',marker='v',mfc='lightsteelblue',mec='k',color='k', ms=5)
fig, ax = plt.subplots()
fig.set_size_inches(2, 2)


plt.plot(ODdata['ObsPBoth'],ODdata['expBoth'],'^',color='khaki',ms=5,mec='k',mew=0.5,alpha=0.65)
plt.plot(Mix_654_514_data['ObsPBoth'],Mix_654_514_data['expBoth'],'<',color='palegreen',ms=5,mec='k',mew=0.5,alpha=0.65)
plt.plot(Mix_656_614_data['ObsPBoth'],Mix_656_614_data['expBoth'],'>',color='lightpink',ms=5,mec='k',mew=0.5,alpha=0.65)
plt.plot(Mix_654_656_data['ObsPBoth'],Mix_654_656_data['expBoth'],'v',color='lightsteelblue',ms=5,mec='k',mew=0.5,alpha=0.65)
plt.plot(BiBi_654_514_data['ObsPBoth'],BiBi_654_514_data['expBoth'],'D',color='r',ms=5,mec='k',mew=0.5,alpha=0.65)
plt.plot(BiBi_656_614_data['ObsPBoth'],BiBi_656_614_data['expBoth'],'s',color='b',ms=5,mec='k',mew=0.5,alpha=0.65)

ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='-')
#ax.grid(which='minor', color='#CCCCCC', linestyle='-')

plt.ylabel('expected frequency of expressing both \n if independent')
plt.xlabel('observed frequency \n expressing both')

plt.xlim(0.001,1)
plt.ylim(0.001,1)

plt.yscale('log')
plt.xscale('log')


# meanBiBi_656_614 = BiBi_656_614_data.groupby('ODoneStrain').mean()
# errorBiBi_656_614 = BiBi_656_614_data.groupby('ODoneStrain').sem()

# meanMix_656_614 = Mix_656_614_data.groupby('ODoneStrain').mean()
# errorMix_656_614 = Mix_656_614_data.groupby('ODoneStrain').sem()

# meanBiBi_654_514 = BiBi_654_514_data.groupby('ODoneStrain').mean()
# errorBiBi_654_514 = BiBi_654_514_data.groupby('ODoneStrain').sem()

# meanMix_654_514 = Mix_654_514_data.groupby('ODoneStrain').mean()
# errorMix_654_514 = Mix_654_514_data.groupby('ODoneStrain').sem()

# meanMix_654_656 = Mix_654_656_data.groupby('ODoneStrain').mean()
# errorMix_654_656 = Mix_654_656_data.groupby('ODoneStrain').sem()

















# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['ObsPBoth'], sdBiBiData['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='v',mfc='yellow',mec='k',color='k', ms=6,capsize=3.5)
# #plt.plot(meanBiBiData.index,meanBiBiData['ObsPBoth']*0.1,'r-')
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# #plt.ylim(-0.1,1.1)#np.min(FitYG)*0.1
# plt.xlim(0.001,1)
# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of labeled strains')
# plt.ylabel('fraction of cells \n expressing both GFP and RFP')
# plt.legend(['BiBi','mix'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(2.1,1.1))
# plt.title('mix+BiBi data from 10/23/23 + BiBi from 11/20/23 \n RFP pVS1 Kan ; GFP BBR1 Spec \n total OD constant 0.5')

# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)

# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracGFP'], sdBiBiData['fracGFP'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# plt.errorbar(meanMixData.index/2, meanMixData['fracGFP'], sdMixData['fracGFP'],ls='none',marker='^',mfc='yellow',mec='k',color='k', ms=6,capsize=3.5)

# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of GFP strain or BiBi strain')
# plt.ylabel('fraction of cells \n expressing GFP')
# plt.legend(['BiBi','mix'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1.04))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n RFP pVS1 Kan ; GFP BBR1 Spec \n total OD constant 0.5')

# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracRFP'], sdBiBiData['fracRFP'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# plt.errorbar(meanMixData.index/2, meanMixData['fracRFP'], sdMixData['fracRFP'],ls='none',marker='^',mfc='yellow',mec='k',color='k', ms=6,capsize=3.5)

# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of RFP strain or BiBi strain')
# plt.ylabel('fraction of cells \n expressing RFP')
# plt.legend(['BiBi','mix'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1.04))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n RFP pVS1 Kan ; GFP BBR1 Spec \n total OD constant 0.5')


# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['ObsPBoth'], sdBiBiData['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=6,capsize=3.5)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracGFP'], sdBiBiData['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# #plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='^',mfc='salmon',mec='b',color='k', ms=11,capsize=5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of BiBi strain')
# plt.ylabel('fraction of cells')
# plt.legend(['BiBi expressing both','BiBi expressing GFP'],title = 'mean $\pm$ sd across plants', bbox_to_anchor =(1, 1.04))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n RFP pVS1 Kan ; GFP BBR1 Spec \n total OD constant 0.5')
# plt.show()

# #%%
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['ObsPBoth'], sdBiBiData['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=6,capsize=3.5)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracRFP'], sdBiBiData['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# #plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='^',mfc='salmon',mec='b',color='k', ms=11,capsize=5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of BiBi strain')
# plt.ylabel('fraction of cells')
# plt.legend(['BiBi expressing both','BiBi expressing RFP'],title = 'mean $\pm$ sd across plants', bbox_to_anchor =(1, 1.04))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n RFP pVS1 Kan ; GFP BBR1 Spec \n total OD constant 0.5')
# plt.show()

# #%% This means that some cells express GFP but not RFP and the opposite is also true, some cells express RFP but not GFP
# # the cells that express GFP but not RFP were contacted by an agro but the RFP plasmid did not get transferred.
# # similarly, a cell expressing RFP but not GFP got contacted by an agro but the GFP plasmid didn't succeed.
# # can we estimate the efficiency of plasmid expression given contact?
# # define 'contact' as the establishment of
# # the probability of transfering the RFP plasmid given contact

# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracGFP2'], sdBiBiData['fracGFP2'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='^',color='orchid',alpha = 0.6,s=60)
# plt.errorbar(meanBiBiData.index, meanBiBiData['fracRFP2'], sdBiBiData['fracRFP2'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6,capsize=3.5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# plt.xlabel ('OD BiBi strain')
# plt.ylabel('fraction of transformed cells \n that are transformed by FP')
# plt.legend(['GFP','RFP'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n BiBi \n GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5')


# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="NotGFP",marker='o',color='limegreen',alpha = 0.6,s=60)
# plt.errorbar(meanBiBiData.index/2, meanMixData['fracGFP2'], sdMixData['fracGFP2'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=BiBiData, x="OD", y="NotRFP",marker='^',color='orchid',alpha = 0.6,s=60)
# plt.errorbar(meanBiBiData.index/2, meanMixData['fracRFP2'], sdMixData['fracRFP2'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6,capsize=3.5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')
# plt.xscale('log')
# plt.xlabel ('OD of each labeled strain')
# plt.ylabel('fraction of transformed cells \n that are transformed by FP')
# plt.legend(['GFP','RFP'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1))
# plt.title('mix data from 10/23/23 + BiBi from 11/20/23 \n Mix \n GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5')



# #%% now do the confocal data
# # load stuff
# confocalDataPath1 = '/Volumes/JSALAMOS/lsm710/2023/10-16-23/Max_projections/AllData3.csv'
# confocalData1 = pd.read_csv(confocalDataPath1)

# confocalDataPath2 = '/Volumes/JSALAMOS/lsm710/2023/11-20-23/Max_projections/AllData3.csv'
# confocalData2 = pd.read_csv(confocalDataPath2)

# confocalData = pd.concat([confocalData1,confocalData2]) # combine them

# fractionTransformable = 0.55 # fraction of all nuclei that can get transformed

# confocalData['fracGFP'] = confocalData['NRFP']/(confocalData['NBFP']*fractionTransformable) #IMPORTANT!! GFP and RFP channels are swapped
# confocalData['fracGFP'].loc[confocalData['fracGFP']>1]=1
# confocalData['fracRFP'] = confocalData['NGFP']/(confocalData['NBFP']*fractionTransformable)
# confocalData['fracRFP'].loc[confocalData['fracRFP']>1]=1
# confocalData['fracEither'] = (confocalData['fracRFP'] + confocalData['fracGFP']) - confocalData['ObsPBoth']
# confocalData['NotGFP'] = 1 - confocalData['fracGFP']
# confocalData['NotRFP'] = 1 - confocalData['fracRFP']
# confocalData['NotBoth'] = 1 - confocalData['ObsPBoth']
# confocalData['ODoneStrain'] = confocalData['OD']/2

# BiBiData_c = confocalData[confocalData['filename'].str.contains('BiBi')]
# MixData_c = confocalData[~ confocalData['filename'].str.contains('BiBi')]

# meanBiBiData_c = BiBiData_c.groupby('OD').mean()
# sdBiBiData_c = BiBiData_c.groupby('OD').std()
# meanMixData_c = MixData_c.groupby('OD').mean()
# sdMixData_c = MixData_c.groupby('OD').std()

# #%% BiBi cells expressing both and mixed cells expressing both
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['ObsPBoth'], sdBiBiData_c['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# plt.errorbar(meanMixData_c.index, meanMixData_c['ObsPBoth'], sdMixData_c['ObsPBoth'],ls='none',marker='^',mfc='yellow',mec='k',color='k', ms=6,capsize=3.5)

# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# plt.yscale('log')
# #plt.ylim(0,0.2)
# #plt.xlim(0.0015, 0.03)
# plt.xlabel ('OD of labeled strains')
# plt.ylabel('fraction of cells \n expressing both GFP and RFP')
# plt.legend(['BiBi','mix'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(2.1, 1.04))
# plt.title('GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5 \n confocal \n BiBi and Mix 10/23/23 + BiBi 11/20/23')


# #%% BiBi cells expressing both and BiBi cells expressing RFP
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['ObsPBoth'], sdBiBiData_c['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['fracGFP'], sdBiBiData_c['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# #plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='^',mfc='salmon',mec='b',color='k', ms=11,capsize=5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of BiBi strain')
# plt.ylabel('fraction of cells')
# plt.legend(['BiBi expressing both','BiBi expressing GFP'],title = 'mean $\pm$ sd across plants', bbox_to_anchor =(1, 1.04))
# plt.title('GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5 \n confocal BiBi and Mix 10/23/23 + BiBi 11/20/23')

# #%% BiBi cells expressing both and BiBi cells expressing RFP
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['ObsPBoth'], sdBiBiData_c['ObsPBoth'],ls='none',marker='s',mfc='royalblue',mec='k',color='k', ms=5,capsize=3.5)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['fracRFP'], sdBiBiData_c['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# #plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='^',mfc='salmon',mec='b',color='k', ms=11,capsize=5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of BiBi strain')
# plt.ylabel('fraction of cells')
# plt.legend(['BiBi expressing both','BiBi expressing RFP'],title = 'mean $\pm$ sd across plants', bbox_to_anchor =(1, 1.04))
# plt.title('GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5 \n confocal')

# #%% BiBi cells expressing both and BiBi cells expressing RFP
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=BiBiData, x="OD", y="ObsPBoth",marker='o',color='blue',alpha = 0.3,s=60)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['fracGFP'], sdBiBiData_c['fracGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=5,capsize=3.5)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['fracRFP'], sdBiBiData_c['fracRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=6,capsize=3.5)
# #sns.scatterplot(data=MixData, x="OD", y="ObsPBoth",marker='^',color='red',alpha = 0.3,s=60)
# #plt.errorbar(meanMixData.index, meanMixData['ObsPBoth'], sdMixData['ObsPBoth'],ls='none',marker='^',mfc='salmon',mec='b',color='k', ms=11,capsize=5)
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel ('OD of BiBi strain')
# plt.ylabel('fraction of cells')
# plt.legend(['BiBi expressing GFP','BiBi expressing RFP'],title = 'mean $\pm$ sd across plants', bbox_to_anchor =(1, 1.04))
# plt.title('GFP pVS1 Kan ; RFP BBR1 Spec \n total OD constant 0.5 \n confocal')

# #%% fluorescence, mixed strains

# fig = plt.figure()
# plt.style.use('ggplot') 
# fig.set_size_inches(4, 4)
# # plot all the data points
# #sns.scatterplot(data=MixData, x="OD", y="meanIntFluoRFP",marker='o',color='orchid',alpha = 0.5,s=60)
# plt.errorbar(meanMixData_c.index, meanMixData_c['meanIntFluoRFP'], sdMixData_c['meanIntFluoRFP'],ls='none',marker='o',mfc='orchid',mec='b',color='k', ms=8,capsize=5)
# plt.errorbar(meanMixData_c.index, meanMixData_c['meanIntFluoGFP'], sdMixData_c['meanIntFluoGFP'],ls='none',marker='o',mfc='limegreen',mec='b',color='k', ms=8,capsize=5)

# plt.xscale('log')
# #plt.ylim
# # plt.yscale('log')
# plt.xlabel ('OD of labeled strains')
# plt.ylabel('nucleus fluorescence')
# plt.legend(['SAPS656 RFP-NLS BBR Spec', 'SAPS614 GFP-NLS pVS1 Kan'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1.04))
# plt.title('average fluorescence of detected nuclei \n mixed strains')

# #%% fluorescence, BiBi 

# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # plot all the data points
# #sns.scatterplot(data=MixData, x="OD", y="meanIntFluoRFP",marker='o',color='orchid',alpha = 0.5,s=60)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['meanIntFluoRFP'], sdBiBiData_c['meanIntFluoRFP'],ls='none',marker='^',mfc='orchid',mec='k',color='k', ms=8,capsize=5)
# plt.errorbar(meanBiBiData_c.index, meanBiBiData_c['meanIntFluoGFP'], sdBiBiData_c['meanIntFluoGFP'],ls='none',marker='o',mfc='limegreen',mec='k',color='k', ms=8,capsize=5)

# plt.xscale('log')
# #plt.ylim
# # plt.yscale('log')
# plt.xlabel ('OD of labeled strains')
# plt.ylabel('nucleus fluorescence')
# plt.legend(['SAPS656 RFP-NLS BBR Spec', 'SAPS614 GFP-NLS pVS1 Kan'],title = 'mean $\pm$ sd across plants',bbox_to_anchor =(1, 1.04))
# plt.title('average fluorescence of detected nuclei \n BiBi strain')



# #%% compare confocal with widefield

# # remove from widefield data the ODs that I didn't test in confocal
# meanBiBiData = meanBiBiData.reset_index()
# meanBiBiData_c = meanBiBiData_c.reset_index()
# meanBiBiData = meanBiBiData[meanBiBiData['OD'] <= np.max(meanBiBiData_c['OD'])]

# meanBiBiData_merged = pd.merge(meanBiBiData, meanBiBiData_c, on = 'OD') # merge widefield (x) and confocal (y)
# # plot and compare
# fig = plt.figure()
# plt.style.use('ggplot') 
# fig.set_size_inches(4, 4)
# plt.plot(meanBiBiData_merged['ObsPBoth_x'],meanBiBiData_merged['ObsPBoth_y'],'ko')
# plt.plot(meanBiBiData_merged['ObsPBoth_x'],meanBiBiData_merged['ObsPBoth_x'],'k-')
# plt.xlabel('Leica widefield')
# plt.ylabel('Zeiss confocal')


# #%% but I want to compare punch by punch, i.e. image by image
# # ODdataConfocal = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/9-19-23/AllData3_confocal.csv')
# # ODdataConfocal['fracGFPconf'] = ODdataConfocal['NGFP']/ODdataConfocal['NBFP']
# # ODdataConfocal['fracRFPconf'] = ODdataConfocal['NRFP']/ODdataConfocal['NBFP']
# # ODdataConfocal['fracEitherconf'] = (ODdataConfocal['fracRFPconf'] + ODdataConfocal['fracGFPconf']) - ODdataConfocal['ObsPBoth']

# # ODdataWideField = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/9-19-23/AllData3.csv')
# # ODdataWideField['fracGFP'] = ODdataWideField['NGFP']/ODdataWideField['NBFP']
# # ODdataWideField['fracRFP'] = ODdataWideField['NRFP']/ODdataWideField['NBFP']
# # ODdataWideField['fracEither'] = (ODdataWideField['fracRFP'] + ODdataWideField['fracGFP']) - ODdataWideField['ObsPBoth']

# ODboth = pd.merge(widefieldData, confocalData, on=['OD','ODtot','plant'])
# ODbothBiBi = ODboth[ODboth['plant'].str.contains('BiBi')]
# ODbothMix = ODboth[~ODboth['plant'].str.contains('BiBi')]
# # keep only the widefiled data with the same ODs as 


# #P = sns.color_palette("husl", 6)
# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # sns.scatterplot(data=ODbothBiBi,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='royalblue',alpha=0.6)
# # sns.scatterplot(data=ODbothMix,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='yellow',alpha=0.6)
# #sns.scatterplot(data=ODboth,x='fracRFPconf',y='fracRFP',hue='ODtot',s=60, palette = P, marker="o")
# plt.plot([0.001,0.5],[0.001,0.5],'k-')
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# #ax.grid(which='minor', color='#CCCCCC', linestyle='-')

# sns.scatterplot(data=ODboth,x='ObsPBoth_x',y='ObsPBoth_y',s=40, marker="s",color='royalblue',alpha=0.6,edgecolor='none')

# plt.xlabel('fraction transformed \n with both (widefiled)')
# plt.ylabel('fraction transformed \n with both (confocal)')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('comparing microscopes \n BiBi and mix combined \n SAPS656 RFP-NLS BBR Spec \n SAPS614 GFP-NLS pVS1 Kan')
# plt.show()



# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # sns.scatterplot(data=ODbothBiBi,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='royalblue',alpha=0.6)
# # sns.scatterplot(data=ODbothMix,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='yellow',alpha=0.6)
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# #ax.grid(which='minor', color='#CCCCCC', linestyle='-')
# sns.scatterplot(data=ODboth,x='fracRFP_x',y='fracRFP_y',s=40, marker="^",color='orchid',alpha=0.6,edgecolor='none')
# #sns.scatterplot(data=ODboth,x='fracRFPconf',y='fracRFP',hue='ODtot',s=60, palette = P, marker="o")
# plt.plot([0.035,1],[0.035,1],'k-')
# plt.xlabel('fraction transformed \n with RFP (widefiled)')
# plt.ylabel('fraction transformed \n with RFP (confocal)')
# plt.xscale('log')
# plt.yscale('log')
# # plt.xlim(0.005,1)
# # plt.ylim(0.005,1)
# plt.title('comparing microscopes \n BiBi and mix combined \n SAPS656 RFP-NLS BBR Spec \n SAPS614 GFP-NLS pVS1 Kan')
# plt.show()



# fig, ax = plt.subplots()
# fig.set_size_inches(2, 2)
# # sns.scatterplot(data=ODbothBiBi,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='royalblue',alpha=0.6)
# # sns.scatterplot(data=ODbothMix,x='ObsPBoth_x',y='ObsPBoth_y',s=80, marker="o",color='yellow',alpha=0.6)
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

# # Change minor ticks to show every 5. (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# # Turn grid on for both major and minor ticks and style minor slightly
# # differently.
# ax.grid(which='major', color='#CCCCCC', linestyle='-')
# sns.scatterplot(data=ODboth,x='fracGFP_x',y='fracGFP_y',s=40, marker="o",color='limegreen',alpha=0.6,edgecolor='none')
# #sns.scatterplot(data=ODboth,x='fracRFPconf',y='fracRFP',hue='ODtot',s=60, palette = P, marker="o")
# plt.plot([0.0015,1.1],[0.0015,1.1],'k-')

# plt.xlabel('fraction transformed \n with GFP (widefiled)')
# plt.ylabel('fraction transformed \n with GFP (confocal)')
# plt.xscale('log')
# plt.yscale('log')
# # plt.xlim(0.0015,1.1)
# # plt.ylim(0.0015,1.1)
# plt.title('comparing microscopes \n BiBi and mix combined \n SAPS656 RFP-NLS BBR Spec \n SAPS614 GFP-NLS pVS1 Kan')
# plt.show()














#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:06:06 2023

@author: simon_alamos
"""
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import optimize
from scipy.stats import iqr

#%% fit to tissue-level expression using Mitch's data
# load Mitch data
datapath = '/Users/simon_alamos/Documents/Shih_lab/Data/Mitch_data/simon_raw_od.xlsx'
MTODdata = pd.read_excel(open(datapath, 'rb'), sheet_name='promoter_data')

dataPath = '/Users/simon_alamos/Documents/Shih_lab/Data/Mitch_data/6-12-23/2023-6-12_gv_od.csv'
MTODdata = pd.read_csv(dataPath)

#%% look at the data

# look at one promoter first
PC4data = MTODdata[MTODdata['Strain'].str.contains('PC4')]
MeanPerOD4 = PC4data.groupby(by=["OD"]).median()
SDPerOD4 = PC4data.groupby(by=["OD"]).std()
Q1PerOD4 = PC4data.groupby(by=["OD"]).quantile(0.25)#-MeanPerOD4 #lower interquartile range
Q3PerOD4 = PC4data.groupby(by=["OD"]).quantile(0.75) #upper interquartile range

# look at one promoter first
PC5data = MTODdata[MTODdata['Strain'].str.contains('PC5')]
MeanPerOD5 = PC5data.groupby(by=["OD"]).median()
SDPerOD5 = PC5data.groupby(by=["OD"]).std()
Q1PerOD5 = PC5data.groupby(by=["OD"]).quantile(0.25) #lower interquartile range
Q3PerOD5 = PC5data.groupby(by=["OD"]).quantile(0.75) #upper interquartile range


# look at one promoter first
PC6data = MTODdata[MTODdata['Strain'].str.contains('PC6')]
MeanPerOD6 = PC6data.groupby(by=["OD"]).median()
SDPerOD6 = PC6data.groupby(by=["OD"]).std()
Q1PerOD6 = PC6data.groupby(by=["OD"]).quantile(0.25) #lower interquartile range
Q3PerOD6 = PC6data.groupby(by=["OD"]).quantile(0.75) #upper interquartile range



fig = plt.figure()
fig.set_size_inches(4, 4)
sns.scatterplot(data=PC4data, x="OD", y="GFP",color='b',alpha = 0.1,s=55)
#plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'],yerr=np.array([Q1PerOD4['GFP'], Q3PerOD4['GFP']]), fmt="o", color="b",mfc='white',mec='black', ms=6)

sns.scatterplot(data=PC5data, x="OD", y="GFP",color='orange',alpha = 0.1,s=55)
#plt.errorbar(MeanPerOD5.index, MeanPerOD5['GFP'], SDPerOD5['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.errorbar(MeanPerOD5.index, MeanPerOD5['GFP'],yerr=np.array([Q1PerOD5['GFP'], Q3PerOD5['GFP']]), fmt="o", color="orange",mfc='white',mec='black', ms=6)


sns.scatterplot(data=PC6data, x="OD", y="GFP",color='green',alpha = 0.1,s=55)
#plt.errorbar(MeanPerOD6.index, MeanPerOD6['GFP'], SDPerOD6['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.errorbar(MeanPerOD6.index, MeanPerOD6['GFP'],yerr=np.array([Q1PerOD6['GFP'], Q3PerOD6['GFP']]), fmt="o", color="g",mfc='white',mec='black', ms=6)


#plt.legend(labelNames,title="infiltration",bbox_to_anchor =(1, 1.04))
plt.title('pC-GFP OD titration') 
plt.ylabel('GFP')
plt.xlabel('agrobacterium OD$_{600}$')
plt.xscale('log')
plt.yscale('log')
#plt.legend(['PC4','','PC5','','PC6',''])
plt.show()

#%% fit to a model

# a useful function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# define the objective function
def tissueExpression2param(OD, ODsat, Ng): # here we group together N and g as a single parameter
    # N = number of trials
    # OD = the culture OD
    # ODsat = OD at which the probability of success becomes 1
    # g = expression per success,  i.e. how many AUs you get per transformation event
    sat_idx = find_nearest(OD,ODsat)# index of the OD closest to ODsat
    OD0 = OD[0:sat_idx]
    OD1 = np.ones(np.size(OD[sat_idx:]))
    
    gfp0 = Ng*(OD0/ODsat)
    gfp1 = Ng*OD1
    return np.concatenate([gfp0,gfp1])

def tissueExpression3param(OD, ODsat, N,g): # we leave N and g to change independently
    # N = number of trials
    # OD = the culture OD
    # ODsat = OD at which the probability of success becomes 1
    # g = expression per success, i.e. how many AUs you get per transformation event
    sat_idx = find_nearest(OD,ODsat)# index of the OD closest to ODsat
    OD0 = OD[0:sat_idx]
    OD1 = np.ones(np.size(OD[sat_idx:]))
    
    gfp0 = N*g*(OD0/ODsat)
    gfp1 = N*g*OD1
    return np.concatenate([gfp0,gfp1])


# will try fitting the means of PC4 first
xForFit = MeanPerOD4.index
yForFit = MeanPerOD4['GFP']
FitBounds = ([0.1,100000],[2,2000000]) # lower and upper bounds for [ODsat, Ng]
popt, pcov = scipy.optimize.curve_fit(tissueExpression2param, xForFit, yForFit,bounds = FitBounds)
fit_ODsat = popt[0]
fit_Ng = popt[1]
fitY = tissueExpression2param(xForFit, fit_ODsat, fit_Ng)
xForFit_cont = np.logspace(-2,0.7,100)
fitY_cont = tissueExpression2param(xForFit_cont, fit_ODsat, fit_Ng)

# look at the data and the fit
fig = plt.figure()
fig.set_size_inches(4, 4)
sns.scatterplot(data=PC4data, x="OD", y="GFP",color='b',alpha = 0.1,s=55)
plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.plot(xForFit_cont,fitY_cont,'b-')
plt.xscale('log')
plt.yscale('log')
plt.title('2 parameter fit')

xForFit = MeanPerOD4.index
yForFit = MeanPerOD4['GFP']
FitBounds = ([0.1,4,10000],[2,10,2000000]) # lower and upper bounds for [ODsat, N, g]
popt, pcov = scipy.optimize.curve_fit(tissueExpression3param, xForFit, yForFit,bounds = FitBounds)
fit_ODsat = popt[0]
fit_N = popt[1]
fit_g = popt[2]
fit3Y = tissueExpression3param(xForFit, fit_ODsat, fit_N, fit_g)
xForFit_cont = np.logspace(-2,0.7,100)
fitY_3cont = tissueExpression3param(xForFit_cont, fit_ODsat, fit_N, fit_g)

# look at the data and the fit 
fig = plt.figure()
fig.set_size_inches(4, 4)
sns.scatterplot(data=PC4data, x="OD", y="GFP",color='b',alpha = 0.1,s=55)
plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.plot(xForFit_cont,fitY_3cont,'b-')
plt.xscale('log')
plt.yscale('log')
plt.title('3 parameter fit')


#%% now try fitting all the datapoints, not just the means

# gather and sort the data. Not sure if sorting is necessary...
xForFit_unsorted = PC4data['OD']
yForFit_unsorted = PC4data['GFP']
sortOrder = np.argsort(xForFit_unsorted)
FitX = xForFit_unsorted.reset_index(drop=True)[sortOrder] # I need to do the reset index for the sorted indexes to work
FitY = yForFit_unsorted.reset_index(drop=True)[sortOrder]

xForFit_cont = np.logspace(-2,0.7,100)

#perform the 2 parameter fit
FitBounds = ([0.1,100000],[2,2000000]) # lower and upper bounds for [ODsat, Ng]
popt, pcov = scipy.optimize.curve_fit(tissueExpression2param, FitX, FitY,bounds = FitBounds)
fit_ODsat = popt[0]
fit_Ng = popt[1]
fitY = tissueExpression2param(xForFit, fit_ODsat, fit_Ng)
fitY_cont = tissueExpression2param(xForFit_cont, fit_ODsat, fit_Ng)

# look at the data and the fit
fig = plt.figure()
fig.set_size_inches(4, 4)
sns.scatterplot(data=PC4data, x="OD", y="GFP",color='b',alpha = 0.1,s=55)
plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)
plt.plot(xForFit_cont,fitY_cont,'b-')
plt.xscale('log')
plt.yscale('log')
plt.title('2 parameter fit')

# ok, this works well. do all 3 promoters at once now using all their datapoints.

#%% simultaneous fit of all datapoints of all promoters - 2 parameter model

#gather the data for the fits
xForFit_unsorted4 = PC4data['OD']
yForFit_unsorted4 = PC4data['GFP']
sortOrder4 = np.argsort(xForFit_unsorted4)
FitX4 = xForFit_unsorted4.reset_index(drop=True)[sortOrder4]
FitY4 = yForFit_unsorted4.reset_index(drop=True)[sortOrder4]

xForFit_unsorted5 = PC5data['OD']
yForFit_unsorted5 = PC5data['GFP']
sortOrder5 = np.argsort(xForFit_unsorted5)
FitX5 = xForFit_unsorted5.reset_index(drop=True)[sortOrder5]
FitY5 = yForFit_unsorted5.reset_index(drop=True)[sortOrder5]

xForFit_unsorted6 = PC6data['OD']
yForFit_unsorted6 = PC6data['GFP']
sortOrder6 = np.argsort(xForFit_unsorted6)
FitX6 = xForFit_unsorted6.reset_index(drop=True)[sortOrder6]
FitY6 = yForFit_unsorted6.reset_index(drop=True)[sortOrder6]

#perform the 2 parameter fit
FitBounds = ([0.1,100000],[2,2000000]) # lower and upper bounds for [ODsat, Ng]
# PC4
popt, pcov = scipy.optimize.curve_fit(tissueExpression2param, FitX4, FitY4, bounds = FitBounds)
fit_ODsat = popt[0]
fit_Ng = popt[1]
fitY_cont4 = tissueExpression2param(xForFit_cont, fit_ODsat, fit_Ng) # fitted GFP fluo for continuous ODs
# PC5
popt, pcov = scipy.optimize.curve_fit(tissueExpression2param, FitX5, FitY5, bounds = FitBounds)
fit_ODsat = popt[0]
fit_Ng = popt[1]
fitY_cont5 = tissueExpression2param(xForFit_cont, fit_ODsat, fit_Ng) # fitted GFP fluo for continuous ODs
# PC6
popt, pcov = scipy.optimize.curve_fit(tissueExpression2param, FitX6, FitY6, bounds = FitBounds)
fit_ODsat = popt[0]
fit_Ng = popt[1]
fitY_cont6 = tissueExpression2param(xForFit_cont, fit_ODsat, fit_Ng) # fitted GFP fluo for continuous ODs


# plot everything, the raw data and the fits
fig = plt.figure()
fig.set_size_inches(4, 4)
sns.scatterplot(data=PC4data, x="OD", y="GFP",color='b',alpha = 0.1,s=55)
plt.plot(xForFit_cont,fitY_cont4,'b-')
#plt.errorbar(MeanPerOD4.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)

sns.scatterplot(data=PC5data, x="OD", y="GFP",color='orange',alpha = 0.1,s=55)
plt.plot(xForFit_cont,fitY_cont5,'-',color='orange')
#plt.errorbar(MeanPerOD5.index, MeanPerOD4['GFP'], SDPerOD4['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)

sns.scatterplot(data=PC6data, x="OD", y="GFP",color='green',alpha = 0.1,s=55)
plt.plot(xForFit_cont,fitY_cont6,'g-')
#plt.errorbar(MeanPerOD6.index, MeanPerOD6['GFP'], SDPerOD6['GFP'], fmt="o", color="k",mfc='white',mec='black', ms=6)

plt.plot(xForFit_cont,fitY_cont4,'b-')
plt.xscale('log')
plt.yscale('log')
plt.title('2 parameter fit')



#%% Simultaneous fit of all datapoints in all 3 datasets.
# we want them to share some parameters (ODsat, N) but each dataset (promoter) to have its own fitted 'g'.

from lmfit import minimize, Parameters, report_fit


# define the core model functions
# def gauss(x, amp, cen, sigma):
#     """Gaussian lineshape."""
#     return amp * np.exp(-(x-cen)**2 / (2.*sigma**2))

def tissueExpression3param(OD, ODsat, N, g): # we let N and g to change independently
    # N = number of trials
    # OD = the culture OD
    # ODsat = OD at which the probability of success becomes 1
    # g = expression per success
    sat_idx = find_nearest(OD,ODsat)# index of the OD closest to ODsat. This function is defined earlier in the script
    OD0 = OD[0:sat_idx]
    OD1 = np.ones(np.size(OD[sat_idx:]))
    
    gfp0 = N*g*(OD0/ODsat)
    gfp1 = N*g*OD1
    return np.concatenate([gfp0,gfp1])



# define the parameter handling functions
# def gauss_dataset(params, i, x):
#     """Calculate Gaussian lineshape from parameters for data set."""
#     amp = params[f'amp_{i+1}']
#     cen = params[f'cen_{i+1}']
#     sig = params[f'sig_{i+1}']
#     return gauss(x, amp, cen, sig)

def tissueExp_dataset(params, i, x):
    """Calculate Binomial tissue expression lineshape from parameters for data set."""
    ODsat = params[f'ODsat_{i+1}']
    N = params[f'N_{i+1}']
    g = params[f'g_{i+1}']
    return tissueExpression3param(x, ODsat, N, g)



# define the objective functions to be minimized
# def objectiveGauss(params, x, data):
#     """Calculate total residual for fits of Gaussians to several data sets."""
#     ndata, _ = data.shape
#     resid = 0.0*data[:]
#     # make residual per data set
#     for i in range(ndata):
#         resid[i, :] = data[i, :] - gauss_dataset(params, i, x)
#     # now flatten this to a 1D array, as minimize() needs
#     return resid.flatten()

def objectiveBinExp(params, x, data):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - tissueExp_dataset(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()




# # make fake Gaussian data
# np.random.seed(2021)
# xGauss = np.linspace(-1, 2, 151)
# data = []
# for _ in np.arange(5):
#     amp = 0.60 + 9.50*np.random.rand()
#     cen = -0.20 + 1.20*np.random.rand()
#     sig = 0.25 + 0.03*np.random.rand()
#     dat = gauss(xGauss, amp, cen, sig) + np.random.normal(size=xGauss.size, scale=0.1)
#     data.append(dat)
# data = np.array(data)


# load OD and GFP data of 3 datasets (promoters)
ODvals = []
dataGFP = []
dataGFP.append(FitY4)
ODvals.append(FitX4)
dataGFP.append(FitY5)
ODvals.append(FitX5)
dataGFP.append(FitY6)
ODvals.append(FitX6)
# data has shape (3, 336)
dataGFP = np.array(dataGFP)
ODvals = np.array(ODvals)
#assert(dataGFP.shape) == (3, 336)


# # run fitting for gaussian
# fit_params = Parameters()
# for iy, y in enumerate(data):
#     fit_params.add(f'amp_{iy+1}', value=0.5, min=0.0, max=200)
#     fit_params.add(f'cen_{iy+1}', value=0.4, min=-2.0, max=2.0)
#     fit_params.add(f'sig_{iy+1}', value=0.3, min=0.01, max=3.0)

# for iy in (2, 3, 4, 5):
#     fit_params[f'sig_{iy}'].expr = 'sig_1'
    
# outGauss = minimize(objectiveGauss, fit_params, args=(xGauss, data))
# report_fit(outGauss.params)

# run fitting for Binomial 

fit_paramsBin = Parameters()
for iy, y in enumerate(dataGFP):
    fit_paramsBin.add(f'ODsat_{iy+1}', value=0.4, min=0.2, max=4) # bounds for the saturation OD
    fit_paramsBin.add(f'N_{iy+1}', value=15, min=10, max=25) # bounds for the number of 'chances' of transformation per cell
    fit_paramsBin.add(f'g_{iy+1}', value=100000, min=100, max=1000000000) # bounds for the number of AUs that each extra transformation event adds

for iy in (2, 3):
    fit_paramsBin[f'ODsat_{iy}'].expr = 'ODsat_1'
    fit_paramsBin[f'N_{iy}'].expr = 'N_1'
    
outBin = minimize(objectiveBinExp, fit_paramsBin, args=(ODvals[0,:], dataGFP))
report_fit(outBin.params)
fittedODsat = outBin.params['ODsat_1'].value
fittedN = outBin.params['N_1'].value


# #plot
# plt.figure() #gaussian fits
# for i in range(5):
#     y_fit = gauss_dataset(outBin.params, i, xGauss)
#     plt.plot(xGauss, data[i, :], 'o', xGauss, y_fit, '-')


xBin = np.logspace(-2,0.7,100) #continuous OD values for plotting the fitted function
fig = plt.figure()
fig.set_size_inches(4, 4)
Colors = ['blue','orange','green']
for i in range(3):
    y_fit = tissueExp_dataset(outBin.params, i, xBin)
    C = Colors[i]
    plt.plot(ODvals[i,:], dataGFP[i, :], 'o',alpha = 0.05,ms=7,color=C)
    plt.plot(xBin, y_fit, '-',color=C)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('OD')
plt.ylabel('GFP fluorescence (a.u)')
plt.title('OD$_{sat}$ = '+str(np.round(fittedODsat,2)) + '\n N = '+str(np.round(fittedN,2)))



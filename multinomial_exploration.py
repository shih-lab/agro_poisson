#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 17:20:24 2023

@author: simon_alamos
"""


import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

from lmfit import minimize, Parameters, report_fit

#%%

k = 2 # number of sides of the die = number of strains
n = 6 # number of rolls of the dice = max number of contacts per cell
ODsat = 0.35
beta = 1/ODsat#proportionality constant to go from OD to probability
ODmix = np.logspace(-3,0,6) # ODs of the combined labeled strains
ODsLabeledStrain = ODmix/k

ODtot = 2
ODdarkStrain = ODtot - ODmix # the dark strain
# calculate the 'adjusted' ODs
if ODtot > ODsat:
    ODsLabeledStrain_adj = ODsLabeledStrain * (ODsat/ODtot)
    ODdarkStrain_adj = ODdarkStrain * (ODsat/ODtot)

# convert ODs to probability

PlabeledStrain = ODsLabeledStrain_adj * beta
PdarkStrain = ODdarkStrain_adj * beta
    
print('this should be all 1 = ' + str((PlabeledStrain*2)+PdarkStrain))

# now calculate the multinomial probability

PzeroSuccesses_oneStrain = (1 - PlabeledStrain)**n
PatleastOneSuccess_oneStrain = 1 - PzeroSuccesses_oneStrain

# now for either strain
PzeroSuccesses_eitherStrain = (1 - PlabeledStrain*2)**n
PatleastOneSuccess_eitherStrain = 1 - PzeroSuccesses_eitherStrain


fig = plt.figure()
fig.set_size_inches(2, 2)
plt.plot(ODmix,PatleastOneSuccess_eitherStrain,'-o')
plt.xlabel('OD of reporter strains')
plt.ylabel('probability of at least one success \n either strain')
plt.xscale('log')
plt.yscale('log')

#%% now do in a loop of different ODtot values

# load the data
fractionTransformable = 0.55 # fraction of all nuclei that can get transformed

ODdata1 = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/8-28-23/AllData3.csv')
ODdata2 = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/9-19-23/AllData3.csv')
ODdata = pd.concat([ODdata1,ODdata2]) # combine them
ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1
ODdata['fracEither'] = (ODdata['fracRFP'] + ODdata['fracGFP']) - ODdata['ObsPBoth']

ODdata['fracEither'] = (ODdata['fracRFP'] + ODdata['fracGFP']) - ODdata['ObsPBoth']

n = 20 # number of rolls of the dice = max number of contacts per cell
ODsat = 0.3
beta = 1/ODsat#proportionality constant to go from OD to probability
#beta = 3
#ODEachColor = np.array([0.001,0.003,0.005,0.01,0.025,0.05,0.1,0.25,0.5]) # ODs of the combined labeled strains
#ODLabeledStrain = np.logspace(-3,0.5,50)


ODtots = [0.1, 0.5, 2]
colors = ['b','orange','g']
fig = plt.figure()
fig.set_size_inches(5, 3.5)

for idx, ODtot in enumerate(ODtots):
    ODLabeledStrain = np.logspace(-3,0.5,50)
    ODLabeledStrain = ODLabeledStrain[ODLabeledStrain <= ODtot] # the OD of the mix can't be higher than the total OD
    ODdarkStrain = ODtot - ODLabeledStrain # the dark strain
    #ODdarkStrain[ODdarkStrain<0]=0
    
    # convert ODs to probability    
    PlabeledStrain = ODLabeledStrain * beta
    PdarkStrain = ODdarkStrain * beta
    Pnone = 1 - PlabeledStrain + PdarkStrain
    Ptrans = PlabeledStrain + PdarkStrain
    
    # adjust the probability to account for competition
    if Ptrans[0] > 1:
        PlabeledStrain = PlabeledStrain/Ptrans
        PdarkStrain = PdarkStrain /Ptrans
        Pnone = 0  
    
    print('this should be all 1 = ' + str((PlabeledStrain)+PdarkStrain+Pnone))

    # now calculate the multinomial probability
    PzeroSuccesses_LabeledStrain = (1 - PlabeledStrain)**n
    PatleastOneSuccess_LabeledStrain = 1 - PzeroSuccesses_LabeledStrain
    
    
    plt.plot(ODLabeledStrain,PatleastOneSuccess_LabeledStrain,'-',color=colors[idx],lw=3)

sns.lineplot(data=ODdata,x='OD',y='fracGFP',err_style="bars",style="ODtot",hue='ODtot',
             markers=True,dashes=False,ms=10,palette = colors,linestyle = '')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('OD of labeled strain')
plt.ylabel('fraction of cells transformed by labeled strain')
plt.legend(['0.1', '0.5','2'],title='total OD')
# plt.xlabel('Multinomial simulated cotransformation probability')
# plt.ylabel('p^2')
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.02,1.2)
plt.show()

    
#if there's no such thing as OD sat, or OD sat is really high, by increasing the total OD
# the probability of the labeled strain stays the same, but that of the dark strain increases at the expense
# of the probability of none.

#%% probability of being transformed by k different strains
# we seek to understand the likelyhood of a cell being transformed by multiple strains, each one at the same OD
#

n = 20 # number of rolls of the dice = max number of contacts per cell
ODsat = 0.35 # OD at which all the rolls of the dice turn out positive, i.e. a transformation event
beta = 1/ODsat#proportionality constant to go from OD to probability
#beta = 3
fractionTransformable = 0.55 # fraction of all nuclei that can get transformed
#ODEachColor = np.array([0.001,0.003,0.005,0.01,0.025,0.05,0.1,0.25,0.5]) # ODs of the combined labeled strains
#ODLabeledStrain = np.logspace(-3,0.5,50)

ODtots = [0.01,0.05,0.1, 0.25, 0.5, 1, 2] # total OD of the mix
ODtots = np.logspace(-2,0.5,10)
# colors = ['b','r','orange','k','green']

# fig = plt.figure()
# fig.set_size_inches(5, 3.5)
#Data = pd.DataFrame([], columns=['ODtot','numStrains','PallStrains'])
kvals = np.arange(1,15) # total number of strains
PsForPlot = np.empty([len(kvals),len(ODtots)])
for idx1, ODtot in enumerate(ODtots):   
    #c= colors[idx1]  
    for idx2, k in enumerate(kvals):
        ODeachStrain = ODtot/k # the OD of each one of the strains in the mix
        
        # convert ODs to probability    
        # adjust the probability to account for competition
        if ((ODeachStrain * beta) * k) >= 1:
            PeachStrain = 1/k
            Pnone = 0
        else:
             PeachStrain = ODeachStrain * beta   
             Pnone = 1-(PeachStrain * k)
            
        # now calculate the multinomial probability of getting just one kind of strain, two, three, etc...
        OutcomeProbs = np.append([Pnone],[PeachStrain]*k) #we're adding the probability of no strain
        Ntrials = 1000000
        PMatrix = np.random.multinomial(n,OutcomeProbs, size = Ntrials)
        # in this matrix each column is a strain and each row is a random trial.
        # the first column corresponds to NO strain
        # the numbers indicate how many times that strain was succesful in transforming
        # now count how many times (counts) there was at least one success for each strain (column)
        PMatrixBinary = PMatrix[:,1:]>0 # wer're not counting the first column since that's the NO STRAIN case
        
        PAllStrains = fractionTransformable * (np.sum(np.sum(PMatrixBinary,1)==k) / Ntrials) # fraction of cells that had at least one success for each strain
        
        #plt.plot(k,PAllStrains,'o',color = c)
        PsForPlot[idx2,idx1] = PAllStrains

# plt.xlabel('number of different strains')
# plt.ylabel('fraction of cells transformed by \n all different strains')
# plt.yscale('log')
# plt.show()


fig = plt.figure()
fig.set_size_inches(5, 3.5)
plt.plot(kvals,PsForPlot,'-o')
plt.xlabel('number of different strains')
plt.ylabel('fraction of cells \n transformed by all different strains')
plt.yscale('log')
plt.legend(list(map(str, np.round(ODtots,2))),title= 'total OD',bbox_to_anchor=(1.1, 1.05),facecolor='k')
plt.ylim(0,0.6)
plt.show()


fig = plt.figure()
fig.set_size_inches(5, 3.5)
plt.plot(ODtots,np.transpose(PsForPlot),'-')
plt.xlabel('total OD')
plt.ylabel('fraction of cells \n transformed by all different strains')
plt.xscale('log')
plt.yscale('log')
plt.legend(list(map(str, kvals)),title= 'number of strains',bbox_to_anchor=(1.1, 1.05))
#plt.ylim(0,0.6)
plt.show()



#%% fit all data to model

# load the data
fractionTransformable = 0.5 # fraction of all nuclei that can get transformed

ODdata1 = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/8-28-23/AllData3.csv')
ODdata2 = pd.read_csv('/Users/simon_alamos/Documents/Shih_lab/Data/Microscopy/RawData/9-19-23/AllData3.csv')
ODdata = pd.concat([ODdata1,ODdata2]) # combine them
ODdata['fracGFP'] = ODdata['NGFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracGFP'].loc[ODdata['fracGFP']>1]=1
ODdata['fracRFP'] = ODdata['NRFP']/(ODdata['NBFP']*fractionTransformable)
ODdata['fracRFP'].loc[ODdata['fracRFP']>1]=1

# prepare the data into the right shape
FractionGFP01 = ODdata['fracGFP'].loc[ODdata['ODtot']==0.1].values
ODs01 = ODdata['OD'].loc[ODdata['ODtot']==0.1].values
FractionGFP05 = ODdata['fracGFP'].loc[ODdata['ODtot']==0.5].values
ODs05 = ODdata['OD'].loc[ODdata['ODtot']==0.5].values
FractionGFP2 = ODdata['fracGFP'].loc[ODdata['ODtot']==2].values
ODs2 = ODdata['OD'].loc[ODdata['ODtot']==2].values

# they don't all have the same size, but this is important for the fitting functions
# so I'll pad with NANs
longestDataset = np.max([len(FractionGFP01),len(FractionGFP05),len(FractionGFP2)])

pad01 = np.nan*np.ones(longestDataset-len(FractionGFP01))
FractionGFP01 = np.concatenate((FractionGFP01,pad01))
ODs01 = np.concatenate((ODs01,pad01))

pad05 = np.nan*np.ones(longestDataset-len(FractionGFP05))
FractionGFP05 = np.concatenate((FractionGFP05,pad05))
ODs05 = np.concatenate((ODs05,pad05))

pad2 = np.nan*np.ones(longestDataset-len(FractionGFP05))
FractionGFP2 = np.concatenate((FractionGFP2,pad2))
ODs2 = np.concatenate((ODs2,pad2))


TitrationODvals = []
dataGFP = []
dataGFP.append(FractionGFP01)
TitrationODvals.append(ODs01)
dataGFP.append(FractionGFP05)
TitrationODvals.append(ODs05)
dataGFP.append(FractionGFP2)
TitrationODvals.append(ODs2)
dataGFP = np.array(dataGFP)
TitrationODvals = np.array(TitrationODvals)

ODtotVals = [0.1,0.5,2]

# define objective function 
def BinomialProbability(ODx,ODtot,ODsat,N):
    # ODx are the x values, the OD of the diluted strain
    # Fy are the y values, the fraction of transformed cells
    # ODtot is the total OD of the mix
    # ODsat is the first constant we're trying to fit: the saturation OD
    # N is the second constant we want to fit: the number of chances per plant cell   
    beta = 1/ODsat#proportionality constant to go from OD to probability   
    # convert ODs to probability    
    PlabeledStrain = ODx * beta
    PdarkStrain = (ODtot - ODx) * beta
    Ptrans = PlabeledStrain + PdarkStrain    
    # adjust the probability to account for competition
    if Ptrans[0] > 1:
        PlabeledStrain = PlabeledStrain/Ptrans    
    # now calculate the multinomial probability
    PzeroSuccesses_LabeledStrain = (1 - PlabeledStrain)**N
    PatleastOneSuccess_LabeledStrain = 1 - PzeroSuccesses_LabeledStrain
    
    return PatleastOneSuccess_LabeledStrain

# define the parameter handling function
def singleCell_dataset(params, i, x, ODtot):
    """Calculate Binomial cell transformation fraction from parameters for data set."""
    ODsat = params[f'ODsat_{i+1}']
    N = params[f'N_{i+1}']
    return BinomialProbability(x, ODtot, ODsat, N)

# define the wrapper objective function 
def objectiveBinExp(params, x, data):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        ODtot = ODtotVals[i]
        resid[i, :] = data[i, :] - singleCell_dataset(params, i, x, ODtot)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

# run fitting of single cell transformation frequencyes to Binomial model
fit_paramsBin = Parameters()

for iy, y in enumerate(dataGFP):
    fit_paramsBin.add(f'ODsat_{iy+1}', value=0.2, min=0.1, max=3) # bounds for the number of 'chances' of transformation per cell
    fit_paramsBin.add(f'N_{iy+1}', value=10, min=3, max=30) # bounds for the number of AUs that each extra transformation event adds

for iy in (2, 3):
    fit_paramsBin[f'ODsat_{iy}'].expr = 'ODsat_1'
    fit_paramsBin[f'N_{iy}'].expr = 'N_1'
    
outBin = minimize(objectiveBinExp, fit_paramsBin, args=(TitrationODvals[0,:], dataGFP),nan_policy='omit')
report_fit(outBin.params)
fittedODsat = outBin.params['ODsat_1'].value
fittedN = outBin.params['N_1'].value
fittedbeta = 1/fittedODsat

#%% plot the fit result

fig = plt.figure()
#plt.style.use('ggplot') 
fig.set_size_inches(4, 4)
palette = ['b','orange','g']
for idx, od in enumerate(ODtotVals):
    ThisODdata = ODdata[ODdata['ODtot']==od]  
    #remove the ODtot = 0.2 that I had included in the ODtot=0.1 experiment
    if od == 0.1:
        ThisODdata = ThisODdata[ThisODdata.OD != 0.2]   
    #remove the ODtot = 1 that I had included in the ODtot=0.5 experiment
    if od == 0.5:
        ThisODdata = ThisODdata[ThisODdata.OD != 1]
        
    ThisODdata['fracNotGFP'] = 1 - ThisODdata['fracGFP']
    # calculate the means for plotting
    MeanPerOD = ThisODdata.groupby(by=["OD"]).mean()
    SDPerOD = ThisODdata.groupby(by=["OD"]).sem()
    NotGFPMeanPerOD = MeanPerOD['fracNotGFP']
    NotGFPSDPerOD = SDPerOD['fracNotGFP']
    ODvals = MeanPerOD.index

    sns.scatterplot(data=ThisODdata, x="OD", y="fracNotGFP",marker='o',color=palette[idx],alpha = 0.5,s=55)
    


for idx, ODtot in enumerate(ODtotVals):
    ODLabeledStrain = np.logspace(-3,0.5,50)
    ODLabeledStrain = ODLabeledStrain[ODLabeledStrain <= ODtot] # the OD of the mix can't be higher than the total OD
    ODdarkStrain = ODtot - ODLabeledStrain # the dark strain
    #ODdarkStrain[ODdarkStrain<0]=0
    
    # convert ODs to probability    
    PlabeledStrain = ODLabeledStrain * fittedbeta
    PdarkStrain = ODdarkStrain * fittedbeta
    Pnone = 1 - PlabeledStrain + PdarkStrain
    Ptrans = PlabeledStrain + PdarkStrain
    
    # adjust the probability to account for competition
    if Ptrans[0] > 1:
        PlabeledStrain = PlabeledStrain/Ptrans
        PdarkStrain = PdarkStrain /Ptrans
        Pnone = 0  

    # now calculate the multinomial probability
    PzeroSuccesses_LabeledStrain = (1 - PlabeledStrain)**fittedN
    PatleastOneSuccess_LabeledStrain = 1 - PzeroSuccesses_LabeledStrain
    
    plt.plot(ODLabeledStrain,PzeroSuccesses_LabeledStrain,'-',color=colors[idx],lw=3)

# sns.lineplot(data=ODdata,x='OD',y='fracNotGFP',err_style="bars",style="ODtot",hue='ODtot',
#              markers=True,dashes=False,ms=10,palette = colors,linestyle = '')
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('OD of labeled strain')
plt.ylabel('fraction of transformable cells \n not transformed by labeled strain')
plt.legend(['0.1', '0.5','2'],title='total OD')
# plt.xlabel('Multinomial simulated cotransformation probability')
# plt.ylabel('p^2')
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(0.02,1.2)
plt.show()




#%% CORRELATIONS: ARE TRANSFORMATION EVENTS INDEPENDENT IN THIS MODEL?
# Hypothesis: because 'slots' are limited, at some point you gotta be transformed by one strain OR another, i.e. they'd be anticorrelated.
# we repeat the simulation

n = 20 # number of rolls of the dice = max number of contacts per cell
ODsat = 0.35 # OD at which all the rolls of the dice turn out positive, i.e. a transformation event
beta = 1/ODsat#proportionality constant to go from OD to probability
#beta = 3
fractionTransformable = 0.55 # fraction of all nuclei that can get transformed
#ODEachColor = np.array([0.001,0.003,0.005,0.01,0.025,0.05,0.1,0.25,0.5]) # ODs of the combined labeled strains
#ODLabeledStrain = np.logspace(-3,0.5,50)

ODtots = [0.05,0.1, 0.5, 1, 2, 3] # total OD of the mix
ODdils = [0.001,0.0025,0.01, 0.025, 0.05, 0.1, 0.5, 1, 2]
#ODtots = np.logspace(-2,0.5,10)
# colors = ['b','r','orange','k','green']

# fig = plt.figure()
# fig.set_size_inches(5, 3.5)
#Data = pd.DataFrame([], columns=['ODtot','numStrains','PallStrains'])
kvals = np.arange(3,10) # total number of strains
PsForPlot = np.empty([len(ODdils),len(ODtots)])
ExpPBoth = np.empty([len(ODdils),len(ODtots)])
PObsBoth = np.empty([len(ODdils),len(ODtots)])

for idx1, ODtot in enumerate(ODtots):   
    ODdils2 = [x for x in ODdils if x<= ODtot]
    for idx2, ODi in enumerate(ODdils2):  
        ODEachLabeledStrain = ODi/2
        ODCompetingStrain = ODtot-(ODi/2)
        
        # convert ODs to probability    
        # adjust the probability of the  strains to account for competition
        if (2*(ODEachLabeledStrain*beta) + ODCompetingStrain*beta) >= 1 or 2*(ODEachLabeledStrain*beta) >=1 or ODCompetingStrain*beta >=1:
            print('saturated')
            PEachLabeledStrain = ODEachLabeledStrain/ODtot
            PCompetingStrain = ODCompetingStrain/ODtot
            Pnone = 0
        else:
            print('not saturated')
            PEachLabeledStrain = ODEachLabeledStrain * beta   
            PCompetingStrain = ODCompetingStrain*beta
            Pnone = 1-(2*PEachLabeledStrain + PCompetingStrain)        
            
        # now calculate the multinomial probability of getting just one kind of strain, two, three, etc...
        OutcomeProbs = np.concatenate(([Pnone],[PEachLabeledStrain]*2,[PCompetingStrain]), axis=None) #we're adding the probability of no strain
        Ntrials = 1000000
        PMatrix = np.random.multinomial(n,OutcomeProbs, size = Ntrials)
        # in this matrix each column is a strain and each row is a random trial.
        # the first column corresponds to NO strain
        # the second column is the first labeled strain
        # the third column is the second labeled strain
        # the fourth column is the competing strain
        # the numbers indicate how many times that strain was succesful in transforming
        
        # now count how many times (counts) there was at least one success for each strain (column)
        PMatrixBinary = PMatrix>0 
        
         # we ask how many out of Ntrials columns 1 is true
        PObsFirst = np.sum(PMatrixBinary[:,1]>0)/Ntrials
        PObsSecond = np.sum(PMatrixBinary[:,2]>0)/Ntrials
        ExpPBoth[idx2,idx1] = PObsFirst * PObsSecond
        
        # we ask how many out of Ntrials columns 1 and 2 are true
        NObsBoth = np.sum(np.sum(PMatrixBinary[:,1:3],1)==2)
        PObsBoth[idx2,idx1] = NObsBoth/Ntrials
        
        #PAllStrains = fractionTransformable * (np.sum(np.sum(PMatrixBinary,1)==k) / Ntrials) # fraction of cells that had at least one success for each strain
        
        #plt.plot(k,PAllStrains,'o',color = c)
        #PsForPlot[idx2,idx1] = PAllStrains

# plt.xlabel('number of different strains')
# plt.ylabel('fraction of cells transformed by \n all different strains')
# plt.yscale('log')
# plt.show()
fig = plt.figure()
fig.set_size_inches(2, 2)
plt.plot(ExpPBoth,PObsBoth,'-o')
plt.plot(ExpPBoth,ExpPBoth,'k')
plt.xlabel('expected cotransformation \n if independent')
plt.ylabel('observed')
#plt.yscale('log')
plt.legend(list(map(str, np.round(ODtots,2))),title= 'total OD',bbox_to_anchor=(1.1, 1.05))
plt.ylim(0,1)
plt.xlim(0,1)

plt.show()




#%%

fig = plt.figure()
fig.set_size_inches(5, 3.5)
plt.plot(kvals,PsForPlot,'-o')
plt.xlabel('number of different strains')
plt.ylabel('fraction of cells \n transformed by all different strains')
plt.yscale('log')
plt.legend(list(map(str, np.round(ODtots,2))),title= 'total OD',bbox_to_anchor=(1.1, 1.05),facecolor='k')
plt.ylim(0,0.6)
plt.show()


fig = plt.figure()
fig.set_size_inches(5, 3.5)
plt.plot(ODtots,np.transpose(PsForPlot),'-')
plt.xlabel('total OD')
plt.ylabel('fraction of cells \n transformed by all different strains')
plt.xscale('log')
plt.yscale('log')
plt.legend(list(map(str, kvals)),title= 'number of strains',bbox_to_anchor=(1.1, 1.05))
#plt.ylim(0,0.6)
plt.show()









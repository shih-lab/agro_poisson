#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:37:11 2023

@author: simon_alamos
"""

import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% simulation

p = np.logspace(-3,-0.1,60) # an array of probability
Ntot = 100 #total number of cells
Fsus = 0.85 # fraction of cells that can be transformed
Nsus = Ntot*Fsus # number of cells that can be transformed

# if only Nsus can be transformed
N1 = p * Nsus
P1 = N1/Ntot
N2 = p * Nsus
P2 = N2/Ntot
Nboth = (P1 * P2) * Nsus
Pboth = Nboth/Ntot

#if all cells can be transformed
M1 = p * Ntot
Q1 = M1/Ntot
M2 = p * Ntot
Q2 = M2/Ntot
Mboth = (Q1 * Q2) * Ntot
Qboth = Mboth/Ntot

#%%

fig = plt.figure()
fig.set_size_inches(3.5, 3.5)
plt.plot(Qboth,Pboth,'b-')
plt.plot(Qboth,Qboth,'k-')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('simulated probability of co-transformation \n if only ' + str(100*Fsus)+ '% of all cells can be transformed')
plt.ylabel('simulated probability of co-transformation \n if ALL cells can be transformed')

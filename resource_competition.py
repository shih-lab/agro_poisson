#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:54:03 2023

@author: simon_alamos
"""

# Import some packages
import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial # for n choose k such as comb(n, k, exact=False)

#%% define functions
def Multiplicity_sum(Nc, L, Nm):
    # Nc = number of competing bacteri
    # L = total number of ligands
    # x = number of ligands bound to marked bacteria
    # Nm = number of marked bacteria
    # epsilon = boltzmann energy
    maxiter = np.min([L,Nm,Nc]) # figure out up to what number we can do the sum, whichever is smallest
    sumarray = np.zeros(maxiter)
    iterarray = np.arange(maxiter)
    
    for i, x in enumerate(iterarray):
        sumarray[i] = comb(Nc,L-x) * comb(Nm,x)
        
    return np.sum(sumarray)

def Multiplicity_sum2(Nc, L, Nm):
    # Nc = number of competing bacteri
    # L = total number of ligands
    # x = number of ligands bound to marked bacteria
    # Nm = number of marked bacteria
    # epsilon = boltzmann energy
    maxiter = np.min([L,Nm,Nc]) # figure out up to what number we can do the sum, whichever is smallest
    multiplicity = np.zeros(maxiter)
    iterarray = np.arange(1,maxiter)
    
    for i, x in enumerate(iterarray):
        multiplicity[i] = comb(Nc,L-x) * comb(Nm,x)
        
    return multiplicity
    # array showing the number of ways x ligands can be bound to Nm bacteria and L-x to Nc bacteria
    # for x = 1-min(L,Nm,Nc)

#%%
tot_cells = 100
Nm_vals = np.logspace(0,np.log10(tot_cells),50) # number of marked bacteria in units of OD
Nc_vals = tot_cells - Nm_vals # number of competing bacteria in units of OD
L = 10
bf = 0.01
#xvalues = 

pBoundValues = np.zeros(len(Nm_vals))

for i, Nm in enumerate(Nm_vals):
    Nm = int(Nm)
    Nc = int(tot_cells - Nm) # number of competing bacteria in units of OD
    Zm = (Multiplicity_sum(Nc, L, Nm)) + comb(Nm,L) # L-x ligands in Nc bacteria, x ligands in Nm bacteria
    Zc = comb(Nc,L) # L ligands in Nc bacteria
    pBoundValues[i] = Zm/(1 + Zc + Zm)

#print(pBoundValues)
fig = plt.figure()
fig.set_size_inches(4, 4)
plt.plot(100*Nm_vals/tot_cells,pBoundValues)
plt.xlabel('percentage of cells that are marked cells')
plt.ylabel('fraction of marked cells bound to ligand')
plt.ylim(0,1.1)



#%%
L = 5
Kd = 1
Nm  = 5
Nc = 5


# the number of ways L ligands can be bound to Nc receptors:
mult1 = comb(Nc,min([L,Nc]))
weight1 = L/Kd
# the number of ways L ligands can be bound to Nm receptors:
mult2 = comb(Nm,min([L,Nm]))
weight2 = L/Kd
# the number of ways x ligands can be bound to Nm and L-x to Nc is:
mult3 = np.sum(Multiplicity_sum2(Nc, L, Nm))
weight3 = L/Kd
# the partition function is
Z =  mult1*weight1 + mult2*weight2 + mult3*weight3

# the probability that all Nm bacteria are bound to the ligand is
PNc_all = (mult1 * weight1)/Z
PNm_all = (mult2 * weight2)/Z
print(PNm_all,PNc_all)

# the multiplicity of 1, 2, 3...Nm bacteria are bound by the ligand and L-1, L-2, L-3...ligands are bound to Nc
Nm_x = Multiplicity_sum2(Nc, L, Nm)
PNm_x = Nm_x*weight3/Z
print(PNm_x)

Ptot = PNc_all + PNm_all + np.sum(PNm_x)
print(Ptot)
plt.plot(PNm_x)


# average number of Ligand-Nm cells
AvgNmBound = L*PNm_all + 1*(PNm_x[0]) + 2*(PNm_x[1]) + 3*(PNm_x[2]) + 4*(PNm_x[3])

#%%
omega = float(1000000) # number of boxes
L = 5 # number of ligands
Esol = -2 # energy of the ligand in solution
Ebound = -9 # energy of the ligand when bound
kT = 1
beta = 1/kT
Nc = 5 # number of competitor receptors
Nt = 5 # number of target receptors

def SumWeights(Nc,Nt,L,omega,beta,Esol,Ebound):
    
    iterNc = np.arange(Nc+1)
    iterNt = np.arange(Nt+1)
    Weights = np.zeros([len(iterNc),len(iterNt)])
    for i in iterNc:
        for j in iterNt:
            if L-i-j > 0:
                emptyWeight = np.power(omega,(L-i-j))/factorial(L-i-j) * np.exp(-(L-i-j)*Esol)
                boundWeight = comb(Nc,i) * comb(Nt,j) * np.exp(-(i+j)*Ebound)
                Weights[i,j] = emptyWeight * boundWeight
    return Weights
    
BoundWeights = SumWeights(Nc,Nt,L,omega,beta,Esol,Ebound) # the sum of of the weights were at least one of either kind of receptor is bound
Z = np.power(omega,L)/factorial(L) * np.exp(-L*Esol) + np.sum(BoundWeights[:])

pNoneBound = np.power(omega,L)/factorial(L) * np.exp(-L*Esol)  / Z # probability that no ligands are bound, i.e. all receptors are empty

print(pNoneBound)








































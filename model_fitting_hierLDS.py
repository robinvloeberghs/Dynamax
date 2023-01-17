# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:49:01 2022

@author: Robin Vloeberghs (KU Leuven)
"""


# see https://github.com/lindermanlab/ssm/blob/master/notebooks/1b%20Simple%20Linear%20Dynamical%20System.ipynb
# see also https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py#L825

# K = 1       # number of discrete states: 1, no switching
# D = 1       # number of latent dimensions: 1, AR model
# N = 1       # number of observed dimensions: only choices. Could add in RT/confidence?
# M = 3       # input dimensions

# To do:
# random restarts?
# fix parameters to certain values
# check update hierarchical prior
# allow different prior variance for Fs[0] (stimulus, perceptual sensitivity)?


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import simulate_choices
import ssm

np.random.seed(seed=123)


# Initialize simulation params
ntrials             = 1000 # trials per subject
nsubjects           = 5

# Allow some variability in param values across subjects
df_sens             = 10 + .1*np.random.randn(nsubjects) # perceptual sensitivity
df_bias             = 0 + .1*np.random.randn(nsubjects) # intercept, 0 is unbiased if evidence is effect coded in simulations

df_prevresp         = 0 + .1*np.random.randn(nsubjects) # weights previous response
df_prevconf         = 0 + .1*np.random.randn(nsubjects)
df_prevrespconf     = 0 + .1*np.random.randn(nsubjects)
df_prevsignevi      = 0 + .1*np.random.randn(nsubjects)
df_prevabsevi       = 0 + .1*np.random.randn(nsubjects)
df_prevsignabsevi   = 0 + .1*np.random.randn(nsubjects)

σd                  = 0.05 # variance AR process slow drift

# np.mean(df_prevrespconf)



# a hierarchical model needs its input organised in a tuple, with subsets for each subject, and a separate variable indicating the tags
# structure derived from: https://github.com/lindermanlab/ssm/blob/e97ea4f0904cd204f392c2cfc4528ef860d71f9d/notebooks/8%20Hierarchical%20ARHMM%20Demo.ipynb
# see file test_ARHMM.py
# see if simulated data below can be estimated with HMM to rule out problem in simulated data structure: CHECK!


t_inputs = () # empty tuple
t_choices = ()
t_drift = ()
t_tags = ()

# Generate data for each subject from SDT framework
for sub in range(nsubjects):
    d_inputs, choices, drift = simulate_choices.simulateChoice_normalEvi_slowdriftConf(ntrials,
                                              estimateUpdating = True,
                                              fixedConfCriterion = False,
                                              postDecisionalEvi = True,
                                              σd = σd, 
                                              sens = df_sens[sub], 
                                              bias = df_bias[sub], 
                                              w_prevresp = df_prevresp[sub],
                                              w_prevconf = df_prevconf[sub],
                                              w_prevrespconf = df_prevrespconf[sub],
                                              w_prevsignevi = df_prevsignevi[sub],
                                              w_prevabsevi = df_prevabsevi[sub],
                                              w_prevsignabsevi = df_prevsignabsevi[sub],
                                              seed = sub) # if a constant then same drift for each person
    
    t_inputs = t_inputs + (d_inputs,)
    t_choices = t_choices + (choices,)
    t_drift = t_drift + (drift,)
    t_tags = t_tags + (sub,)
    
    
inputDim = np.shape(d_inputs)[1] # observed input dimensions 
stateDim = 1 # latent states
lags = 1
n_iters = 50


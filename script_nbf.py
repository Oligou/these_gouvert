#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import numpy as np
import cPickle as pickle
import sys
sys.path.append("model/nbf")

from pf_vi import pf_vi
from nbf_vi import nbf_vi
from nbfnz_vi import nbfnz_vi
import preprocess_data  as prep

#%% Data
with open('data/TPS/tps_98765_U1.50e+03_I7.86e+02_min_uc20_sc50', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']

seed_train_test = 1992
Y_train,Y_test = prep.divide_train_test(Y,prop_test=0.2,seed=seed_train_test)

# data size (F = PxQ)
U,I = Y.shape;

#%%
Ks = [50,100,200,400]

# Hyper
alphaWH = 1.
opt_hyper = ['betaH']

# Set values of alpha
alphas = [.1,1.,10.,100.]; 
n_alpha= len(alphas);

#%% Set algorithm parameters

# Nb of random initialisations
seeds = [1404,3764,692,116,8334] 
n_seed = len(seeds)

# Stopping criterion
precision = 10**(-6);

# number of iterations
min_iter = 0
max_iter = 10**(5) 

save_dir = 'out/nbf'

#%%  PF
for seed in seeds:
    for K in Ks:
        # PF 
        if False: # init random
            model_raw = pf_vi(K=K, 
                          alphaW = alphaWH, alphaH=alphaWH)
            model_raw.fit(Y=Y_train,
                      seed=seed, init='rand',
                      opt_hyper=opt_hyper,
                      verbose=False,
                      precision=precision, max_iter=max_iter, min_iter=min_iter,
                      save = True, save_dir=save_dir, prefix='tps')
            print('+1')
            
        # PF bin
        if False:
            model_bin = pf_vi(K=K, 
                          alphaW = alphaWH, alphaH=alphaWH)
            model_bin.fit(Y=Y_train>0,
                      seed=seed, init='rand',
                      opt_hyper=opt_hyper,
                      verbose=False,
                      precision=precision, max_iter=max_iter, min_iter=min_iter,
                      save = True, save_dir=save_dir, prefix='tps_bin')
            print('+1')
                                 
#%% NB
if False: # Init random
    for seed in seeds:
        for init in ['rand']: # 'pf','pf_bin'
            for K in Ks:
                for alpha in alphas:
                    model_nb = nbf_vi(K=K,
                                   alphaW = alphaWH, alphaH=alphaWH,
                                   alphaNB=alpha)
                    model_nb.fit(Y=Y_train,
                              seed=seed, init=init,
                              opt_hyper=opt_hyper,
                              verbose=False,
                              precision=precision, max_iter=max_iter, min_iter=min_iter,
                              save = True, save_dir=save_dir, prefix='tps')
                    print('+1')
                
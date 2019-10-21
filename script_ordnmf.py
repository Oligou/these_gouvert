#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

#%% 
import sys
sys.path.append("model/ordnmf")

import cPickle as pickle
import numpy as np

from ONMF_implicit import ONMF_implicit

import preprocess_data  as prep

#%% 
prop_test = 0.2
seed_test = 1992

with open('data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']

U,I = Y.shape

# Thresholding the data
threshold = np.array([[0,1,2,5,10,20,50,100,200,500]])
compare = Y.data[:,np.newaxis]>threshold
Y.data = np.sum(compare,1)

Y_train,Y_test = prep.divide_train_test(Y,prop_test=prop_test,seed=seed_test)

save_dir = 'out/ordnmf'

#%% IN 
alpha = .3 
opt_hyper = ['beta']  
approx = False # Approx Bernoulli -> Poisson

Ks = [150,100,200,300]
Seeds = [1404, 2510, 9876, 6060, 4892] # Seed of the different initializations
tol = 10**(-5)
min_iter = 0
max_iter = 10**5
    
#%% Ord NMF
for K in Ks:
    for seed in Seeds:
        if False:
            model = ONMF_implicit(K=K, alphaW=alpha, alphaH=alpha)
            model.fit(Y_train, T=Y_train.max(), 
                      seed=seed, opt_hyper=opt_hyper, 
                      approx = approx, 
                      precision=tol, min_iter=min_iter, max_iter=max_iter,
                      save=True, save_dir=save_dir,prefix='tps', 
                      verbose=False)
   
#%%         
for K in Ks:
    for seed in Seeds:
        if False:
            model = ONMF_implicit(K=K, alphaW=alpha, alphaH=alpha)
            model.fit(Y_train>0, T=1, 
                      seed=seed, opt_hyper=opt_hyper, 
                      approx = approx, 
                      precision=tol, min_iter=min_iter, max_iter=max_iter,
                      save=True, save_dir=save_dir,prefix='tps_bin', 
                      verbose=False)
        
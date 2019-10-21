#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert
"""

import numpy as np
import cPickle as pickle

from function.train_test import divide_train_test

import sys
sys.path.append("model/dcpf")

from dcpf_Log import dcpf_Log
from dcpf_ZTP import dcpf_ZTP
from dcpf_Geo import dcpf_Geo
from dcpf_sNB import dcpf_sNB

#%% DATA
seed_test = 1992
prop_test = 0.2
# pre-processed data 
with open('data/TPS/tps_12345_U1.63e+04_I1.21e+04_min_uc20_sc20', 'rb') as f:
    out = pickle.load(f)
    Y = out['Y_listen']

Y_train,Y_test = divide_train_test(Y,prop_test=prop_test,seed=seed_test)

#%% 
Seed = [1404,2510,4892,6060,9876] # Seed of the different initializations 1404, 
Ks = [150] # Number of latent factors 50,100,150,200,300,
alphas = [.3] # shape parameter of the gamma priors of W and H 

# Note that limit cases: PF on raw data and PF on binarized data are obtained, 
# for example, when p=0 and p=1 for dcpf with Log 

tol = 10**(-5)
min_iter = 0
max_iter = 10**5

save_dir = 'out/dcpf'

####################
#%% dcPF: Log
####################

if False:
    opt_hyper = ['beta']  
    for seed in Seed:
        for p in np.linspace(0,1.,11): # Grid-search
            for K in Ks:
                for alpha in alphas:
                    model = dcpf_Log(K=K, p=p, alphaW=alpha,alphaH=alpha)
                    model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                              precision=tol, min_iter=min_iter, max_iter=max_iter,
                             save=True, save_dir=save_dir,prefix='tps', 
                             verbose=True)
                    print '+1'

if False:
    opt_hyper = ['p','beta']  
    p = 0.5
    for seed in Seed:
        for K in Ks:
            for alpha in alphas:
                model = dcpf_Log(K=K, p=p, alphaW=alpha,alphaH=alpha)
                model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                         save=True, save_dir=save_dir,prefix='tps', 
                         verbose=True)
                print '+1'
                    
####################
#%% dcPF: ZTP
####################

if False:
    opt_hyper = ['beta']  
    for seed in Seed:
        for p in [0.,.1,.5,1.,2.,5.,10.,float('inf')]: # Grid-search
            for K in Ks:
                for alpha in alphas:
                    model = dcpf_ZTP(K=K, p=p, alphaW=alpha,alphaH=alpha)
                    model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                              precision=tol, min_iter=min_iter, max_iter=max_iter,
                             save=True, save_dir=save_dir,prefix='tps', 
                             verbose=False)
                    print '+1'
   
if False:
    opt_hyper = ['p','beta']  
    p = 1.
    for seed in Seed:
        for K in Ks:
            for alpha in alphas:
                model = dcpf_ZTP(K=K, p=p, alphaW=alpha,alphaH=alpha)
                model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                         save=True, save_dir=save_dir,prefix='tps', 
                         verbose=False)
                print '+1'
                
####################
#%% dcPF: Geo
####################

if False:
    opt_hyper = ['beta']  
    for seed in Seed:
        for p in np.linspace(0,1.,11): # Grid-search
            for K in Ks:
                for alpha in alphas:
                    model = dcpf_Geo(K=K, p=p, alphaW=alpha,alphaH=alpha)
                    model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                              precision=tol, min_iter=min_iter, max_iter=max_iter,
                             save=True, save_dir=save_dir,prefix='tps', 
                             verbose=False)
                    print '+1'
               
if False:
    opt_hyper = ['p','beta']  
    p = .5
    for seed in Seed:
        for K in Ks:
            for alpha in alphas:
                model = dcpf_Geo(K=K, p=p, alphaW=alpha,alphaH=alpha)
                model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                         save=True, save_dir=save_dir,prefix='tps', 
                         verbose=False)
                print '+1'
                
####################
#%% dcPF: sNB
####################

if False:
    opt_hyper = ['beta']  
    for seed in Seed:
        for p in np.linspace(0,1.,11): # Grid-search
            for K in Ks:
                for alpha in alphas:
                    model = dcpf_sNB(K=K, p=p, alphaW=alpha,alphaH=alpha)
                    model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                              precision=tol, min_iter=min_iter, max_iter=max_iter,
                             save=True, save_dir=save_dir,prefix='tps', 
                             verbose=False)
                    print '+1'
               
if False:
    opt_hyper = ['p','beta']  
    p = .5
    for seed in Seed:
        for K in Ks:
            for alpha in alphas:
                model = dcpf_sNB(K=K, p=p, alphaW=alpha,alphaH=alpha)
                model.fit(Y_train, opt_hyper=opt_hyper, seed=seed,
                          precision=tol, min_iter=min_iter, max_iter=max_iter,
                         save=True, save_dir=save_dir,prefix='tps', 
                         verbose=False)
                print '+1'
                   
    
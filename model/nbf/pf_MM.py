#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:03:35 2017

@author: ogouvert
"""

#%% GIBBS FOR ZIPF

## Model
# W ~ Gamma(aphaW,betaW)    ## UxK xN
# H ~ Gamma(aphaH,betaH)    ## IxK xN
# V ~ Gamma(apha,beta)      ## UxI xN       
# c ~ Poisson(V*W*H)        ## UxIxK xN
# y = sum(c)                ## UxI

## Conditional
# W|H,V,c ~ Gamma(aphaW+sum(c), betaW+sum(V*H))
# V|W,H,c ~ Gamma(alpha+sum(c), alpha+WH)
# c|W,H,V ~ Mult(y, W*H)

## Order: W,H,V,C
#%%
import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import os
import sys
import cPickle as pickle
import time

class pf_MM():
    def __init__(self,K):
        self.K = K
        self.score={}
        self.classname = 'pf_MM' 
        # Save arg
        self.saved_args_init = locals()
        del self.saved_args_init['self']
        
    def fit(self, Y, Mask,
            seed=None, init='rand',
            precision=10**(-5), min_iter=0, max_iter=10**5,
            verbose=False, 
            save=True, save_dir='', prefix=None, suffix=None):
        
        self.seed = seed
        if isinstance(seed, int):
            np.random.seed(seed)
        self.init = init
        self.verbose = verbose
        self.precision = precision
        self.min_iter = min_iter
        self.max_iter = max_iter
        # Save
        self.save = save
        self.save_dir = save_dir
        self.filename = self.filename(prefix, suffix)
        # Save arg
        self.saved_args_fit = locals()
        del self.saved_args_fit['Y']
        del self.saved_args_fit['Mask']
        del self.saved_args_fit['self']
        # Timer
        start_time = time.time()
        
        # CONSTANTS
        self.Cost = [float("inf")]
        
        # Constant
        eps = 10**(-16)
        U,I = Y.shape
        
        # INIT
        if init=='pf_bin':
            model_init = pf_MM(K=self.K)
            model_init.fit(Y>0, Mask=Mask,
                      precision=precision*10, min_iter=min_iter/10, max_iter=max_iter/10,
                      save=True, save_dir=save_dir, prefix='INIT_pfbin_'+prefix)
            Ew = model_init.Ew
            Eh = model_init.Eh
            self.model_init = model_init
        else:
            Ew = np.random.gamma(1.,1.,(U,self.K))
            Eh = np.random.gamma(1.,1.,(I,self.K))
            Ew,Eh = renorm(Ew,Eh)
        Yap = Ew.dot(Eh.T) + eps
        Y = Y + eps
        
        for n in range(max_iter):    
            if self.verbose:
                print('ITERATION #%d' % n)
            # Updates 
            Num = (Mask*Y/Yap).T.dot(Ew)
            Den = Mask.T.dot(Ew)
            Eh = Eh*Num/Den
            Ew,Eh = renorm(Ew,Eh)
            Yap = Ew.dot(Eh.T) + eps
                
            Num = (Mask*Y/Yap).dot(Eh)
            Den = Mask.dot(Eh)
            Ew = Ew*Num/Den
            Ew,Eh = renorm(Ew,Eh)
            Yap = Ew.dot(Eh.T) + eps
            
            ## Cost
            cost = kl(Y, Yap, Mask) 
            self.rate = -(cost-self.Cost[-1])/np.abs(self.Cost[-1])
            if verbose:
                print(self.rate)
            if cost>self.Cost[-1]:
                self.Cost.append(cost) 
                raise ValueError('Cost augmente!')
            if np.isnan(cost):
                raise ValueError('Cost NAN')
            elif self.rate<precision and n>min_iter:
                self.Cost.append(cost) 
                break
            self.Cost.append(cost) 

        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        
        self.duration = time.time()-start_time
        # Save
        if self.save:
            self.save_model()

    def filename(self,prefix,suffix):
        if prefix is not None:
            prefix = prefix+'_'
        else:
            prefix = ''
        if suffix is not None:
            suffix = '_'+suffix
        else:
            suffix = ''
        return prefix + self.classname + \
                '_K%d' % (self.K) + \
                '_init_' + self.init + \
                '_tol%.1e' %(self.precision) + \
                '_iter%.1e_%.1e' %(self.min_iter, self.max_iter) + \
                '_seed' + str(self.seed) + suffix
                
    def save_model(self):
        with open(os.path.join(self.save_dir, self.filename), 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
    def generate(self):
        pass
    
    def copy_attributes(self,oobj):
        self.__dict__ = oobj.__dict__.copy()
 
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def kl(Y, Yap, Mask):
    C = Y*np.log(Y/Yap) + Yap - Y
    D = (C*Mask).sum()
    if np.isnan(D):
        print 'NAN DIV'
    return D

def renorm(W,H):
    scale = np.sum(W,0,keepdims=True)
    W = W/scale
    H = H*scale
    return W,H
    
#%% TEST ON SYNTHETIC DATA
if False:
    import matplotlib.pyplot as plt
    U = 100
    I = 100
    K = 5
    
    np.random.seed(2854)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    L = np.dot(W,H.T)
    Y = np.random.poisson(L)
    Mask = np.ones_like(Y)
    Mask = np.random.binomial(1,0.75,(U,I))
    
    assert np.sum(np.sum(Mask*Y!=0,0)==0)==0
    assert np.sum(np.sum(Mask*Y!=0,1)==0)==0
    
    #%% 
    model = pf_MM(K=K)
    model.fit(Y, Mask=Mask,
              seed=10, init='pf_bin',
              verbose=True,
              precision=10**(-3), min_iter = 10**2, prefix='',
              save = False)
    
    #%%
    Ew = model.Ew
    Eh = model.Eh
    plt.figure('Cost')
    plt.loglog(model.Cost)
    
    #%%
    plt.figure('Truth')
    plt.imshow(L,interpolation='nearest')
    plt.colorbar()
    plt.figure('Obs')
    plt.imshow(Y,interpolation='nearest')
    plt.colorbar()
    plt.figure('Mask')
    plt.imshow(Mask,interpolation='nearest')
    plt.colorbar()
    plt.figure('Reconstruction')
    plt.imshow(np.dot(Ew,Eh.T),interpolation='nearest')
    plt.colorbar()
    
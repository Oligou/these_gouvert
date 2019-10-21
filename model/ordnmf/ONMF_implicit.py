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
# c ~ Poisson(V*W*H)        ## UxIxK xN
# y = sum(c)                ## UxI

## Conditional
# W|H,V,c ~ Gamma(aphaW+sum(c), betaW+sum(H))
# c|W,H ~ Mult(y, W*H)

## Order: W,H,C
#%%
import numpy as np
import scipy.special as special
import scipy.sparse as sparse
import os
import time
import cPickle as pickle
import sys
import pandas as pd

class ONMF_implicit():
    def __init__(self, K,
                 alphaW = 1., alphaH = 1., betaW=1., betaH = 1.):
        self.K = K
        self.alphaW = alphaW
        self.alphaH = alphaH
        self.betaW = betaW
        self.betaH = betaH
        self.score={}
        self.classname = 'ONMF_implicit'
        # Save arg
        saved_args_init = locals()
        saved_args_init.pop('self')
        self.saved_args_init = saved_args_init
        
    def fit(self, Y, T, 
            seed=None, 
            opt_hyper = ['beta'],
            approx = False,
            precision=10**(-5), max_iter=10**5, min_iter=0,
            verbose=False,
            save=True, save_dir='', prefix=None, suffix=None):
        
        self.seed = seed
        np.random.seed(seed)
        self.T = T
        self.opt_hyper = opt_hyper
        self.approx = approx
        self.verbose = verbose
        self.precision = precision
        # Save
        self.save = save
        self.save_dir = save_dir
        self.filename = self.filename(prefix, suffix)
        # Save arg
        saved_args_fit = locals()
        saved_args_fit.pop('self')
        saved_args_fit.pop('Y')
        self.saved_args_fit = saved_args_fit
        # Timer
        start_time = time.time()
                
        # Shape
        U,I = Y.shape
        u,i = Y.nonzero()
        # Init matrice compagnon
        delta = np.ones(T+1); delta[0]=0; 
        H = (np.triu(np.ones((T+1,T+1))).dot(delta[:,np.newaxis]))[:,0] 
        theta0 = H[0]
        G = theta0 - H
        Gy = transform_Y(Y,G)
        # Init - W & H
        Ew = np.random.gamma(1.,1.,(U,self.K))
        Eh = np.random.gamma(1.,1.,(I,self.K))
        s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
        
        # Local
        Sw, Sh, En, elboLoc = self.q_loc(Y,delta,Ew,Eh)
            
        self.Elbo = [-float("inf")]
        self.info = []
        for n in range(max_iter):
            # Time
            if verbose:
                print('ITERATION #%d' % n)
                start_t = _writeline_and_time('\tUpdates...')
            # Hyper parameter
            if np.isin('beta',opt_hyper):
                self.betaW = self.alphaW/Ew.mean(axis=1,keepdims=True)
                self.betaH = self.alphaH/Eh.mean(axis=1,keepdims=True)
            if np.isin('betaH',opt_hyper):
                self.betaH = self.alphaH / np.mean(Eh)
            # Updates Delta
            lbd = np.sum(Ew[u,:]*Eh[i,:],1)
            S_lbd = s_wh
            for l in range(T,0,-1): # {T,...,1}
                S_lbd = S_lbd - np.sum(lbd[Y.data==l+1])
                delta[l] = np.sum(En[Y.data==l])/S_lbd
            H = (np.triu(np.ones((T+1,T+1))).dot(delta[:,np.newaxis]))[:,0] 
            theta0 = H[0]
            G = theta0 - H
            Gy = transform_Y(Y,G)
            # Global 
            Ew, Elogw, elboW = q_Gamma(self.alphaW , Sw, 
                                       self.betaW, theta0*np.sum(Eh,0,keepdims=True) - Gy.dot(Eh))
            Eh, Elogh, elboH = q_Gamma(self.alphaH, Sh,
                                       self.betaH, theta0*np.sum(Ew,0,keepdims=True) - Gy.T.dot(Ew))
            s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
            # Local
            Sw, Sh, En, elboLoc = self.q_loc(Y,delta,np.exp(Elogw),np.exp(Elogh))
            # Elbo
            elbo = elboLoc - theta0*s_wh + np.sum(Ew*Gy.dot(Eh)) + elboW + elboH
            self.rate = (elbo-self.Elbo[-1])/np.abs(self.Elbo[-1])
            if verbose:
                print('\r\tUpdates: time=%.2f'% (time.time() - start_t))
                print('\tRate:' + str(self.rate))
            if elbo<self.Elbo[-1]:
                self.Elbo.append(elbo) 
                raise ValueError('Elbo diminue!')
            if np.isnan(elbo):
                #pass
                raise ValueError('elbo NAN')
            elif self.rate<precision and n>=min_iter:
                self.Elbo.append(elbo) 
                break
            self.Elbo.append(elbo) 
            self.info.append(delta.copy())
        
        self.delta = delta
        self.theta = (np.triu(np.ones((T+1,T+1)),1).dot(delta[:,np.newaxis]))[:,0]
        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        self.En = En.copy()
        self.Elogw = Elogw.copy()
        self.Elogh = Elogh.copy()
        
        self.duration = time.time()-start_time
        
        # Save
        if self.save:
            self.save_model()
     
    def q_loc(self,Y,delta,W,H):
        # Product
        u,i = Y.nonzero()
        Lbd = np.sum(W[u,:]*H[i,:],1)
        delta_y = transform_Y(Y,delta).data
        # En
        if self.approx == False:
            en = Lbd*delta_y/(1.-np.exp(-Lbd*delta_y))#delta_y/(1.-np.exp(-Lbd*delta_y))
            en[np.isnan(en)] = 1.
        else :
            en = np.ones_like(Lbd)
        # Sum C
        R = sparse.csr_matrix((en/Lbd,(u,i)),shape=Y.shape) # UxI
        Sw = W*(R.dot(H)) 
        Sh = H*(R.T.dot(W))
        # ELBO
        elbo = np.sum(np.log(np.expm1(Lbd*delta_y)))
        return Sw, Sh, en, elbo

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
                '_T%d' % (self.T) + \
                '_alpha%.2f_%.2f' %(self.alphaW, self.alphaH) + \
                '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
                '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
                '_approxN_' + str(self.approx) + \
                '_tol%.1e' %(self.precision) + \
                '_seed' + str(self.seed) + suffix
            
    def save_model(self):
        with open(os.path.join(self.save_dir, self.filename), 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
    def generate(self):
        pass
    
    def copy_attributes(self,oobj):
        self.__dict__ = oobj.__dict__.copy()
                
def stat_gamma(shape,rate):
    E = shape/rate
    dig_shape = special.digamma(shape)
    Elog = dig_shape - np.log(rate)
    entropy = shape - np.log(rate) + special.gammaln(shape) + (1-shape)*dig_shape
    return E, Elog, entropy
  
def gamma_elbo(shape, rate, Ex, Elogx):
    return (shape-1)*Elogx -rate*Ex +shape*np.log(rate) -special.gammaln(shape)

def q_Gamma(shape, _shape, rate, _rate):
    E,Elog,entropy = stat_gamma(shape+_shape, rate+_rate)
    elbo = gamma_elbo(shape, rate, E, Elog)
    elbo = elbo.sum() + entropy.sum()
    return E, Elog, elbo
 
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def transform_Y(Y,values): # 1->values[1]; 2->values[2]; ...
    transformation = Y.copy()
    transformation = transformation.astype(float)
    for l in range(1,len(values)):
        transformation.data[Y.data==l] = values[l]
    return transformation

#%%
if False:
    import matplotlib.pyplot as plt

    U = 100
    I = 100
    K = 3
    M = 10
    
    np.random.seed(98)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    La = .1*np.dot(W,H.T)
    Ya = np.random.poisson(La)
    Y = sparse.csr_matrix(Ya)
        
    #%%
    model = ONMF_implicit(K=K)
    model.fit(Y, T=Y.max(), precision=10**(-5), verbose=True, approx=False)
    print model.Elbo[-1]
    
    #%%
    Ew = model.Ew
    Eh = model.Eh            
    Yr = np.dot(Ew,Eh.T)
    
    #%%
    plt.figure('Obs')
    plt.imshow(Ya,interpolation='nearest', aspect = 'auto')
    plt.colorbar()
    
    #%%
    plt.figure('True Low-Rank')
    plt.imshow(La,interpolation='nearest', aspect = 'auto')
    plt.colorbar()
    
    #%%
    plt.figure('Reconstruction')
    plt.imshow(Yr,interpolation='nearest', aspect = 'auto')
    plt.colorbar()
    
    #%%
    plt.figure()
    plt.plot(model.Elbo)
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
from pf_vi import pf_vi

class nbf_vi():
    def __init__(self,K,alphaNB,
                 alphaW = 1.,alphaH = 1., betaW=1., betaH=1.):
        self.K = K
        self.alphaNB = alphaNB
        self.alphaW = alphaW
        self.alphaH = alphaH
        self.betaW = betaW
        self.betaH = betaH
        self.score={}
        self.classname = 'nbf_vi' 
        # Save arg
        self.saved_args_init = locals()
        del self.saved_args_init['self']
        
    def fit(self,Y, 
            opt_hyper=[], 
            seed=None, init='rand',
            precision=10**(-5), min_iter=0, max_iter=10**5,
            verbose=False,
            save=True, save_dir='', prefix=None, suffix=None):
        
        self.seed = seed
        if isinstance(seed, int):
            np.random.seed(seed)
        self.init = init
        self.opt_hyper = opt_hyper
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
        del self.saved_args_fit['self']
        # Timer
        start_time = time.time()
        
        # CONSTANTS
        U,I = Y.shape
        s_y = Y.sum()
        self.Elbo = [-float("inf")]
        elbo_cst = -np.sum(special.gammaln(Y.data+1))
        self.shapeV = self.alphaNB + Y.A
        elboV_cst = np.sum(Y.data + special.gammaln(self.alphaNB+Y.data) - special.gammaln(self.alphaNB))
        
        # INIT
        if init=='pf':
            model_init = pf_vi(K=self.K,
                               alphaW = self.alphaW, alphaH = self.alphaH)
            model_init.fit(Y, 
                          precision=precision*10, min_iter=min_iter/10, max_iter=max_iter/10,
                          save=True, save_dir=save_dir, prefix='INIT_pf_'+str(prefix))
            Ew = model_init.Ew
            Eh = model_init.Eh
            self.model_init = model_init
        elif init=='pf_bin':
            model_init = pf_vi(K=self.K,
                               alphaW = self.alphaW, alphaH = self.alphaH)
            model_init.fit(Y>0, 
                           precision=precision*10, min_iter=min_iter/10, max_iter=max_iter/10,
                           save=True, save_dir=save_dir, prefix='INIT_pfbin_'+str(prefix))
            Ew = model_init.Ew
            Eh = model_init.Eh
            self.model_init = model_init
        else:
            Ew = np.random.gamma(1.,1.,(U,self.K))
            Eh = np.random.gamma(1.,1.,(I,self.K))
        s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
        Ev,elboV = self.q_V(Ew,Eh)
        Sw, Sh, elboLoc = self.q_Mult(Y,Ew,Eh)
        
        for n in range(max_iter):    
            if verbose:
                print('ITERATION #%d' % n)
                start_t = _writeline_and_time('\tUpdates...')
            # HYPER
            if np.isin('beta',opt_hyper):
                self.betaW = self.alphaW/Ew.mean(axis=1,keepdims=True)
                self.betaH = self.alphaH/Eh.mean(axis=1,keepdims=True)
            if np.isin('betaH',opt_hyper):
                self.betaH = self.alphaH / np.mean(Eh)
            # Global 
            Ew, Elogw, elboW = q_Gamma(self.alphaW , Sw, 
                                       self.betaW, np.dot(Ev,Eh))
            Eh, Elogh, elboH = q_Gamma(self.alphaH, Sh,
                                       self.betaH, np.dot(Ev.T,Ew))
            s_wh = np.dot(np.sum(Ew,0,keepdims=True),np.sum(Eh,0,keepdims=True).T)[0,0]
            # Local
            Ev,elboV = self.q_V(Ew,Eh)
            Sw, Sh, elboLoc = self.q_Mult(Y,np.exp(Elogw),np.exp(Elogh))
            ## ELBO
            elbo = elboLoc - np.sum(Ev*np.dot(Ew,Eh.T)) + \
                    elboW + elboH + elboV + elbo_cst + elboV_cst
            self.rate = (elbo-self.Elbo[-1])/np.abs(self.Elbo[-1])
            if verbose:
                print('\r\tUpdates: time=%.2f'% (time.time() - start_t))
                print('\tRate:' + str(self.rate))
            if self.rate<-10**(-8):
                self.Elbo.append(elbo) 
                raise ValueError('Elbo diminue!')
            if np.isnan(elbo):
                raise ValueError('elbo NAN')
            elif self.rate<precision and n>min_iter:
                self.Elbo.append(elbo) 
                break
            self.Elbo.append(elbo) 

        del self.shapeV
        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        
        self.duration = time.time()-start_time
        # Save
        if self.save:
            self.save_model()

    def q_V(self,Ew,Eh):
        rate = self.alphaNB + np.dot(Ew,Eh.T)
        E = self.shapeV/rate
        elbo = -self.shapeV*(np.log(rate) + self.alphaNB/rate) + self.alphaNB*np.log(self.alphaNB) + self.alphaNB 
        return E, np.sum(elbo)
    
    def q_Mult(self,Y,W,H):
        # Product
        u,i = Y.nonzero()
        Ydata = Y.data
        s = np.sum(W[u,:]*H[i,:],1)     
        # Mult
        R = sparse.csr_matrix((Ydata/s,(u,i)),shape=Y.shape) # UxI
        Sh = ((R.T).dot(W))*H 
        Sw = (R.dot(H))*W 
        elbo = np.sum(Ydata*np.log(s))
        return Sw, Sh, elbo 

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
                '_alphaNB%.2f' % (self.alphaNB) + \
                '_alpha%.2f_%.2f' %(self.alphaW, self.alphaH) + \
                '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
                '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
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

#%% TEST ON SYNTHETIC DATA
if False:
    import matplotlib.pyplot as plt
    U = 500
    I = 500
    K = 5
    
    np.random.seed(2854)
    W = np.random.gamma(1.,1., (U,K))
    H = np.random.gamma(1.,1., (I,K))
    L = np.dot(W,H.T)*0.1
    Ya = np.random.poisson(L)
    Y = sparse.csr_matrix(Ya)
    
    #%% 
    model = nbf_vi(K=K,alphaNB=.1)
    model.fit(Y, 
              seed=10, init='pf',
              verbose=True, min_iter=10**(3),
              save = False)
    
    #%%
    Ew = model.Ew
    Eh = model.Eh
    plt.figure('Elbo')
    plt.plot(model.Elbo)
    
    #%%
    plt.figure('Truth')
    plt.imshow(L,interpolation='nearest')
    plt.colorbar()
    plt.figure('Obs')
    plt.imshow(Ya,interpolation='nearest')
    plt.colorbar()
    plt.figure('Reconstruction')
    plt.imshow(np.dot(Ew,Eh.T),interpolation='nearest')
    plt.colorbar()
    
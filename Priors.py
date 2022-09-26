#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All implemented priors are included here

Contains:
    Gaussian prior
    Jeffreys prior
    Modified Jeffreys prior
    Uniform prior

Author: Federica Rescigno
Version: 27.05.2022
'''

import numpy as np
import abc
ABC= abc.ABC

def Prior_list():
    PRIORS = {
        "Gaussian": ['param', 'mu', 'sigma'],
        "Jeffreys": ['param', 'minval', 'maxval'],
        "Modified_Jeffreys": ['param', 'minval', 'maxval', 'kneeval'],
        "Uniform": ['param', 'minval', 'maxval']}
    return PRIORS


def PrintPriorList():
    print("Implemented priors:")
    PRIORS = Prior_list()
    print(PRIORS)


###################################
########## PARENT PRIOR ##########
###################################

class Prior:
    '''Parent Class for all priors. All new priors should inherit from this class and follow its structure.
    
    Each new prior will require a __init__ method to override the parent class. In the __init__ function
    call the necessary parameters to define the prior.'''
    
    @abc.abstractproperty
    def name(self):
        pass
    
    @abc.abstractproperty    
    def __repr__(self):
        pass
    
    @abc.abstractmethod
    def logprob(self, x):
        pass


###################################
########## GAUSSIAN PRIOR ##########
###################################


class Gaussian(Prior):
    '''Gaussian prior computed as:
        
        -0.5 * ((x - mu) / sigma)**2 -0.5 * np.log(2*pi * sigma**2)
        
    Args:
        hparam (string): parameter label
        mu (float): centre of Gaussian Prior
        sigma (float): width of the Gaussian Prior
    '''
    
    def __init__(self, param, mu, sigma):
        '''
        Parameters
        ----------
        param : string
            Label of parameter on which the prior will be applied
        mu : float
            Center of the Gaussian prior
        sigma : float
            FWHM of the Gaussian prior
        '''
        self.param = param
        self.mu = float(mu)
        self.sigma = float(sigma)
        
    @property
    def name(self):
        return("Gaussian Prior")   
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior.
        '''
        message = ("Gaussian prior on the parameter {}, with mu = {} and sigma = {}").format(self.param, self.mu, self.sigma)
        print(message)
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        
        logprob = -0.5 * ((x - self.mu) / self.sigma)**2 -np.log(np.sqrt(2*np.pi) * self.sigma)
        return logprob


###################################
########## JEFFREYS PRIOR ##########
###################################


class Jeffreys(Prior):
    '''Jeffreys prior computed as:
        
        p(x) proportional to  1/x
        with upper and lower bound to avoid singularity at x = 0
        
        and normalized as:
            1 / ln(maxval/minval)
    
    Args:
        param (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    def __init__(self, param, minval, maxval):
        '''
        Parameters
        ----------
        param : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        '''
        self.param = param
        self.minval = minval
        self.maxval = maxval
        
        assert self.minval < self.maxval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
    
    @property
    def name(self):
        return("Jeffreys Prior")
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Jeffreys prior on the paramter {}, with minimum and maximum values of x = ({}, {})").format(self.param, self.minval, self.maxval)
        print(message)
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -np.inf
            return logprob
        else:
            normalisation = 1./(np.log(self.maxval/self.minval))
            prob = normalisation * 1./x
            logprob = np.log(prob)
            return logprob



#############################################
########## MODIFIED JEFFREYS PRIOR ##########
#############################################


class Modified_Jeffreys(Prior):
    ''' Modified Jeffreys prior computed as:
        
        p(x) proportional to  1/(x-x0)
        with upper bound
    
    
    Args:
        param (string): parameter label
        kneeval (float): x0, knee of the Jeffrey prior
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    
    def __init__(self, param, minval, maxval, kneeval):
        '''
        Parameters
        ----------
        param : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        kneeval: float
            Kneww value of prior (x0)
        '''
        self.param = param
        self.minval = minval
        self.maxval = maxval
        self.kneeval = kneeval
        
        assert self.minval < self.maxval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
        
    @property
    def name(self):
        return("Modified Jeffreys Prior")
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Modified Jeffreys prior on the paramter {}, with minimum and maximum values of x = ({}, {}) and knee = {}").format(self.hparam, self.minval, self.maxval, self.kneeval)
        print(message)
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : float
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -np.inf
            return logprob
        else:
            normalisation = 1./(np.log((self.maxval+self.kneeval)/(self.kneeval)))
            prob = normalisation * 1./(x-self.kneeval)
            logprob = np.log(prob)
            return logprob


###################################
########## UNIFORM PRIOR ##########
###################################


class Uniform(Prior):
    ''' Uniform prior
    
    Args:
        param (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    
    def __init__(self, param, minval, maxval):
        '''
        Parameters
        ----------
        param : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        '''
        self.param = param
        self.minval = minval
        self.maxval = maxval
    
        assert self.maxval > self.minval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
    
    
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Uniform prior on the paramter {}, with minimum and maximum values of x = ({}, {})").format(self.hparam, self.minval, self.maxval)
        print(message)
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -np.inf
            return logprob
        else:
            logprob = 0
            return logprob



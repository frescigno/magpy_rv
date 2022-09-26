#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Check file. This file contains the function to check whether to
approve an MCMC step or not.

Contains:
    get_model

Author: Federica Rescigno
Version: 12.07.2022
'''

import numpy as np
import Models
import Auxiliary as aux
import Get as get


class Parameter_Check:
    
    
    def __init__(self, hparam, kernel_name, mod_param, model_names, x):
        '''
        Parameters
        ----------
        hparam : array, floats
            Array containing all hyperparameters of the chosen kernels
        kernel_name : list of strings, or string
            Name of the kernel(s) chosen
        mod_param : array, floats
            Array containing all model parameters of the chosen models
        model_name : list of strings, or string
            Name of the model(s) chosen
        x : array, floats
            x-array of the observations
        '''
        
        self.hparam = hparam
        self.kernel_name = kernel_name
        if isinstance(self.kernel_name, str):
            self.kernel_name = [self.kernel_name]
        self.mod_param = mod_param
        self.model_names = model_names
        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]
        self.x = x
        self.check = True       # Automatic pass 
    
    
    def no_negative(self, Kernel=True):
        ''' No negative check requires all values in the parameter array to be
        larger than zero
        
        Parameters
        ----------
        Kernel : boolean, optional
            Are we considering kernel or model? If True, we consider the kernel parameters.
            If False, we consider the model parameters.
        
        Returns
        ----------
        check : boolean
            Check passed or not?
        '''
        if Kernel:
            if np.min(self.hparam) < 0.:
                check = False
        if not Kernel:
            if np.min(self.mod_param) < 0.:
                check = False
        return check
        
    
    def kepler_check(self):
        ''' Find any keplerian model and check the viability of Sk and Ck and of the other
        keplerian parameters
        
        Returns
        ----------
        check : boolean
            Check passed or not?
        '''
        # Count the number of parameters for each model to make sure we are picking the right ones
        current_par = 0
        for name in self.model_names:
            numb_param_model = get.get_model(name, self.x, self.mod_param, get_number=True)
            if name.startswith("kepl") or name.startswith("Kepl"):
                
                # Period, amplitude and t0 must be positive
                if self.mod_param[current_par]< 0. or self.mod_param[current_par+1]<0. or self.mod_param[current_par+4]<0.:
                    check = False
                    return check
                
                # Sk and Ck can be negative, but the sum of their squares must be less than 1
                if (self.mod_param[current_par+2]**2 + self.mod_param[current_par+3]**2) > 1.:
                    check = False
                    return check
            # Add to the current number to pick the proper parameters
            current_par += numb_param_model
    
    def starcross_check(self, Rstar, Mstar):
        ''' Check that the planets do not fall into the star
        
        Parameters
        ----------
        Rstar : float
            Radius of the star in Solar radii
        Mstar : float
            Mass of the star in Solar masses
        
        Returns
        ----------
        check : boolean
            Check passed or not?
        '''
        
        # Attention, assumes period is in days
        # Count the number of parameters for each model to make sure we are picking the right ones
        current_par = 0
        for name in self.model_names:
            numb_param_model = get.get_model(name, self.x, self.mod_param, get_number=True)
            if name.startswith("kepl") or name.startswith("Kepl"):
                ecc_pl = self.mod_param[current_par+2]**2 + self.mod_param[current_par+3]**2
                Rsun = 6.95508e8    # in meters 
                AU = 149597871e3    # in meters
                # Check is based on whether this 1 - ratio is smaller than the eccentricity
                ratio = (Rstar*Rsun/AU)/(((self.mod_param[current_par]/365.23)**2) * Mstar)**(1/3)

                if ecc_pl < 1 - ratio:
                    check = True
                    return check
                if ecc_pl >= 1 - ratio:
                    check = False
                    return check
            # If the model is not a keplerian, skip and add the skipped number of parameters
            current_par += numb_param_model
    
    
    def planetcross_check(self):
        ''' Check if the planet orbits cross eachother
        
        ~ TO COME
        '''
        check = True
        return check
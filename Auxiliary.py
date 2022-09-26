#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Big Auxiliary file

Contains:
    

Author: Federica Rescigno
Version: 27.06.2022
'''

import numpy as np



def to_ecc(Sk, Ck, errSk=None, errCk=None):
    ''' Function to go from Sk and Ck to ecc and omega values, with errors if necessary.
    
    Paramters
    ----------
    Sk : float
        Sk value
    Ck : float
        Ck value
    errSk : float, optional
        error on Sk, default is None. When both this value and errCk are given,
        the errors on ecc and omega are calculated
    errCk : float, optional
        error on Ck, default is None. When both this value and errSk are given,
        the errors on ecc and omega are calculated
    '''
    ecc = Sk**2 + Ck**2
    # If circlar orbit assume periastron at 90deg
    if ecc == 0.:
        omega = np.pi/2
    else:
        omega = np.arctan(Sk/Ck)
    
    # Compute errors if both errors are given
    if errSk is not None and errCk is not None:
        # Errors derived by basic error propagation
        errecc = np.sqrt(errSk**2 * 4*Sk**2 + errCk**2 * 4*Ck**2)
        erromega = np.sqrt(errSk**2 * (Ck/(Ck**2 + Sk**2)**2 + errCk**2 * (-Sk/(Ck**2 + Sk**2)**2)))
        
        return ecc, omega, errecc, erromega
    
    return ecc, omega




def to_SkCk(ecc, omega, ecc_err, omega_err):
    ''' Function to go from eccentricity and omega to Sk and Ck.
    Follows Ford+2006
    ecc = Sk^2 + Ck^2
    omega = arctan(Sk/Ck)

    Parameters
    ----------
    ecc : float
        Eccentricity
    omega : float, radians
        Angle of periastron
    ecc_err : float
        Error on the eccentricity
    omega_err : float
        Error on angle of periastron

    Returns
    -------
    Sk : float
        Sk value
    Ck : float
        Ck value
    Sk_err : float
        Error on Sk
    Ck_err : float
        Error on Ck
    '''
    
    Sk = np.sqrt(ecc) * np.sin(omega)
    Ck = np.sqrt(ecc) * np.cos(omega)
    
    if ecc == 0.:
        Sk_err = ecc_err
        Ck_err = ecc_err
    else:
        Sk_err = np.sqrt((ecc_err**2 * (np.sin(omega))**2 / (4*ecc)) + (omega_err**2 * ecc * (np.cos(omega))**2))
        Ck_err = np.sqrt((ecc_err**2 * (np.cos(omega))**2 / (4*ecc)) + (omega_err**2 * ecc * (np.sin(omega))**2))
    
    return Sk, Ck, Sk_err, Ck_err



def initial_dist_creator(params, params_err, numb_chains, vary_params = None, allow_neg = False):
    ''' Function to populate the number of chains chosen, starting from the initial guess parameters
    and their errors

    Parameters
    ----------
    param : list, floats
        List of the initial guess parameters
    param_err : list, floats
        List of the errors on the initial guess parameters
    numb_chains : int
        Number of chains
    vary_params : list, boolean
        Can I vary the parameter? If None all parameters are varied. Default is None
    allow_neg : boolean
        Allow negative starting values. Default is False
        
    Returns
    -------
    chains_param : 2D list, floats
        2D array of number params x numb chains

    '''
    chains_params = []
    # For the first chain, use the guesses themselves
    chains_params.append(params)
    
    # For the rest create them by multipling a random number from gaussian distribution
    # between -1 and 1 and modulate by the error
    for l in range(numb_chains-1):
        pos = params + params_err * np.random.normal(0., 1., (1,len(params)))
    
        # In case we do not want to allow for negative numbers
        if not allow_neg:
            while np.min(pos) < 0:
                pos = params + params_err * np.random.uniform(-1.,1.,(1,len(params)))

        # If a vary array is given, only create new starting parameters if the parameters
        # is actually allowed to vary, otherwise stick with initial guess
        if vary_params is not None:
            for vary, i in zip(vary_params, len(vary_params)):
                if vary:
                    pos[i] = pos[i]
                if not vary:
                    pos[i] = params[i]
        
        # Append this new chain set
        chains_params.append(pos[0])
    
    # chains_params should have on the horizontal the parameter values for each chain
    # on the vertical the number of chains
    return chains_params
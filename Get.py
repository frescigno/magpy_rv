#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
All "get" functions.

Contains:
    get_numb_params
    get_model

Author: Federica Rescigno
Version: 14.07.2022
'''

import numpy as np
import Models
import Auxiliary as aux




def get_mod_numb_params(model_name):
    ''' Function to give the number of expected parameters per model
    
    Parameters
    ---------
    model_name : string
        Name of the considered model
    
    Returns
    ---------
    model_param_number : int
        Number of parameter per model
    '''
    
    if model_name.startswith("no") or model_name.startswith("No"):
        model_param_number = 1
    if model_name.startswith("off") or model_name.startswith("Off"):
        model_param_number = 1
    if model_name.startswith("kepl") or model_name.startswith("Kepl"):
        model_param_number = 5
    if model_name.startswith("lin") or model_name.startswith("Lin"):
        model_param_number = 2
    if model_name.startswith("unc") or model_name.startswith("Unc"):
        model_param_number = 1
    
    return model_param_number


###########################
########## MODEL ##########
###########################


def get_model(model_name, time, model_par, to_ecc=False):
    '''
    Function that generates and sums all the models requested.
    Also saves the number for parameters expected
    
    Parameters
    ----------
    model_name : list of strings
        Name of model used
    time : array, floats
        Time array over which to calculate the model
    model_par : dictionary
        Set of parameters (within the parameter object) with which to compute the model
    to_ecc : boolean, optional
        Do we need to trasform from Sk and Ck to ecc and omega? Default is False
   
    Returns
    -------
    model_y : array, floats
        Radial velocity of the model
    optional
        tot_par_numb : int
            Total number of model parameters required for the chosen set of models
    '''
    
    model_y = np.zeros(len(time))
    current_par_numb = 0
    zero_mod=0
    off_mod=0
    kepl_mod=0
    lintrend_mod=0
    uncorr_mod=0
    
    # Loop trough all models
    for mod in model_name:
        model_param_number = get_mod_numb_params(mod)
        parameters ={key: value for key, value in model_par.items() if (list(model_par).index(key) >= current_par_numb and list(model_par).index(key) < current_par_numb+model_param_number)}
        
        if mod.startswith("zero") or mod.startswith("Zero") or mod.startswith("no"):
            model = Models.Zero_Model(time, parameters)
            model_y += model.model()
            zero_mod += 1
        
        elif mod.startswith("Offset") or mod.startswith("offset"):
            model = Models.Offset(time, parameters)
            model_y += model.model()
            off_mod += 1
        
        elif mod.startswith("Kepler") or mod.startswith("kepler"):
            if to_ecc:
                # Do naming dance, and turn Sk into Ck into ecc and omega
                try:
                    parameters['ecc'].value, parameters['omega'].value = aux.to_ecc(parameters['ecc'].value, parameters['omega'].value)
                except KeyError:
                    try:
                        parameters['ecc_'+str(kepl_mod)].value, parameters['omega_'+str(kepl_mod)].value = aux.to_ecc(parameters['ecc_'+str(kepl_mod)].value, parameters['omega_'+str(kepl_mod)].value)
                        break
                    except KeyError:
                        raise KeyError("Issue with the naming of parameters for transformation to eccentricty")
            model = Models.Keplerian(time, parameters)
            model_y += model.model()
            kepl_mod += 1
        
        elif mod.startswith("Lin") or mod.startswith("lin"):
            model = Models.LinTrend_Model(time, parameters)
            model_y += model.model()
            lintrend_mod += 1
        
        elif mod.startswith("Uncor") or mod.startswith("uncor"):
            model = Models.UncorrNoise_Model(time, parameters)
            model_y += model.model()
            uncorr_mod += 1
            
        else:
            raise ValueError("There is no such implemented model for {}".format(mod))
        
        current_par_numb += model_param_number

    return model_y
        
            



'''
Auxiliary functions for the MCMC code to use

Contains:
    Number of expected parameters per model function
    Get model function
    Parameter check function
    

Author: Bryce Dixon
Version 21.07.2023
'''

import numpy as np
import Models as mod
import auxiliary as aux

def numb_param_per_model(model_name):
    ''' Function to give the number of expected parameters per model
    '''
    if model_name.startswith("no") or model_name.startswith("No"):
        model_param_number = mod.No_Model.numb_param()
    if model_name.startswith("off") or model_name.startswith("Off"):
        model_param_number = mod.Offset.numb_param()
    if model_name.startswith("kep") or model_name.startswith("Kep"):
        model_param_number = mod.Keplerian.numb_param()
    if model_name.startswith("poly") or model_name.startswith("Poly"):
        model_param_number = mod.Polynomial.numb_param()
    return model_param_number



def get_model(model_name, time, model_par, to_ecc=False, flags=None):
    '''
    Parameters
    ----------
    model_name : list of strings
        Name of model used
    time : array, floats
        Time array over which to calculate the model
    model_par : dictionary
        Set of parameters (within the parameter object) with which to compute the model

    Returns
    -------
    model_y : array, floats
        Radial velocity of the model
    '''
    
    MODELS = mod.defModelList()
    
    model_y = np.zeros(len(time))
    i=0
    a=0
    for name in model_name:
        numb_param_mod = numb_param_per_model(name)
        parameters ={key: value for key, value in model_par.items() if (list(model_par).index(key) >= i and list(model_par).index(key) < i+numb_param_mod)}
        if name.startswith("no") or name.startswith("No"):
            model = mod.No_Model(time, parameters)
        elif name.startswith("off") or name.startswith("Off"):
            model = mod.Offset(flags, parameters)
        elif name.startswith("poly") or name.startswith("Poly"):
            model = mod.Polynomial(time, parameters)
        elif name.startswith("kep") or name.startswith("Kep"):
            if to_ecc:
                if len(model_name) == 1:
                    parameters['ecc'].value, parameters['omega'].value = aux.to_ecc(parameters['ecc'].value, parameters['omega'].value)
                else:
                    parameters['ecc_'+str(a)].value, parameters['omega_'+str(a)].value = aux.to_ecc(parameters['ecc_'+str(a)].value, parameters['omega_'+str(a)].value)
            model = mod.Keplerian(time, parameters)
            a +=1
        else:
            raise KeyError("model not yet implemented, please from currently implemented models: " + str(MODELS.keys()))
        model_y += model.model()
        i += numb_param_mod
        
    parameter=None
    return model_y


def parameter_check(parameters, names, Rstar=None, Mstar=None):
    ''' Function to check if the parameters are within the bounds
    
    Parameters
    ----------
    parameters : array, floats
        Array of parameters for all models
    names : list of strings
        Names of all the models used, can be one or more
    Rstar : float, optional
        Radius of the star in solar radii. Default is None. Needed for the orbit check
    Mstar : float, optional
        Mass of the star in solar masses. Default is None. Needed for the orbit check

    Returns
    -------
    check : bool
        Are all paramaters physically possible?
    '''
    check = True
    o = 0
    for name in names:
        numb_params = numb_param_per_model(name)
            
        if name.startswith('kepl') or name.startswith('Kepl'):
            # Period, amplitude and t0 must be positive
            if parameters[o]< 0. or parameters[o+1]<0. or parameters[o+4]<0.:
                check = False
                return check
            
            # Sk and Ck can be negative, but the sum of their squares must be less than 1
            if ((parameters[o+2]**2 + parameters[o+3]**2) > 1.): # or (parameters[o+2] < 0.) or (parameters[o+3] < 0.):
                check = False
                return check
            
            if Rstar is not None and Mstar is not None:
                # Check planet does not fall into the star
                ####### NOW COLLED STAR CROSS!!!!!
                orbit_check = aux.orbit_check(parameters[o+2], parameters[o+3], Rstar, parameters[o], Mstar)
                if not orbit_check:
                    check = False
                    return check
            
        o += numb_params
    
    return check
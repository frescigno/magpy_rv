'''
Auxiliary functions for the MCMC code

Contains:
    numb_param_per_model
        Number of expected parameters per model function
    get_model
        Get model to use
    initial_pos_creator
        generates initial conditions for multiple chains in the mcmc
    star_cross
        check that the planet orbit is larger than the radius of the star
    parameter_check
        check if the parameters are acceptable
    

Author: Federica Rescingo, Bryce Dixon
Version 22.08.2023
'''
import numpy as np

import magpy_r.models as mod
import magpy_r.auxiliary as aux


def numb_param_per_model(model_name):
    ''' Function to give the number of expected parameters per model
    
    Parameters
    ----------
    model_name : string
        Name of the model
        
    Returns
    -------
    model_param_number : int
        number of parameter required in the model
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



def initial_pos_creator(param, param_err, numb_chains, allow_neg = False, param_names=None):
    '''

    Parameters
    ----------
    param : list, floats
        List of the initial guess parameters
    param_err : list, floats
        List of the errors on the initial guess parameters
    numb_chains : int
        Number of chains
    allow_neg : boolean
        Allow negative starting values. Default is False
    Returns
    -------
    chains_param : 2D list, floats
        2D array of

    '''
    
    chains_param = np.zeros(shape = (1, numb_chains, len(param)))
    # For the first chain, use the guesses themselves
    chains_param[0,0,] = param
    
    # For the rest create them by multipling a random number between -1 and 1
    for l in range(numb_chains-1):
        pos = param + param_err * np.random.uniform(-1.,1.,(1,len(param)))
        # Fix double parenthesis
        #print(pos)
        #pos.tolist()
        if not allow_neg:
            while np.min(pos) < 0:
                if param_names is None:
                    pos = param + param_err * np.random.uniform(-1.,1.,(1,len(param)))
                elif param_names is not None:
                    #print("in")
                    for i in range(len(param_names)):
                        if param_names[i].startswith("ecc") or param_names[i].startswith("omega"):
                            #print("ecc or omega")
                            pass
                        else:
                            while pos[0][i] < 0:
                                pos[0][i] = param[i] + param_err[i] * np.random.uniform(-1.,1.,(1,len(param[i])))
                                #print("pos", pos)
        chains_param[0,l+1,] = pos[0]
    
    return chains_param




def star_cross(Sk, Ck, Rstar, P, Mstar):
    '''
    Parameters
    ----------
    Sk : float
        Sk value from MCMC
    Ck : float
        Ck value from MCMC
    Rstar : float
        Radius of the host star, in Solar Radii
    P : float
        Period of planet, in days
    Mstar : float
        Mass of the star, in Solar Masses

    Returns
    -------
    bool
        If True, the semi-major orbit axes does never fall into the star
        If False, the orbit falls into the star and the step should be dismissed

    '''
    ecc_pl = Sk**2 + Ck**2
    Rsun = 6.95508e8    # in meters 
    AU = 149597871e3    # in meters
    ratio = (Rstar*Rsun/AU)/(((P/365.23)**2) * Mstar)**(1/3)
    
    if ecc_pl < 1 - ratio:
        return True
    if ecc_pl >= 1 - ratio:
        return False
 
 
 

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
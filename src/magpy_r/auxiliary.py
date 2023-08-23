"""
Set of auxiliary functions for the GP_solar and MCMC codes.

Contains:
    printProgressBar
        Creates the progress bar for the MCMC iterations
    
    transit_to_periastron
        transforms from transit time to periastron time
    periastron_to_transit
        transforms from periastron time to transit time
    
    to_SkCk
        transforms eccentricity and omega into Sk and Ck (as defined in Rescigno et al. 2023)
    to_ecc
        transforms Sk and Ck to eccentricity and omega
    
    mass_calc
        computes masses from RV information
    
    transpose
        transposes list
    
    model_param_names
        outputs the name of the parameters in the chosen models
    hparam_names
        outputs the name of the parameters in the chosen kernel
    
    phasefold
        phasefolds data
    
    

Author: Federica Rescigno
Last Updated: 22.08.2023
"""

import numpy as np

import magpy_r.kernels as ker
import magpy_r.models as mod


def printProgressBar (iteration, total, prefix = 'Progress: ', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    ''' Call in a loop to create terminal progress bar
    Parameters
    ----------
    iteration:  integer
        Current iteration
    total:      integer
        Total expected iterations
    prefix:     string, optional
        String before progress bar
    suffix:     string, optional
        String after percentage
    decimals:   integer, optional
        Number of decimals in the percetage
    length:     integer, optional
        Character lenght of the progress bar
    fill:       string, optional
        Bar fill character
    printEnd:   string, optional
        End character (e.g. "\r", "\r\n")
    '''
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()





def transit_to_periastron(t_tr, P, ecc, omega):
    '''
    Parameters
    ----------
    t_tr : float
        Value of the time of transit (start of transit)
    P : float
        Period of the planet
    ecc : float
        Eccentricity of orbit
    omega : float
        Argument of periastron

    Returns
    -------
    t_0 : float
        Time of periastron

    '''
    v_tr = np.pi/2 - omega
    E_tr = 2 * np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(v_tr/2))
    t_0 = t_tr - (P/(2*np.pi) * (E_tr - ecc*np.sin(E_tr)))
    
    return t_0




def periastron_to_transit(t_0, P, ecc, omega):
    '''
    Parameters
    ----------
    t_0 : float
        Time of periastron
    P : float
        Period of the planet
    ecc : float
        Eccentricity of orbit
    omega : float
        Argument of periastron

    Returns
    -------
    t_tr : float
        Value of the time of transit (start of transit)

    '''
    v_tr = np.pi/2 - omega
    E_tr = 2 * np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(v_tr/2))
    t_tr = t_0 + (P/(2*np.pi) * (E_tr - ecc*np.sin(E_tr)))
    
    return t_tr
   
    


def to_SkCk(ecc, omega, ecc_err=None, omega_err=None):
    '''

    Parameters
    ----------
    ecc : float
        Eccentricity
    omega : float, radians
        Angle of periastron
    ecc_err : float, optional
        Error on the eccentricity, defaults to None
    omega_err : float, optional
        Error on angle of periastron, defaults to None

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
    
    if ecc_err is not None and omega_err is not None:
        if ecc == 0.:
            Sk_err = ecc_err
            Ck_err = ecc_err
        else:
            Sk_err = np.sqrt((ecc_err**2 * (np.sin(omega))**2 / (4*ecc)) + (omega_err**2 * ecc * (np.cos(omega))**2))
            Ck_err = np.sqrt((ecc_err**2 * (np.cos(omega))**2 / (4*ecc)) + (omega_err**2 * ecc * (np.sin(omega))**2))
            
        return Sk, Ck,Sk_err, Ck_err
    else:
        return Sk, Ck



def to_ecc(Sk, Ck, errSk=None, errCk=None):
    '''
    
    Parameters
    ----------
    Sk : float
        sqr(e)sin(omega)
    Ck : float
        sqr()cos(omega)
    errSk : float, optional
        error on Sk. Default None
    errCk : float, optional
        error on Ck. Default None
    
    Returns
    -------
    ecc : float
        Eccentricity
    omega : float, radians
        Angle of periastron
    ecc_err : float, optional
        Error on the eccentricity
    omega_err : float, optional
        Error on angle of periastron
    '''
    ecc = Sk**2 + Ck**2
    if ecc == 0.:
        omega = np.pi/2
    else:
        #omega = np.arctan(Sk/Ck)
        omega = np.arctan2(Sk, Ck)
    
    if errSk is not None and errCk is not None:
        errecc = np.sqrt(errSk**2 * 4*Sk**2 + errCk**2 * 4*Ck**2)
        erromega = np.sqrt(errSk**2 * (Ck/(Ck**2 + Sk**2)**2 + errCk**2 * (-Sk/(Ck**2 + Sk**2)**2)))
        
        return ecc, omega, errecc, erromega
    
    return ecc, omega



def mass_calc(model_param, Mstar, earth_mass = False):
    '''
    Parameters
    ----------
    model_param : 2d array
        Array of all the model parameter in the mcmc (with Sk and Ck instead of ecc and omega)
    Mstar : float
        Stellar mass in solar masses
    earth_mass : bool, optional
        True returns the planet mass in Earth masses, False returns the planet mass in Jupiter masses

    Returns
    -------
    Mpl_sini : float
        Minimum mass of the planet in Jupiter masses
    Mpl_sini_e : float
        Minimum mass of the planet in Earth masses
    '''
    
    P, K, Ck, Sk, t0 = model_param[-1][0], model_param[-1][1], model_param[-1][2], model_param[-1][3], model_param[-1][4]
    ecc = Ck**2 + Sk**2
    omega = np.arctan(Sk/Ck)
    Mpl_sini = 4.9191*10**(-3) * K * np.sqrt(1-ecc**2) * P**(1/3) * Mstar**(2/3)
    
    if earth_mass == False:
        return Mpl_sini
    if earth_mass == True:
        Mpl_sini_e = Mpl_sini * 317.9
        return Mpl_sini_e



def transpose(lst):
    '''
    Parameters
    ----------
    lst : list
        List you want to transpose

    Returns
    -------
    trans2 : list
        Transposed list

    '''
    arr2 = np.array(lst)
    trans = arr2.T
    trans2 = trans.tolist()
    
    return trans2




def model_param_names(model_list, SkCk=False, plotting = True):
    """
    Function to get model names

    Parameters
    ----------
    model_name : string or list
        Name of the model, or list of names of the models
    SkCk : boolean, optional
        If True, return the names of the Sk and Ck parameters. Default is False
    plotting : bool, optional
        If True, return names in format for plots, if False, return names in standard format
        
    Returns
    -------
    param_names : list of strings
        Name of parameters
    """
    
    # Check if it's a single model
    if isinstance(model_list, str):
        model_list=[model_list]
    if (isinstance(model_list, list) and len(model_list) == 1):
        numb = 1
    elif isinstance(model_list, list) and len(model_list) > 1:
        numb = len(model_list)
    else:
        raise ValueError("Model must be a string or a list of strings")
    
        
    # If it's a single model
    if numb == 1:
        if model_list[0].startswith("Kep") or model_list[0].startswith("kep"):
            param_names = mod.Keplerian.params(plotting = plotting, SkCk = SkCk)
            
        if model_list[0].startswith("No_Model") or model_list[0].startswith("No") or model_list[0].startswith("no"):
            param_names = mod.No_Model.params(plotting = plotting)
        
        if model_list[0].startswith("Off") or model_list[0].startswith("off"):
            param_names = mod.Offset.params(plotting = plotting)
        
        if model_list[0].startswith("Polynomial") or model_list[0].startswith("polynomial"):
            param_names = mod.Polynomial.params(plotting = plotting)
    else:
        # Check how many times each model is called
        n_kep = 0
        n_no = 0
        n_off = 0
        n_poly = 0
        param_names = []
        for mod_name in model_list:
            param_names_mods = None
            if mod_name.startswith("Kep") or mod_name.startswith("kep"):
                param_names_mods = mod.Keplerian.params(model_num = n_kep, plotting = plotting, SkCk = SkCk)
                param_names.extend(param_names_mods)
                n_kep += 1
            if mod_name.startswith("No_Model") or mod_name.startswith("No") or mod_name.startswith("no"):
                param_names_mods = mod.No_Model.params(model_num = n_no, plotting = plotting)
                param_names.extend(param_names_mods)
                n_no += 1
            if mod_name.startswith("Off") or mod_name.startswith("off"):
                param_names_mods = mod.Offset.params(model_num = n_off, plotting = plotting)
                param_names.extend(param_names_mods)
                n_off += 1
            if mod_name.startswith("Poly") or mod_name.startswith("poly"):
                param_names_mods = mod.Polynomial.params(model_num = n_poly, plotting = plotting)
                param_names.extend(param_names_mods)
                n_poly += 1
    
    return param_names


def hparam_names(kernel_name, plotting = True):
    """
    Function to get kernel hyperparameters names

    Parameters
    ----------
    kernel_name : string
        Name of the kernel
    plotting : bool, optional
        If True, return names in format for plots, if False, return names in standard format
        
    Returns
    -------
    hparam_names : list of strings
        Name of hyperparameters
    """
    if kernel_name.startswith("Cos") or kernel_name.startswith("cos"):
        hparam_names = ker.Cosine.hparams(plotting)
    if kernel_name.startswith("expsquare") or kernel_name.startswith("ExpSquare") or kernel_name.startswith("Expsquare") or kernel_name.startswith("expSquare"):
        hparam_names = ker.ExpSquared.hparams(plotting)
    if kernel_name.startswith("ExpSin") or kernel_name.startswith("expsin") or kernel_name.startswith("expSin") or kernel_name.startswith("Expsin"):
        hparam_names = ker.ExpSinSquared.hparams(plotting)
    if kernel_name.startswith("Quas") or kernel_name.startswith("quas"):
        hparam_names = ker.QuasiPer.hparams(plotting)
    if kernel_name.startswith("Jit") or kernel_name.startswith("jit"):
        hparam_names = ker.JitterQuasiPer.hparams(plotting)
    if kernel_name.startswith("Matern5") or kernel_name.startswith("matern5"):
        hparam_names = ker.Matern5.hparams(plotting)
    if kernel_name.startswith("Matern3") or kernel_name.startswith("matern3"):
        hparam_names = ker.Matern3.hparams(plotting)
    
    return hparam_names





def phasefold(time, period, t0, zerocentre=True, returnepoch=False):
    '''
    Function to phase-fold data

    Parameters
    ----------
    time : array, float
        Time array
    period : float
        Period of the orbit
    t0 : float
        Time of periastron passage
    zerocentre : boolean
        If True, the time array is shifted to the zero-centre. Default is True
    returnepoch : boolean
        Return the epoch of the phase-folded data. Default is False

    Returns
    -------
    true_phase : array, floats
        Phase array
    epoch : array, floats, optional
        Epoch of the orbit at all points. Starting from 0. Only returned if returnepoch is True.

    '''
    phase = (time - t0)/period
    # Want phase to be between 0 and 1
    epoch = np.floor(phase)
    true_phase = phase - epoch
    
    if zerocentre:
        end = np.where(true_phase >= 0.5)[0]
        true_phase[end] -= 1.0
        
    if returnepoch:
        return true_phase, epoch
    else:
        return true_phase

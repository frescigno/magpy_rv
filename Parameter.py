#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Parametrisations central

Contains:
    Parameter class
    Kernel_Par_Creator
    Model_Par_Creator
    Prior_Info_Creator

Author: Federica Rescigno
Version: 27.06.2022
'''


##########################################
########## BASIC PARAMETER CLASS ##########
##########################################



class Parameter:
    '''Object to assign initial values to a parameter and define whether it is
    allowed to vary in the fitting
    '''
    
    def __init__(self, value=None, error=None, vary=True):
        '''
        Parameters
        ----------
        value : float, optional
            Assumed initial value of the chosen variable. The default is None.
        error : float, optional
            Error on the value. The default is None
        vary : True or False, optional
            Is the variable allowed to vary in MCMC? The default is True.
        '''
        
        self.value = value
        self.error = error
        self.vary = vary
        
        # If no error is given, compute the error as 20% of the value
        if self.error is None:
            self.error = 0.2*self.value
    
    
    # String to see what the parameters look like
    def __repr__(self):
        '''        
        Returns
        -------
        message : string
            List of specific parameter value and characteristics
        '''
        message = ("Parameter object: value = {}, error={} (vary = {}) \n").format(self.value, self.error, self.vary)
        return message
    
    def vary(self):
        return self.vary
    def value(self):
        return self.value
    def error(self):
        return self.error



####################################################
########## KERNEL HYPER-PARAMETER CREATOR ##########
####################################################


def Kernel_Par_Creator(kernel):
    '''Object to create the set of parameters necessary for the chosen kernel.
    
    Parameters:
        kernel : string
            Name of the implemented kernel.
    
    Returns:
        hparams: dictionary
            Dictionary of all necessary parameters for given kernel'''
    
    if kernel.startswith("QuasiPer") or kernel.startswith("quasiper") or kernel.startswith("Quasiper"):
        hparams = dict(K_per='K_per', K_harmonic='K_harmonic', K_timescale='K_timescale', K_amp='K_amp')
            
    if kernel.startswith("Periodic") or kernel.startswith("periodic") or kernel.startswith("ExpSin") or kernel.startswith("expsin") or kernel.startswith("Expsin"):
        hparams = dict(K_amp='K_amp', K_timescale='K_timescale', K_per='K_per')
        
    if kernel.startswith("Cos") or kernel.startswith("cos"):
        hparams = dict(K_amp='K_amp', K_per='K_per')
        
    if kernel.startswith("ExpSqu") or kernel.startswith("expsqu") or kernel.startswith("Expsqu"):
        hparams = dict(K_amp='K_amp', K_length='K_length')
    
    if kernel.startswith("Matern5") or kernel.startswith("matern5"):
        hparams = dict(K_amp='K_amp', K_timescale='K_timescale')
        
    return hparams



#############################################
########## MODEL PARAMETER CREATOR ##########
#############################################

def Model_Par_Creator(models):
    '''Object to create the set of parameters necessary for the chosen model.
    
    Parameters:
        models : string
            Name of the implemented model.
    
    Returns:
        model_params: dictionary
            Dictionary of all necessary parameters for given kernel'''
    
    
    # Check how many models are requested
    if isinstance(models, str):
        models = list(models)
    if isinstance(models, list) and len(models) == 1:
        numb = 1
    elif isinstance(models, list) and len(models) > 1:
        numb = len(models)
    else:
        raise ValueError("Model must be a string or a list of strings")
    
    
    # Store the number of specific models
    zero_mod=0
    off_mod=0
    kepl_mod=0
    lintrend_mod=0
    uncorr_mod=0
    for model in models:
        if model.startswith("No_Model") or model.startswith("Zero") or model.startswith("zero"):
            zero_mod+=1
        elif model.startswith("Offset") or model.startswith("offset"):
            off_mod+=1
        elif model.startswith("Kepler") or model.startswith("kepler"):
            kepl_mod+=1
        elif model.startswith("Lin") or model.startswith("lin"):
            lintrend_mod+=1
        elif model.startswith("Uncor") or model.startswith("uncor"):
            uncorr_mod+=1
        else:
            raise ValueError("There is no such implemented model for {}".format(model))
    
    # Initialise an empty disctionary
    model_params = {}
    
    # Fill the disctionary with the appropriate parameters based on the models
    # In the case of repeated models, add _numb after the name of each parameter
    if zero_mod == 1:
        model_params.update({'zero':'zero'})
    if zero_mod > 1:
        for mod in range(zero_mod):
            model_params.update({'zero_'+str(mod):'zero'})
    
    if off_mod == 1:
        model_params.update({'offset':'offset'})
    if off_mod > 1:
        for mod in range(off_mod):
            model_params.update({'offset_'+str(mod):'offset'})
    
    if kepl_mod == 1:
        model_params.update({'P':'period','K':'semi-amplitude', 'ecc':'eccentricity', 'omega':'angle of periastron', 't0':'t of periastron passage'})
    if kepl_mod > 1:
        for mod in range(kepl_mod):
            model_params.update({'P_'+str(mod):'period','K_'+str(mod):'semi-amplitude', 'ecc_'+str(mod):'eccentricity', 'omega_'+str(mod):'angle of periastron', 't0_'+str(mod):'t of periastron passage'})
    
    if lintrend_mod == 1:
        model_params.update({'m':'slope','c':'intercept'})
    if lintrend_mod > 1:
        for mod in range(lintrend_mod):
            model_params.update({'m_'+str(mod):'slope','c_'+str(mod):'intercept'})
    
    if uncorr_mod == 1:
        model_params.update({'rms':'rms'})
    if uncorr_mod > 1:
        for mod in range(uncorr_mod):
            model_params.update({'rms_'+str(mod):'rms'})
    
    return model_params
    




#################################################
########## PRIORS INFORMATION CREATOR ##########
################################################


def Prior_Info_Creator(prior):
    '''Object to create the set of information necessary for the chosen prior.
    
    Parameters:
        prior : string
            Name of the implemented kernel.
    
    Returns:
        prior_info: dictionary
            Dictionary of all necessary information for given prior'''
    
    if prior.startswith("Gauss") or prior.startswith("gauss"):
        prior_info = dict(mu='mu', sigma='sigma')

    if prior.startswith("Jeff") or prior.startswith("jeff"):
        prior_info = dict(minval='minval', maxval='maxval')

    if prior.startswith("Mod") or prior.startswith("mod"):
        prior_info = dict(minval='minval', maxval='maxval', kneeval='kneeval')

    if prior.startswith("Uni") or prior.startswith("uni"):
        prior_info = dict(minval='minval', maxval='maxval')
            
    return prior_info


#### Prior list to be created as follows
# prior_list = []
# prior_list.append("name of the variable", "name of the prior", prior_info)



####################################################
########## MCMC CHECK INFORMATION CREATOR ##########
####################################################

def Check_Info_Creator(check):
    
    if check.startswith("no neg") or check.startswith("no_neg") or check.startswith("No_neg") or check.startswith("No_Neg"):
        check_info = None
    if check.startswith("kepl") or check.startswith("Kepl"):
        check_info = None
    if check.startswith("star") or check.startswith("Star"):
        check_info = dict(Rstar='Rstar in Rsol', Mstar='Mstar in Msol')
    if check.startswith("planet") or check.startswith("Planet"):
        check_info = None

#### Check list to be created as follows
# check_list = []
# check_list.append("kernel/model", "name of the check", check_info)
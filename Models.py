'''
Code to define the models and model parameters for the gp regression.

Contains:
    Model List
    Model parameter creator function
    No model class
    Offset model class
    Polynomial model class
    Keplerian model class 

Author: Bryce Dixon
Version 18.07.2023
    '''

import numpy as np
import scipy as sc
import Parameters as par


# List of the models for possible planets or offsets

MODELS = {"No_Model": ['rvs'],
"Offset": ['rvs', 'offset'],
"Polinomyal":["a_0","a_1","a_2","a_3"],
"Keplerian": ['time', 'P', 'K', 'ecc', 'omega', 't0'],
}

def PrintModelList():
    print("Implemented models")
    print(MODELS)

def defModelList():
    """Function to return the list of all currently available Models"""
    return MODELS


# Model parameter creator function

def mod_create(model):
    """Function to generate a dictionary of necessary model parameters for the chosen models

    Parameters
    ----------
    model: string or list of strings
        name of the desired model or list of the model names

    Raises
    ------
    ValueError:
        Raised if the model is not a string or list of strings 
        
    Returns
    -------
    model_params: dictionary
        Dictionary of all necessary parameters
    """
    
    # Check if it's a single model
    if isinstance(model, str) or (isinstance(model, list) and len(model) == 1):
        numb = 1
    elif isinstance(model, list) and len(model) > 1:
        numb = len(model)
    else:
        raise ValueError("Model must be a string or a list of strings")   
        
    # If it's a single model
    if numb == 1:
        if model[0].startswith("Kepler") or model[0].startswith("kepler"):
            model_params = dict(P='period', K='semi-amplitude', ecc='eccentricity', omega='angle of periastron', t0='t of per pass')
                
        if model[0].startswith("No_Model") or model[0].startswith("No") or model[0].startswith("no"):
            model_params = dict(no='no')
            model_params['no'] = par.parameter(value=0., error=0., vary=False)
            
        if model[0].startswith("Offset") or model[0].startswith("offset"):
            model_params = dict(offset='offset')
            
        if model[0].startswith("Poly") or model[0].startswith("poly"):
            model_params = dict(a_0="a_0",a_1="a_1",a_2="a_2",a_3="a_3")
    else:
        # Check how many times each model is called
        n_kep = 0
        n_no = 0
        n_off = 0
        n_poly = 0
        model_params = {}
        for mod_name in model:
            if mod_name.startswith("Kepler") or mod_name.startswith("kepler"):
                model_params.update({'P_'+str(n_kep):'period','K_'+str(n_kep):'semi-amplitude', 'ecc_'+str(n_kep):'eccentricity', 'omega_'+str(n_kep):'angle of periastron', 't0_'+str(n_kep):'t of periastron passage'})
                n_kep += 1
            if mod_name.startswith("No_Model") or mod_name.startswith("No") or mod_name.startswith("no"):
                model_params.update({'no_'+str(n_no):'no'})
                n_no += 1
            if mod_name.startswith("Offset") or mod_name.startswith("offset"):
                model_params.update({'offset_'+str(n_off):'offset'})
                n_off += 1
            if mod_name.startswith("Poly") or mod_name.startswith("poly"):
                model_params.update({'a_0_'+str(n_off):'a_0','a_1_'+str(n_off):'a_1','a_2_'+str(n_off):'a_2','a_3_'+str(n_off):'a_3'})
                n_poly += 1
                
        
    return model_params


# Models


##############################
########## No Model ##########
##############################

class No_Model:
    '''The subtracted model from the RV is null
    '''
    
    def __init__(self, y, no):
        '''
        Parameters
        ----------
        y : array
            Observed RV or ys
        '''
        
        self.y = y
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y = np.zeros(len(self.y))
        return model_y


################################
########## Polynomial ##########
################################


class Polynomial:
    '''The subtracted model from the RV is null
    '''
    
    def __init__(self, time, model_params):
        '''
        Parameters
        ----------
        time : array
            Time or x axis array
        '''
        
        self.time = time
        self.model_params = model_params
        
        # Check if we have the right amount of parameters
        assert len(self.model_params) == 4, "Offset Model requires 4 parameters"
        
        # Check if all hyperparameters are numbers
        try:
            self.a_0 = self.model_params['a_0'].value
            self.a_1 = self.model_params['a_1'].value
            self.a_2 = self.model_params['a_2'].value
            self.a_3 = self.model_params['a_3'].value
        except KeyError:
            for i in range(10):
                try:
                    self.a_0 = self.model_params['a_0'+str(i)].value
                    self.a_1 = self.model_params['a_1'+str(i)].value
                    self.a_2 = self.model_params['a_2'+str(i)].value
                    self.a_3 = self.model_params['a_3'+str(i)].value
                    break
                except KeyError:
                    raise KeyError("Offset Model requires 4 parameters")
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y = self.a_0 + self.a_1*self.time + self.a_2*self.time**2 + self.a_3*self.time**3
        return model_y


############################
########## Offset ##########
############################


class Offset:
    '''The subtracted model from RV is a constant offset chosen explicitlty
    '''
    
    def __init__(self, flags, model_params):
        '''
        Parameters
        ----------
        y : array
            Observed RV or y.
        offset : float
            Constant offset
        '''
        
        self.flags = flags
        self.model_params = model_params
        
        # Check if we have the right amount of parameters
        assert len(self.model_params) == 1, "Offset Model requires 1 parameter:" \
            + "'offset'"
        
        # Check if all hyperparameters are numbers
        try:
            self.offset = self.model_params['offset'].value
            self.flag_val = 1.
            print('offset_')
        except KeyError:
            for i in range(10):
                try:
                    self.offset = self.model_params['offset_'+str(i)].value
                    self.flag_val = i+1.
                    print('offset_'+str(i))
                    break
                except KeyError:
                    if i == 10:
                        raise KeyError("Offset Model requires 1 parameter:" \
                        + "'offset'")
                    else:
                        continue
        
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y=[]
        # flags is an array of 0s and 1s
        for i in range(len(self.flags)):
            # if i is a 1, append the offset value to the empty list
            if self.flags[i] == self.flag_val:
                model_y.append(self.offset)
            else:
                #if i is 0, append 0 to the empty list
                model_y.append(0.0)
        
        model_y = np.array(model_y)
        # model y is an array containing all the offsets only for the values needing to be offset
        return model_y


###############################
########## Keplerian ##########
###############################


class Keplerian:
    '''The generalized Keplerian RV model (only use when dealing with RV observation
    of star with possible planet).
    If multiple planets are involved, the model parameters should be inputted as list (not fully implemented yet).
    
    '''
    
    def __init__(self, time, model_params):
        '''
        Parameters
        ----------
        time : array, floats
            Time array over which to compute the Keplerian.
        model_params: dictionary
            Dictionary of model parameters containing:
                P : float
                    Period of orbit in days.
                K : float
                    Semi-aplitude of the rv signal in meter per seconds.
                ecc : float
                    Eccentricity of orbit. Float between 0 and 0.99
                omega : float
                    Angle of periastron, in rad.
                t0 : float
                    Time of periastron passage of a planet
        '''
        
        self.time = time
        self.model_params = model_params
        
        # Check if we have the right amount of parameters
        assert len(self.model_params) == 5, "Keplerian Model requires 5 parameters:" \
            + "'P', 'K', 'ecc', 'omega', 't0'"
        
        # Check if all hyperparameters are number
        try:
            P = self.model_params['P'].value
            K = self.model_params['K'].value
            ecc = self.model_params['ecc'].value
            omega = self.model_params['omega'].value
            t0 = self.model_params['t0'].value
        except KeyError:
            for i in range(10):    
                try:
                    self.model_params['P_'+str(i)].value
                    break
                except KeyError:
                    if i == 10:
                        raise KeyError("Keplerian Model requires 5 parameters:" \
                            + "'P', 'K', 'ecc', 'omega', 't0'")
                    else:
                        continue

            P = self.model_params['P_'+str(i)].value
            K = self.model_params['K_'+str(i)].value
            ecc = self.model_params['ecc_'+str(i)].value
            omega = self.model_params['omega_'+str(i)].value
            t0 = self.model_params['t0_'+str(i)].value
        
        self.P = P
        self.K = K
        self.ecc = ecc
        self.omega = omega
        self.t0 = t0
    
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the model
        '''
        message = "Keplerian RV model with the following set of parameters: \n","P = {} \n".format(self.P), "K = {} \n".format(self.K), "ecc = {} \n".format(self.ecc), "omega = {} \n".format(self.omega), "t0 = {}".format(self.to)
        print(message)
        return message
    
    
    def ecc_anomaly(self, M, ecc, max_itr=200):
        '''
        ----------
        M : float
            Mean anomaly
        ecc : float
            Eccentricity, number between 0. and 0.99
        max_itr : integer, optional
            Number of maximum iteration in E computation. The default is 200.

        Returns
        -------
        E : float
            Eccentric anomaly
        '''
        
        E0 = M
        E = M
        for i in range(max_itr):
            f = E0 - ecc*np.sin(E0) - M
            fp = 1. - ecc*np.cos(E0)
            E = E0 - f/fp
            
            # check for convergence
            if (np.linalg.norm(E - E0, ord=1) <= 1.0e-10):
                return E
                break
            # if not convergence continue
            E0 = E
        
        # no convergence, return best estimate
        return E
    
    
    def true_anomaly(self, E, ecc):
        '''
        Parameters
        ----------
        E : float
            Eccentric anomaly
        ecc : float
            Eccentricity

        Returns
        -------
        nu : float
            True anomaly

        '''
        nu = 2. * np.arctan(np.sqrt((1.+ecc)/(1.-ecc)) * np.tan(E/2.))
        return nu
        
        
    
    
    def model(self):
        '''
        Returns
        -------
        rad_vel : array, floats
            Radial volicity model derived by the number of planet include
        '''
        rad_vel = np.zeros_like(self.time)
        
        # Need to add for multiple planets
        
        # Compute mean anomaly M
        M = 2*np.pi * (self.time-self.t0) / self.P
        
        E = self.ecc_anomaly(M, self.ecc)

        nu = self.true_anomaly(E, self.ecc)
    
            
        rad_vel = rad_vel + self.K * (np.cos(self.omega + nu) + self.ecc*np.cos(self.omega))
        
        model_keplerian = rad_vel
        
        return model_keplerian
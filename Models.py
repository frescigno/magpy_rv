'''
Code to define the models and model parameters for the gp regression.

Contains:
    Model List
    Model parameter creator function
    Get data function
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
                model_params.update({'a_0_'+str(n_poly):'a_0','a_1_'+str(n_poly):'a_1','a_2_'+str(n_poly):'a_2','a_3_'+str(n_poly):'a_3'})
                n_poly += 1
                
        
    return model_params


# flags creator function

def get_data(times, ys, y_errs):
    """Function to combine data from multiple telescopes to an X and Y array and generate an array of flags for the offsets

    Parameters
    ----------
    times: list or tuple of arrays of floats
        list of the time arrays from the telescopes where the first array in the list is the zero point offset
    ys: list or tuple of arrays of floats
        list of the rv arrays from the telescopes where the first array in the list is the zero point offset, arrays entered into list in same order as times
    y_errs: list or tuple of arrays of floats
        list of the rv error arrays from the telescopes where the first array in the list is the zero point offset, arrays entered into list in same order as times
    
    Raises
    ------
    Assertion:
        Raised if times is not a list or tuple
    Assertion
        Raised if ys is not a list or tuple
    Assertion
        Raised if y_errs is not a list or tuple
        
    Returns
    -------
    time: array of floats
        combined time array of all telescopes
    y: array of floats
        combined rv array of all telescopes
    y_err: array of floats
        combined rv error array of all telescopes
    flags: array of floats
        array of flags representing which datapoints in the time array are related to which telescope and so will have which offset
    """
    
    if type(times) == np.ndarray:
            times = times.tolist()
    if type(ys) == np.ndarray:
            ys = ys.tolist()
    if type(y_errs) == np.ndarray:
            y_errs = y_errs.tolist()
    
    assert type(times) == list or tuple, "times should be a list or tuple of arrays"
    assert type(ys) == list or tuple, "ys should be a list or tuple of arrays"
    assert type(y_errs) == list or tuple, "y_errs should be a list or tuple of arrays"
    
    flag_list = []
    for N, i in enumerate(times): 
      flagval = np.zeros_like(i) + N
      flag_list.append(flagval)
    
    time = np.concatenate(times)
    y = np.concatenate(ys)
    y_err = np.concatenate(y_errs)
    flags = np.concatenate(flag_list)
    time, y, y_err, flags = zip(*sorted(zip(time, y, y_err, flags)))
    time = np.array(time)
    return(time, y, y_err, flags)
       
    


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
    '''The model of the rv data follows a polynomial up to the 3rd degree with equation
        a_3*time^3 + a_2*time^2 + a_1*time + a_0
    '''
    
    def __init__(self, time, model_params):
        '''
        Parameters
        ----------
        time : array
            Time or x axis array
        model_params : dictionary of model parameters
            containing a_0, a_1, a_2, a_3
        
        Raises
        ------
        Assertion:
            Raised if the model does not have 4 model parameters
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
                    self.a_0 = self.model_params['a_0_'+str(i)].value
                    self.a_1 = self.model_params['a_1_'+str(i)].value
                    self.a_2 = self.model_params['a_2_'+str(i)].value
                    self.a_3 = self.model_params['a_3_'+str(i)].value
                    break
                except KeyError:
                    if i == 10:
                        raise KeyError("Offset Model requires 4 parameters")
                    else:
                        continue
            
            
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
        flags : array of floats
            array of flags representing which datapoints in the time array are related to which telescope and so will have which offset generated by the get_data function
        model_params : dictionary of model parameters
            dictionary containing the offset model parameter 
            
        Raises
        ------
        Assertion:
            Raised if model parameters is longer than 1
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
        except KeyError:
            for i in range(10):
                try:
                    self.offset = self.model_params['offset_'+str(i)].value
                    self.flag_val = i+1.
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All implemented models are included here

Contains:
    Zero model
    Offset
    Keplerian
    Linear Trend
    Uncorrelated Noise

Author: Federica Rescigno
Version: 27.06.2022
'''

import numpy as np
import abc
ABC= abc.ABC



def Model_list():
    MODELS = {"Zero_Model": ['x'],
        "Offset": ['x', 'offset'],
        "Keplerian": ['x', 'P', 'K', 'ecc', 'omega', 'tperi'],
        "LinTrend": ['x', 'm', 'c'],
        "UncorrNoise": ['x', 'rms']
        }
    return MODELS

# To come:
# larger scale trends (linear or nonlinear)
# celerite
# FF' method


def PrintModelList():
    print("Implemented models")
    MODELS = Model_list()
    print(MODELS)




###################################
########## PARENT MODEL ##########
###################################

class Model(ABC):
    '''Parent class for all models. All new models should inherit from this class and
    follow its structure.
    
    Each new model will require a __init__ method to override the parent class. In the __init__ function
    call the necessary paramaters (generated with a dictionary).
    '''
    
    @abc.abstractproperty
    def name(self):
        pass
    
    @abc.abstractproperty
    def __repr__(self):
        '''Prints message with name of the model and assigned parameters'''       
        pass
    
    @abc.abstractproperty
    def par_number(self):
        '''Number of parameters required by the model'''
        pass



###################################
########## ZERO MODEL ##########
###################################

class Zero_Model(Model):
    '''Class for no model, it produces a zero array of the same size of the observations.
    Allows for a GP regression with no model on top of it.
    Needed for code structure '''
    
    #### took away the 'no' parameter, need fixing in the use of the model in GP
    
    def __init__(self, x):
        '''
        Parameters
        ----------
        x : array
            Observed RV or ys
        '''
        
        self.x = x

    @property
    def name(self):
        return "Zero model"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        message = "The model is null"
        print(message)
        
    @property
    def par_number(self):
        '''
        Returns
        -------
        N: int
            Number of model parameters needed
        '''
        N = 0
        return N
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y = np.zeros(len(self.x))
        return model_y



###################################
########## OFFSET MODEL ##########
###################################

class Offset(Model):
    '''Class that computes an offset model. The subtracted model is a constant offset chosen
    explicitly.
    '''
    
    ### In future add possibility to choose separate offsets for separate sets of data
    
    def __init__(self, x, model_par):
        '''
        Parameters
        ----------
        x : array
            X axis of the datapoints (eg. time)
        model_par : dictionary
            Dictionary od the model parameter containing:
                offset: float
                    Constant offset
        '''
        
        self.x = x
        self.model_par = model_par
        
        # Check if we have the right amount of parameters
        assert len(self.model_par) == 1, "Offset Model requires 1 parameter:" \
            + "'offset'"
        
        # Check if all hyperparameters are numbers
        try:
            # Try to see if this is the only offset model present
            self.offset = self.model_par['offset'].value
        except KeyError:
            # If not try to see if the name 'offset' includes a number, meaning there are more than one offset models
            for i in range(20):            # 20 is chosen since we don't expect a number of chosen models at the same time largern than 20 (can be changed)
                try:
                    self.offset = self.model_par['offset_'+str(i)].value
                    break
                except KeyError:
                    if i == 20:
                        # if both fail and we reached the end, it means that no such parameter is given
                        raise KeyError("Offset Model requires 1 parameter:" \
                        + "'offset'")
                    else:
                        continue
                        
    @property
    def name(self):
        return "Offset model"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        message = "Offset model with the parameter: \n","offset = {} \n".format(self.offset)
        print(message)
    
    @property
    def par_number(self):
        '''
        Returns
        -------
        N: int
            Number of model parameters needed
        '''
        N = 1
        return N
    
    def model(self):
        '''
        Returns
        -------
        model_y : array
            Model y to subtract from the observations
        '''
        model_y = np.ones_like(self.x)*self.offset
        return model_y




#####################################
########## KEPLERIAN MODEL ##########
#####################################


class Keplerian(Model):
    '''The generalized Keplerian RV model.
    
    Computed as:
        RV = K * (cos[omega+nu] + ecc*cos[omega])
    
    in which:
        K = amplitude
        omega = angle of periastron
        nu = true anomaly
        ecc = eccentricity
    
    nu is computed using the time of periastron, period and eccentricity going through mean anomaly
    and eccentric anomaly
    '''
    
    def __init__(self, x, model_par):
        '''
        Parameters
        ----------
        x : array, floats
            Time array over which to compute the Keplerian.
        model_par: dictionary
            Dictionary of model parameters containing:
                P : float
                    Period of orbit in days.
                K : float
                    Semi-aplitude of the rv signal in meter per seconds.
                ecc : float
                    Eccentricity of orbit. Float between 0 and 0.99
                omega : float
                    Angle of periastron, in rad.
                tperi : float
                    Time of periastron passage of a planet
        '''
        
        self.x = x
        self.model_par = model_par
        
        # Check if we have the right amount of parameters
        assert len(self.model_par) == 5, "Keplerian Model requires 5 parameters:" \
            + "'P', 'K', 'ecc', 'omega', 'tperi'"
        
        
        # Check if all hyperparameters are number
        try:
            # Try to see if this is the only offset model present
            self.P = self.model_par['P'].value
            self.K = self.model_par['K'].value
            self.ecc = self.model_par['ecc'].value
            self.omega = self.model_par['omega'].value
            self.tperi = self.model_par['tperi'].value
        except KeyError:
            # If not try to see if the name 'offset' includes a number, meaning there are more than one offset models
            for i in range(20):                 # 20 is chosen since we don't expect a number of chosen models at the same time largern than 20 (can be changed)
                try:
                    self.P = self.model_par['P_'+str(i)].value
                    self.K = self.model_par['K_'+str(i)].value
                    self.ecc = self.model_par['ecc_'+str(i)].value
                    self.omega = self.model_par['omega_'+str(i)].value
                    self.tperi = self.model_par['tperi_'+str(i)].value
                    break
                except KeyError:
                    if i == 20:
                        raise KeyError("Keplerian Model requires 5 parameters:" \
                            + "'P', 'K', 'ecc', 'omega', 'tperi'")
                    else:
                        continue

    
    @property
    def name(self):
        return "Keplerian Model"

    @property
    def __repr__(self):
        message = "Keplerian RV model with the following set of parameters: \n","P = {} \n".format(self.P), "K = {} \n".format(self.K), "ecc = {} \n".format(self.ecc), "omega = {} \n".format(self.omega), "tperi = {}".format(self.to)
        print(message)
    
    @property
    def par_number(self):
        '''
        Returns
        -------
        N: int
            Number of model parameters needed
        '''
        N = 5
        return N
    
    
    def mean_anomaly(self):
        ''' Computing the mean anomaly as:
                M = 2pi * (time - t_peri)/P
            
        Returns
        -------
        M : array, float
            Mean anomaly array for all time datapoints
        '''
        M = 2*np.pi * (self.x-self.tperi) / self.P
        return M
    
    def ecc_anomaly(self, M, max_itr=200):
        ''' Computing the eccentric anomaly via iterative model
        
        Parameters
        ----------
        M : array, floats
            Mean anomaly array for all time datapoints
        max_itr : int
            Maximum number of iterations to compute E, default is 200
            
        Returns
        ----------
        E : array, floats
            Eccentric anomaly array for all time datapoints
        '''
        E0 = M
        E = M
        for i in range(max_itr):
            f = E0 - self.ecc*np.sin(E0) - M
            fp = 1. - self.ecc*np.cos(E0)
            E = E0 - f/fp
            
            # check for convergence (over a randomly small number)
            if (np.linalg.norm(E - E0, ord=1) <= 1.0e-10):
                return E
                break
            # if not convergence continue
            E0 = E
        # in case of no convergence, return best estimate
        return E
    
    def true_anomaly(self, E):
        ''' Computing the true anomaly as:
            nu = 2arctan{ sqrt[(1+ecc)/(1-ecc)] * tan(E/2)}
        
        Parameters
        ----------
        E : array, floats
            Eccentric anomaly array for all time datapoints

        Returns
        -------
        nu : array, floats
            True anomaly array for all time datapoints
        '''
        nu = 2.*np.arctan(np.sqrt((1.+self.ecc)/(1.-self.ecc)) * np.tan(E/2.))
        return nu
    
    def model(self):
        '''
        Returns
        -------
        model_y : array, floats
            Radial velocities derived by Keplerian model
        '''
        
        # Compute true anomaly
        M = self.mean_anomaly()
        E = self.ecc_anomaly(M)
        nu = self.true_anomaly(E)
        
        model_y = self.K * (np.cos(self.omega + nu) + self.ecc*np.cos(self.omega))
        return model_y




#####################################
########## LINEAR TREND MODEL ##########
#####################################
        
class LinTrend_Model(Model):
    ''' Class that computed a general extra linear trend in the data.
    Computed as a straight line:
        m*x + c
    in which:
        m = slope
        c = intercept
    '''
    
    def __init__(self, x, model_par):
        '''
        Parameters
        ----------
        x : array, floats
            Time array over which to compute the Keplerian.
        model_par: dictionary
            Dictionary of model parameters containing:
                m : float
                    Slope of the trend
                c : float
                    y-intercept of the trend
        '''
        
        self.x = x
        self.model_par = model_par
        
        # Check if we have the right amount of parameters
        assert len(self.model_par) == 2, "Linear Trend Model requires 2 parameters:" \
            + "'m', 'c'"
        
        # Check if all hyperparameters are numbers
        try:
            # Try to see if this is the only offset model present
            self.m = self.model_par['m'].value
            self.c = self.model_par['c'].value
        except KeyError:
            # If not try to see if the name 'offset' includes a number, meaning there are more than one offset models
            for i in range(20):            # 20 is chosen since we don't expect a number of chosen models at the same time largern than 20 (can be changed)
                try:
                    self.m = self.model_par['m_'+str(i)].value
                    self.c = self.model_par['c_'+str(i)].value
                    break
                except KeyError:
                    if i == 20:
                        # if both fail and we reached the end, it means that no such parameter is given
                        raise KeyError("Linear Trend Model requires 2 parameter:" \
                        + "'m' and 'c'")
                    else:
                        continue
    
    @property
    def name(self):
        return "Linear Trend Model"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the model
        '''
        message = "Linear Trend model with the parameter: \n","m = {} \n".format(self.m), "c = {} \n".format(self.c)
        print(message)
    
    @property
    def par_number(self):
        '''
        Returns
        -------
        N: int
            Number of model parameters needed
        '''
        N = 2
        return N
    
    def model(self):
        '''
        Returns
        -------
        model_y : array, floats
            Array of y-values for the linear trend
        '''
        model_y = self.m*self.x + self.c*np.ones_like(self.x)
        return model_y





###############################################
########## UNCORRELATED NOISE MODEL ##########
###############################################
        
class UncorrNoise_Model(Model):
    ''' Class that computes an extra jitter, uncorrelated noise model.
    '''
    
    def __init__(self, x, model_par):
        '''
        Parameters
        ----------
        x : array, floats
            Time array over which to compute the Keplerian.
        model_par: dictionary
            Dictionary of model parameters containing:
                rms : float
                    rms of the jitter
        '''
        
        self.x = x
        self.model_par = model_par
        
        # Check if we have the right amount of parameters
        assert len(self.model_par) == 1, "Uncorrelated Noise Model requires 1 parameters:" \
            + "'rms'"
        
        # Check if all hyperparameters are numbers
        try:
            # Try to see if this is the only offset model present
            self.rms = self.model_par['rms'].value
        except KeyError:
            # If not try to see if the name 'offset' includes a number, meaning there are more than one offset models
            for i in range(20):            # 20 is chosen since we don't expect a number of chosen models at the same time largern than 20 (can be changed)
                try:
                    self.rms = self.model_par['rms_'+str(i)].value
                    break
                except KeyError:
                    if i == 20:
                        # if both fail and we reached the end, it means that no such parameter is given
                        raise KeyError("Uncorrelated Noise Model requires 1 parameters:" \
                            + "'rms'")
                    else:
                        continue
    
    @property
    def name(self):
        return "Uncorrelated Noise Model"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the model
        '''
        message = "Uncorrelated Noise model with the parameter: \n","rms = {} \n".format(self.rms)
        print(message)

    @property
    def par_number(self):
        '''
        Returns
        -------
        N: int
            Number of model parameters needed
        '''
        N = 1
        return N
    
    def model(self):
        '''
        Returns
        -------
        model_y : array, floats
            Array of random noise
        '''
        model_y = np.array(np.random.normal(0,self.rms,len(self.x)))
        return model_y



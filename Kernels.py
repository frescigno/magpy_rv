#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Kernels for the Gaussian Process model.

Contains:
    Parent Kernel Class
    Cosine Kernel
    Exponential Squared Kernel
    Exponential Sine Squared Kernel
    Quasi Periodic Kernel
    Matérn 5th Order Kernel

Author: Federica Rescigno
Version: 27.05.2022
'''

import numpy as np
import scipy
import abc
ABC= abc.ABC


###################################
# Add opportunity to sum/multiply different kernels (same way as models)


# List of implemented kernels with hyperparameters
def Kernel_list():
    KERNELS = {
        "Cosine": ['K_amp', 'K_per'],
        "ExpSquared": ['K_amp', 'K_length'],
        "ExpSinSquared": ['K_amp', 'K_timescale', 'K_per'],
        "QuasiPer": ['K_per', 'K_harmonic', 'K_timescale', 'K_amp'],
        "Matern 5/2": ['K_amp', 'K_timescale']
        }
    return KERNELS

# to come:
# Cosine QuasiPer
# Matern 3/2
# Matern 5/2


def PrintKernelList():
    print("Implemented kernels:")
    kernels = Kernel_list()
    print(kernels)




###################################
########## PARENT KERNEL ##########
###################################


class Kernel(ABC):
    '''Parent class for all kerners. All new kernels should inherit from this class and
    follow its structure.
    
    Each new kerner will require a __init__ method to override the parent class. In the __init__ function
    call the necessary hyperparamaters (generated with a dictionary).
    '''
    
    @abc.abstractproperty
    def name(self):
        pass
    
    @abc.abstractproperty
    def __repr__(self):
        '''Prints message with name of the Kernel and assigned hyperparameters'''       
        pass
    
    @abc.abstractmethod
    def compute_distances(self, t1, t2):
        '''Computes the distances between two sets of points
        
        Parameters
        ----------
        t1 : array or list, floats
            First set of time points
        t2 : array or list, floats
            Second set of time points

        Returns
        -------
        self.dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space'''
        
        T1 = np.array([t1]).T
        T2 = np.array([t2]).T
        
        self.dist_e = scipy.spatial.distance.cdist(T1, T2, 'euclidean')
        
        return self.dist_e
    
    @abc.abstractmethod
    def compute_covmatrix(self, errors):
        '''Computes the covariance matrix of the kernel'''
        pass


###################################
########## COSINE KERNEL ##########
###################################


class Cosine(Kernel):
    '''Class that computes the Cosine kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 cos[(2pi . |t-t'|) / H_2]}
    
    in which:
        H_1 = variance/amp
        H_2 = per
    '''

    
    def __init__(self, hparams):
        '''
           Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "Periodic Cosine kernel requires 2 hyperparameters:" \
            + "'K_amp', 'K_per'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['K_amp'].value
            self.hparams['K_per'].value
        except KeyError:
            raise KeyError("Cosine kernel requires 2 hyperparameters:" \
            + "'amp', 'per'")
    
    @property    
    def name(self):
        return "Cosine"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['K_per'].value
        amp = self.hparams['K_amp'].value
        
        message = "Cosine Kernel with amp: {}, per: {}".format(amp, per)
        print(message)
    
    
    def compute_distances(self, t1, t2):
        ''' See parent function '''
        
        self.dist_e = super.compute_distances(t1, t2)

        return self.dist_e
    
    
    def compute_covmatrix(self, errors=None):
        '''
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        '''
        
        per = self.hparams['K_per'].value
        amp = self.hparams['K_amp'].value
        
        K = np.array(amp**2 * np.cos(2*np.pi * self.dist_e / per))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix
    
   
   
###############################################
########## EXPONENTIAL SQUARE KERNEL ##########
###############################################


class ExpSquared(Kernel):
    '''Class that computes the Periodic kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 . exp{-1/2 (|t-t'| / H_2)^2}
    
    in which:
        H_1 = variance/amp
        H_2 = recurrence timescale/length
    '''
    
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with errors
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'K_amp', 'K_timescale'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['K_amp'].value
            self.hparams['K_timescale'].value
        except KeyError:
            raise KeyError("Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'amp', 'timescale'")
    
    @property
    def name(self):
        return "ExpSquared"
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        amp = self.hparams['K_amp'].value
        timescale = self.hparams['K_timescale'].value
        
        message = "Periodic ExpSinSquared Kernel with amp: {}, timescale: {}".format(amp, timescale)
        print(message)
    
    
    def compute_distances(self, t1, t2):
        ''' See parent function '''
        
        self.dist_e = super.compute_distances(t1, t2)

        return self.dist_e
    
    
    def compute_covmatrix(self, errors):
        '''
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        '''
        
        timescale = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        K = np.array(amp**2 * np.exp(-0.5*self.dist_e**2 / timescale**2))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix

 
###############################################
########## EXPONENTIAL SINE SQUARE KERNEL ##########
###############################################


class ExpSinSquared(Kernel):
    '''Class that computes the Periodic kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 . exp{-2/H_3^2 . sin^2[(pi . |t-t'|) / H_2]}
    
    in which:
        H_1 = variance/amp
        H_3 = recurrence timescale/length
        H_2 = period
    '''
    
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 3 elements with errors
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        self.name = 'ExpSinSquared'
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 3, "Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'K_amp', 'K_timescale', 'K_per'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['K_amp'].value
            self.hparams['K_timescale'].value
            self.hparams['K_per'].value
        except KeyError:
            raise KeyError("Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'amp', 'timescale', 'per'")
    
    @property
    def name(self):
        return "ExpSinSquared"
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['K_per'].value
        amp = self.hparams['K_amp'].value
        timescale = self.hparams['K_timescale'].value
        
        message = "Periodic ExpSinSquared Kernel with amp: {}, timescale: {}, per: {}".format(amp, timescale, per)
        print(message)
    
    
    def compute_distances(self, t1, t2):
        ''' See parent function '''
        
        self.dist_e = super.compute_distances(t1, t2)

        return self.dist_e
    
    
    def compute_covmatrix(self, errors):
        '''
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        '''
        
        per = self.hparams['K_per'].value
        timescale = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        K = np.array(amp**2 * np.exp((-2/timescale**2) * (np.sin(np.pi * self.dist_e / per))**2))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix


##########################################
########## QUASIPERIODIC KERNEL ##########
##########################################


class QuasiPer:
    '''Class that computes the quasi-periodic kernel matrix.
    
    Kernel formula from Haywood Thesis 2016, Equation 2.14:
    
        K = H_1^2 . exp{ [-(t-t')^2 / H_2^2] - [ sin^2(pi(t-t')/H_3) / H_4^2] }
    
    in which:
        H_1 = amp
        H_2 = explength, timescale
        H_3 = per
        H_4 = perlength, harmonic
    '''

    
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 4 elements

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 4 required parameters
        '''
        
        # Initialize final result
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 4, "QuasiPeriodic Kernel requires 4 hyperparameters:" \
            + "'K_per', 'K_harmonic', 'K_timescale', 'K_amp'"
        
        # Check if all hyperparameters are numbers
        try:
            self.hparams['K_per'].value
            self.hparams['K_harmonic'].value
            self.hparams['K_timescale'].value
            self.hparams['K_amp'].value
        except KeyError:
            raise KeyError("QuasiPeriodic Kernel requires 4 hyperparameters:" \
            + "'K_per', 'K_harmonic', 'K_timescale', 'K_amp'")
        
    @property
    def name(self):
        return "QuasiPer"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        per = self.hparams['K_per'].value
        harmonic = self.hparams['K_harmonic'].value
        timescale = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        message = "QuasiPeriodic Kernel with amplitude: {}, harmonic complexity: {}, period: {}, timescale: {}".format(amp, harmonic, per, timescale)
        print(message)
    
    
    def compute_distances(self, t1, t2):
        '''
        Parameters
        ----------
        t1 : array or list, floats
            Array of the first time series
        t2 : array or list, floats
            Array of the second time series

        Returns
        -------
        self.dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t')
        self.dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2

        '''
        T1 = np.array([t1]).T
        T2 = np.array([t2]).T
        
        self.dist_e = scipy.spatial.distance.cdist(T1, T2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(T1, T2, 'sqeuclidean')

        return self.dist_e, self.dist_se
    
    
    def compute_covmatrix(self, errors):
        '''
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the quasiperiodic kernel

        '''
        
        per = self.hparams['K_per'].value
        harmonic = self.hparams['K_harmonic'].value
        timescale = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        #print("Paramters for this round:", self.hparams)
        
        K = np.array(amp**2 * np.exp(-self.dist_se/(timescale**2)) 
                     * np.exp(-((np.sin(np.pi*self.dist_e/per))**2)/(harmonic**2)))
        
        # This is the covariance matrix
        self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are present or the array is non-square
            pass
        
        return self.covmatrix


###################################
########## MATERN 5/2 KERNEL ##########
###################################


class Matern5(Kernel):
    '''Class that computes the Matern 5/2 matrix.
    
    Kernel formula:
        
        K = H_1^2 (1 + (sqr(5)|t-t'|)/H_2 + 5|t-t'|^2/3*H_2^2) exp{-sqr(5)|t-t'|/H_2}
    
    in which:
        H_1 = variance/amp
        H_2 = timescale
    '''

    
    def __init__(self, hparams):
        '''
           Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "Matern 5/2 kernel requires 2 hyperparameters:" \
            + "'K_amp', 'K_timescale'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['K_amp'].value
            self.hparams['K_timescale'].value
        except KeyError:
            raise KeyError("Matern 5/2 kernel requires 2 hyperparameters:" \
            + "'amp', 'timescale'")
    
    @property    
    def name(self):
        return "Matern 5/2"
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        message = "Matern 5/2 Kernel with amp: {}, timesclae: {}".format(amp, per)
        print(message)
    
    
    def compute_distances(self, t1, t2):
        '''
        Parameters
        ----------
        t1 : array or list, floats
            Array of the first time series
        t2 : array or list, floats
            Array of the second time series

        Returns
        -------
        self.dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t')
        self.dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2

        '''
        T1 = np.array([t1]).T
        T2 = np.array([t2]).T
        
        self.dist_e = scipy.spatial.distance.cdist(T1, T2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(T1, T2, 'sqeuclidean')

        return self.dist_e, self.dist_se
    
    
    def compute_covmatrix(self, errors=None):
        '''
        Parameters
        ----------
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        '''
        
        timescale = self.hparams['K_timescale'].value
        amp = self.hparams['K_amp'].value
        
        K = np.array(amp**2 * (1 + (np.sqrt(5)*self.dist_e/timescale) + (5*self.dist_se**2/3*timescale**2) * np.exp(-np.sqrt(5)*self.dist_e/timescale)))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix




    


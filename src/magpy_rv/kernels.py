
'''
Kernels for the Gaussian Process model

Contains:
    Kernel List
    Compute Distances Function
    
    Parent Kernel Class
    Cosine Kernel Class
    Exponential Squared Kernel Class
    Exponential Sine Squared Kernel Class
    Quasi Periodic Kernel Class
    Jitter Quasi Periodic Kernel Class
    Matérn 5th Order Kernel Class
    Matérn 3rd Order Kernel Class

Author: Federica Rescigno, Bryce Dixon
Version: 22.08.2023'''

import numpy as np
import scipy as sc
import abc
ABC = abc.ABC

# List of Implemented Kernels with hyperparameters

KERNELS = {
    "Cosine": ['gp_amp', 'gp_per'],
    "ExpSquared": ['gp_amp', 'gp_timescale'],
    "ExpSinSquared": ['gp_amp', 'gp_timescale', 'gp_per'],
    "QuasiPer": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'],
    "JitterQuasiPer": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp', 'gp_jit'],
    "Matern5/2": ['gp_amp', 'gp_timescale'],
    "Matern3/2": ['gp_amp', 'gp_timescale', 'gp_jit'],
    }

def PrintKernelList():
    """Function to print the list of all currently available Kernels.
    """
    print("Implemented kernels:")
    print(KERNELS)

def defKernelList():
    """Function to return the list of all currently available Kernels"""
    return KERNELS

# Compute Distances Function

def compute_distances(t1, t2):
    '''
    Function to compute the spatial distance between each x1 and x2 point in both euclidean and squared euclidean space.
    
    Parameters
    ----------
    t1 : array or list, floats
        Array of the first time series
    t2 : array or list, floats
        Array of the second time series

    Returns
    -------
    dist_e : array, floats
        Spatial distance between each x1-x2 points set in euclidean space
        in formula = (t - t')
    dist_se : array, floats
        Spatial distance between each x1-x2 points set in squared euclidean space
        in formula = (t - t')^2

    '''
    T1 = np.array([t1]).T
    T2 = np.array([t2]).T
    
    dist_e = sc.spatial.distance.cdist(T1, T2, 'euclidean')
    dist_se = sc.spatial.distance.cdist(T1, T2, 'sqeuclidean')
    
    return dist_e, dist_se


###################################
########## PARENT KERNEL ##########
###################################


class Kernel(ABC):
    '''Parent class for all kernels. All new kernels should inherit from this class and
    follow its structure.
    
    Each new kernel will require a __init__ method to override the parent class. In the __init__ function
    call the necessary hyperparamaters (generated with a dictionary).
    '''
    
    @abc.abstractstaticmethod
    def name(self):
        pass
    
    @abc.abstractstaticmethod
    def hparams():
        '''returns the list of hyperparameter names'''
        pass
    
    @abc.abstractproperty
    def __repr__(self):
        '''Prints message with name of the Kernel and assigned hyperparameters'''       
        pass
    
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
        
    Arguments:
        hparams: dictionary with all the hyperparameters
        Should have 2 elements with possibly errors
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Cosine Kernel
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with possibly errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 2 required parameters
        '''
    
        # Initialise
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "Periodic Cosine kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_per'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except KeyError:
            raise KeyError("Cosine kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_per'")
        
    @staticmethod  
    def name():
        return "Cosine"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_amp', 'gp_per']
        if plotting is True:
            return [r'gp$_{amp}$', r'gp$_{per}$']
            
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['gp_per'].value
        amp = self.hparams['gp_amp'].value
        
        message = "Cosine Kernel with amp: {}, per: {}".format(amp, per)
        print(message)
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Cosine Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the cosine kernel
        '''
        
        per = self.hparams['gp_per'].value
        amp = self.hparams['gp_amp'].value
        
        K = np.array(amp**2 * np.cos(2*np.pi * dist_e / per))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except:     #if errors are not present or the array is non-square
            pass
        return self.covmatrix
    
    
###############################################
########## EXPONENTIAL SQUARE KERNEL ##########
###############################################


class ExpSquared(Kernel):
    '''Class that computes the Exponential Squared kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 . exp{-1/2 (|t-t'| / H_2)^2}
    
    in which:
        H_1 = variance/amp
        H_2 = recurrence timescale/length
        
    Arguments:
        hparams: dictionary with all the hyperparameters
        Should have 2 elements with possibly errors
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Exponential Squared Kernel
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with possibly errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 2 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "ExpSquared kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_timescale'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['gp_amp'].value
            self.hparams['gp_timescale'].value
        except KeyError:
            raise KeyError("ExpSquared kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_timescale'")
        
    @staticmethod   
    def name():
        return "ExpSquared"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_amp', 'gp_timescale']
        if plotting is True:
            return [r'gp$_{amp}$', r'gp$_{timescale}$']
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        amp = self.hparams['gp_amp'].value
        timescale = self.hparams['gp_timescale'].value
        
        message = "ExpSquared Kernel with amp: {}, timescale: {}".format(amp, timescale)
        print(message)
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Exponential Squared Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the ExpSquared kernel
        '''
        
        timescale = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        
        K = np.array(amp**2 * np.exp(-0.5*dist_e**2 / timescale**2))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix
    
    
####################################################
########## EXPONENTIAL SINE SQUARE KERNEL ##########
####################################################


class ExpSinSquared(Kernel):
    '''Class that computes the Exponential Sine Squared kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 . exp{-2/H_3^2 . sin^2[(pi . |t-t'|) / H_2]}
    
    in which:
        H_1 = variance/amp
        H_3 = recurrence timescale/length
        H_2 = period
        
    Arguments:
        hparams: dictionary with all the hyperparameters
        Should have 3 elements with possibly errors
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Exponential Sine Squared Kernel
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 3 elements with possibly errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 3 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 3, "Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'gp_amp', 'gp_timescale', 'gp_per'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['gp_amp'].value
            self.hparams['gp_timescale'].value
            self.hparams['gp_per'].value
        except KeyError:
            raise KeyError("Periodic ExpSinSquared kernel requires 3 hyperparameters:" \
            + "'gp_amp', 'gp_timescale', 'gp_per'")
        
    @staticmethod
    def name():
        return "ExpSinSquared"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_amp', 'gp_timescale', 'gp_per']
        if plotting is True:
            return [r'gp$_{amp}$', r'gp$_{timescale}$', r'gp$_{per}$']
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['gp_per'].value
        amp = self.hparams['gp_amp'].value
        timescale = self.hparams['gp_timescale'].value
        
        message = "Periodic ExpSinSquared Kernel with amp: {}, timescale: {}, per: {}".format(amp, timescale, per)
        print(message)
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Exponential Sine Squared Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the periodic kernel
        '''
        
        per = self.hparams['gp_per'].value
        timescale = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        
        K = np.array(amp**2 * np.exp((-2/timescale**2) * (np.sin(np.pi * dist_e / per))**2))
        
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


class QuasiPer(Kernel):
    '''Class that computes the Quasi-periodic kernel matrix.
    
    Kernel formula from Haywood Thesis 2016, Equation 2.14:
    
        K = H_1^2 . exp{ [-(t-t')^2 / H_2^2] - [ sin^2(pi(t-t')/H_3) / H_4^2] }
    
    in which:
        H_1 = amp
        H_2 = explength
        H_3 = per
        H_4 = perlength
    
    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors
        
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Quasi-peridoic Kernel
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 4 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 4 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 4, "QuasiPeriodic Kernel requires 4 hyperparameters:" \
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'"
        
        # Check if all hyperparameters are numbers
        try:
            self.hparams['gp_per'].value
            self.hparams['gp_perlength'].value
            self.hparams['gp_explength'].value
            self.hparams['gp_amp'].value
        except KeyError:
            raise KeyError("QuasiPeriodic Kernel requires 4 hyperparameters:" \
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'")
        
    @staticmethod
    def name():
        return "QuasiPer"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp']
        if plotting is True:
            return [r'gp$_{per}$', r'gp$_{perlength}$', r'gp$_{explength}$', r'gp$_{amp}$']
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        
        per = self.hparams['gp_per'].value
        perlength = self.hparams['gp_perlength'].value
        explength = self.hparams['gp_explength'].value
        amp = self.hparams['gp_amp'].value
        
        message = "QuasiPeriodic Kernel with amp: {}, per length: {}, per: {}, exp length: {}".format(amp, perlength, per, explength)
        print(message)
        return message  
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Quasi-periodic Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the quasi-periodic kernel
        '''
        
        per = self.hparams['gp_per'].value
        perlength = self.hparams['gp_perlength'].value
        explength = self.hparams['gp_explength'].value
        amp = self.hparams['gp_amp'].value
        
        K = np.array(amp**2 * np.exp(-dist_se/(explength**2)) 
                     * np.exp(-((np.sin(np.pi*dist_e/per))**2)/(perlength**2)))
        
        self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are present or the array is non-square
            pass
        
        return self.covmatrix
    
    
##################################################
########## JITTER QUASI-PERIODIC KERNEL ##########
##################################################    
    
    
class JitterQuasiPer(Kernel):
    '''Class that computes the Quasi-periodic kernel matrix + jitter.
    
    Kernel formula from Haywood Thesis 2016, Equation 2.14:
    
        K = H_1^2 . exp{ [-(t-t')^2 / H_2^2] - [ sin^2(pi(t-t')/H_3) / H_4^2] } + delta_nm jit^2
    
    in which:
        H_1 = amp
        H_2 = explength
        H_3 = per
        H_4 = perlength
        jit = jitter
    
    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 5 elements with errors
        
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Quasi-peridoic Kernel with jitter.
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 5 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 5 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 5, "QuasiPeriodic Kernel requires 5 hyperparameters:" \
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp', 'gp_jit'"
        
        # Check if all hyperparameters are numbers
        try:
            self.hparams['gp_per'].value
            self.hparams['gp_perlength'].value
            self.hparams['gp_explength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_jit'].value
        except KeyError:
            raise KeyError("QuasiPeriodic Kernel requires 5 hyperparameters:" \
            + "'gp_per', 'gp_perlength', 'gp_explength', 'gp_amp', 'gp_jit'")
        
    @staticmethod
    def name():
        return "JitterQuasiPer"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_per', 'gp_perlegth', 'gp_explength', 'gp_amp', 'gp_jit']
        if plotting is True:
            return [r'gp$_{per}$', r'gp$_{perlength}$', r'gp$_{explength}$', r'gp$_{amp}$', r'gp$_{jit}$']
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        per = self.hparams['gp_per'].value
        perlength = self.hparams['gp_perlength'].value
        explength = self.hparams['gp_explength'].value
        amp = self.hparams['gp_amp'].value
        jit = self.hparams['gp_jit'].value
        
        message = "QuasiPeriodic Kernel with amp: {}, per length: {}, per: {}, exp length: {}, jit: {}".format(amp, perlength, per, explength, jit)
        print(message)
        return message  
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Quasi-periodic Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the quasi-periodic kernel
        '''
        
        per = self.hparams['gp_per'].value
        perlength = self.hparams['gp_perlength'].value
        explength = self.hparams['gp_explength'].value
        amp = self.hparams['gp_amp'].value
        jit = self.hparams['gp_jit'].value
        
        K = np.array(amp**2 * np.exp(-dist_se/(explength**2)) 
                     * np.exp(-((np.sin(np.pi*dist_e/per))**2)/(perlength**2))) 
        

        sigma = np.identity(K.shape[0]) * jit**2
        # This is the covariance matrix
        try:
            self.covmatrix = K + sigma
        except ValueError:
            self.covmatrix = K

        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are present or the array is non-square
            pass
        
        return self.covmatrix


#######################################
########## MATERN 5/2 KERNEL ##########
#######################################


class Matern5(Kernel):
    '''Class that computes the Matern 5/2 kernel matrix.
    
    Kernel formula:
        
        K = H_1^2 (1 + (sqr(5)|t-t'|)/H_2 + 5|t-t'|^2/3*H_2^2) exp{-sqr(5)|t-t'|/H_2}
    
    in which:
        H_1 = variance/amp
        H_2 = timescale
    
    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with errors
        
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Matern 5/2 Kernel.
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 2 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 2 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 2, "Matern 5/2 kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_timescale'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['gp_amp'].value
            self.hparams['gp_timescale'].value
        except KeyError:
            raise KeyError("Matern 5/2 kernel requires 2 hyperparameters:" \
            + "'gp_amp', 'gp_timescale'")
        
    @staticmethod
    def name():
        return "Matern5/2"
    
    @staticmethod
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_amp', 'gp_timescale']
        if plotting is True:
            return [r'gp$_{amp}$', r'gp$_{timescale}$']
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        per = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        
        message = "Matern 5/2 Kernel with amp: {}, timesclae: {}".format(amp, per)
        print(message)
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Quasi-periodic Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the matern 5/2 kernel
        '''
        
        timescale = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        
        K = np.array(amp**2 * (1 + (np.sqrt(5)*dist_e/timescale) + (5*dist_se**2/3*timescale**2) * np.exp(-np.sqrt(5)*dist_e/timescale)))
        
        self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix


#######################################
########## MATERN 3/2 KERNEL ##########
#######################################


class Matern3(Kernel):
    '''Class that computes the Matern 3/2 kernel matrix.
    
    Kernel formula:
        
        K = (H_1^2 * (1 + (sqrt(3)*|t-t'|/H_2)) * exp(sqrt(3)*|t-t'|/H_2)) + delta_nm jit^2
        
    in which:
        H_1 = variance/amp
        H_2 = timescale
        jit = jitter
    
    Arguments:
        hparams : dictionary with all the hyperparameters
            Should have 3 elements with errors
        
    '''
    
    def __init__(self, hparams):
        '''
        Initialisation function for the Matern 5/2 Kernel.
        
        Parameters
        ----------
        hparams : dictionary with all the hyperparameters
            Should have 3 elements with errors

        Raises
        ------
        KeyError
            Raised if the dictionary is not composed by the 2 required parameters
        '''
    
        # Initialize
        self.covmatrix = None
        self.hparams = hparams
        
        # Check if we have the right amount of parameters
        assert len(self.hparams) == 3, "Matern 3/2 kernel requires 3 hyperparameters:" \
            + "'gp_amp', 'gp_timescale','gp_jit'"
        
        # Check if all parameters are numbers
        try:
            self.hparams['gp_amp'].value
            self.hparams['gp_timescale'].value
            self.hparams['gp_jit'].value
        except KeyError:
            raise KeyError("Matern 3/2 kernel requires 3 hyperparameters:" \
            + "'gp_amp', 'gp_timescale', 'gp_jit")
        
    @staticmethod
    def name():
        return "Matern3/2"
    
    @staticmethod
    def hparams():
        return ['gp_amp', 'gp_timescale', 'gp_jit']
    def hparams(plotting = True):
        if plotting is False:
            return ['gp_amp', 'gp_timescale', 'gp_jit']
        if plotting is True:
            return [r'gp$_{amp}$', r'gp$_{timescale}$', r'gp$_{jit}$']
    
    @property 
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Printable string indicating the components of the kernel
        '''
        per = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        jit = self.hparams['gp_jit'].value
        
        message = "Matern 3/2 Kernel with amp: {}, timescale: {}, jitter: {}".format(amp, per, jit)
        print(message)
    
    def compute_covmatrix(self, dist_e, dist_se, errors=None):
        '''
        Function to compute the covariance matrix for the Quasi-periodic Kernel
        
        Parameters
        ----------
        dist_e : array, floats
            Spatial distance between each x1-x2 points set in euclidean space
            in formula = (t - t'), from the compute_distances function
        dist_se : array, floats
            Spatial distance between each x1-x2 points set in squared euclidean space
            in formula = (t - t')^2, from the compute_distances function
        errors : array, floats
            Array of the errors, if want to add to diagonal of the covariance matrix

        Returns
        -------
        covmatrix : matrix array, floats
            Covariance matrix computed with the matern 3/2 kernel
        '''
        
        timescale = self.hparams['gp_timescale'].value
        amp = self.hparams['gp_amp'].value
        jit = self.hparams['gp_jit'].value
        
        K = np.array(amp**2 * (1 + (np.sqrt(3)*dist_e/timescale)) * np.exp(-np.sqrt(3)*dist_e/timescale))
        
        sigma = np.identity(K.shape[0]) * jit**2
        
        # This is the covariance matrix
        try:
            self.covmatrix = K + sigma
        except ValueError:
            self.covmatrix = K
        
        # Adding errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except  ValueError:     #if errors are not present or the array is non-square
            pass
        
        return self.covmatrix
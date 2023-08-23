'''
Creator for parameters and priors for the gaussian model

Contains:
    Kernel Parameter creator function
    Prior Parameter creator function
    
    Parameter class
    
    Priors list
    Gaussian Prior class
    Jeffrey Prior class
    Modified Jeffrey Prior class
    Uniform Prior Class

Author: Federica Rescigno, Bryce Dixon
Version 22.08.2023
    '''

import numpy as np
import abc
ABC = abc.ABC



# Kernel Parameter Creator Function

def par_create(kernel):
    """Funciton to create the hyperparameters of the kernel to be used in the gp model

    Parameters
    ----------
    kernel: string
            name of the desired kernel

    Returns
    -------
    hparams: dict
        dictionary of necessary hyperparameters for the relevant kernel
    """        
    if kernel.startswith("Cos") or kernel.startswith("cos"):
        hparams = dict(gp_amp='gp_amp', gp_per='gp_per')
            
    if kernel.startswith("ExpSquare") or kernel.startswith("expsquare") or kernel.startswith("Expsquare") or kernel.startswith("expSquare"):
        hparams = dict(gp_amp = 'gp_amp', gp_timescale = 'gp_timescale')
        
    if kernel.startswith("Periodic") or kernel.startswith("periodic") or kernel.startswith("ExpSin") or kernel.startswith("expsin") or kernel.startswith("Expsin"):
        hparams = dict(gp_amp='gp_amp', gp_timescale='gp_timescale', gp_per='gp_per')
        
    if kernel.startswith("QuasiPer") or kernel.startswith("quasiper") or kernel.startswith("Quasiper"):
        hparams = dict(gp_per='gp_per', gp_perlength='gp_perlength', gp_explength='gp_explength', gp_amp='gp_amp')
        
    if kernel.startswith("Jit") or kernel.startswith("jit"):
        hparams = dict(gp_per='gp_per', gp_perlength='gp_perlength', gp_explength='gp_explength', gp_amp='gp_amp', gp_jit='gp_jit')
        
    if kernel.startswith("Matern5") or kernel.startswith("matern5") or kernel.startswith("Matern 5") or kernel.startswith ("matern 5"):
        hparams = dict(gp_amp = 'gp_amp', gp_timescale = 'gp_timescale')  
              
    if kernel.startswith("Matern3") or kernel.startswith("matern3") or kernel.startswith("Matern 3") or kernel.startswith ("matern 3"):
        hparams = dict(gp_amp='gp_amp', gp_timescale='gp_timescale', gp_jit="gp_jit")
        
    return hparams


def PRINTPRIORDER(pri_name = None):
    """Function to print the information and orders for the prior values

    Parameters
    ----------
    pri_name: string, optional
        name of the desired prior to check
        Defaults to None
    """
    if pri_name == None:
        pri_name = "no prior"
    
    if pri_name.startswith("Gaus") or pri_name.startswith("gaus"):
        print("List should take the form [mu, sigma] where all values are floats or ints")
    elif pri_name.startswith("Jef") or pri_name.startswith("jef"):
        print("List should take the form [minval, maxval] where all values are float or ints")
    elif pri_name.startswith("Mod") or pri_name.startswith("mod"):
        print("List should take the form [minval, maxval, kneeval] where all values are floats or ints")
    elif pri_name.startswith("Uni") or pri_name.startswith("uni"):
        print("List should take the form [minval, maxval] where all values are floats or ints")
    else:
        print("Gaussian: List should take the form [mu, sigma] where all values are floats or ints \n"
              "Jeffery: List should take the form [minval, maxval] where all values are floats or ints \n"
              "Modified Jeffery: List should take the form [minval, maxval, kneeval] where all values are floats or ints \n"
              "Uniform: List should take the form [minval, maxval] where all values are floats or ints")
        
    
# Prior Parameter Creator Function

def pri_create(param_name, prior, vals = None):
    """Funciton to generate a set of parameters necessary for the chosen prior

    Parameters
    ----------
    param_name: string
        name of parameter that the prior is being assigned to - should be the same as it appears in the kernel or model
    prior: string
        name of the desired prior
    vals: list or tuple of floats or ints, optional
        list of floats containing the prior parameters in order specified by the PRINTPRIORDER function.
        To view which values belong in the list and the format, run the PRINTPRIORDER function.
    
    Raises
    ------
    Assertion:
        Raised if vals is not None and not a list or a tuple
    Assertion:
        Raised if vals is not None and not made of floats or ints
    Assertion:
        Raised if length of the vals list does not match the required length for the prior
    Assertion:
        Raised if the minval is larger than the maxval for the prior

    Returns
    -------
    prior_params: dictionary
        dictionary of all prior parameters
    """
    
    # if vals is a numpy array, change to a list so the code runs properly
    if type(vals) == np.ndarray:
        vals = vals.tolist()
        
    if vals != None:
        assert type(vals) == list or tuple, "vals must be inputted as a list or tuple"
        for i in vals:
            assert type(i) == float or int, "vals must be a list or tuple of floats or ints"
    
    if prior.startswith("Gauss") or prior.startswith("gauss"):
        prior = "Gaussian"
        if vals == None:
            vals = [0.,0.]
        assert len(vals) == 2, "Gaussian priors require a list with 2 values, mu and sigma"
        prior_dict = dict(mu=vals[0], sigma=vals[1])
        prior_params = (param_name, prior, prior_dict)

    if prior.startswith("Jeffrey") or prior.startswith("jeffrey"):
        prior = "Jeffrey"
        if vals == None:
            vals = [0.,0.]
        assert len(vals) == 2, "Jeffrey priors require a list with 2 values, minval and maxval"
        prior_dict = dict(minval=vals[0], maxval=vals[1])
        prior_params = (param_name, prior, prior_dict)
        assert vals[0] <= vals[1], 'Minval must be less than maxval.' 

    if prior.startswith("Mod") or prior.startswith("mod"):
        prior = "Modified_Jeffrey"
        if vals == None:
            vals = [0.,0.,0.]
        assert len(vals) == 3, "Modified Jeffrey priors require a list with 3 values, minval, maxval and kneeval"
        prior_dict = dict(minval=vals[0], maxval=vals[1], kneeval=vals[2])
        prior_params = (param_name, prior, prior_dict)
        assert vals[0] <= vals[1], 'Minval must be less than maxval.' 

    if prior.startswith("Uni") or prior.startswith("uni"):
        prior = "Uniform"
        if vals == None:
            vals = [0.,0.]
        assert len(vals) == 2, "Uniform priors require a list with 2 values, minval and maxval"
        prior_dict = dict(minval=vals[0], maxval=vals[1])
        prior_params = (param_name, prior, prior_dict)
        assert vals[0] <= vals[1], 'Minval must be less than maxval.'
    
    PRIORS = defPriorList()
    assert prior in PRIORS.keys(), 'prior not yet implemented. Pick from available priors: ' + str(PRIORS.keys()) 
            
    return prior_params



# parameter class

class parameter:
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
            Is the variable allowed to vary? The default is True.
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


#############################
########## PRIORS ###########
#############################

PRIORS = {
    "Gaussian": ['hparam', 'mu', 'sigma'],
     "Jeffrey": ['hparam', 'minval', 'maxval'],
     "Modified_Jeffrey": ['hparam', 'minval', 'maxval', 'kneeval'],
     "Uniform": ['hparam', 'minval', 'maxval']}

def PrintPriorList():
    """Function to print the list of all currently available PRIORS"""
    print("Implemented priors:")
    print(PRIORS)

def defPriorList():
    """Function to return the list of all currently available PRIORS"""
    return PRIORS
    

# parent prior

class Prior(ABC):
    '''
    Parent class for all priors. All new priors should inherit from this class and follow its structure.
    Each new prior will require a __init__() method to override the parent class. In the __init__ function, call the neccesary parameters.
    '''
    
    @abc.abstractproperty
    def __repr__(self):
        '''Prints message with name of the Prior and assigned parameters'''       
        pass
    
    @abc.abstractmethod
    def logprob(self, x):
        '''computes the natural logarithm of the probability of x being the best fit'''
        pass
    
       
# Gaussian prior

class Gaussian(Prior):
    '''Gaussian prior computed as:
        
        -0.5 * ((x - mu) / sigma)**2 -0.5 * np.log(2*pi * sigma**2)
        
    Args:
        hparam (string): parameter label
        mu (float): centre of Gaussian Prior
        sigma (float): width of the Gaussian Prior
    '''
    
    def __init__(self, hparam, mu, sigma):
        '''
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        mu : float
            Center of the Gaussian prior
        sigma : float
            FWHM of the Gaussian prior
        '''
        self.hparam = hparam
        self.mu = float(mu)
        self.sigma = float(sigma)
        
    
    @property    
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior.
        '''
        message = ("Gaussian prior on the parameter {}, with mu = {} and sigma = {}").format(self.hparam, self.mu, self.sigma)
        print(message)
        return message 
    

    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        logprob = -0.5 * ((x - self.mu) / self.sigma)**2 -np.log(np.sqrt(2*np.pi) * self.sigma)
        return logprob


# Jeffrey Prior

class Jeffrey(Prior):
    '''Jeffrey prior computed as:
        
        p(x) proportional to  1/x
        with upper and lower bound to avoid singularity at x = 0
        
        and normalized as:
            1 / ln(maxval/minval)
    
    Args:
        hparam (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    def __init__(self, hparam, minval, maxval):
        '''
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        '''
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval
        
        assert self.minval < self.maxval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
    
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Jeffrey prior on the paramter {}, with minimum and maximum values of x = ({}, {})").format(self.hparam, self.minval, self.maxval)
        print(message)
        return message
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array, floats
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -10e5
            return logprob
        else:
            normalisation = 1./(np.log(self.maxval/self.minval))
            prob = normalisation * 1./x
            logprob = np.log(prob)
            return logprob


# Modified Jeffrey Prior

class Modified_Jeffrey(Prior):
    ''' Modified Jeffrey prior computed as:
        
        p(x) proportional to  1/(x-x0)
        with upper bound
    
    
    Args:
        hparam (string): parameter label
        kneeval (float): x0, knee of the Jeffrey prior
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    
    def __init__(self, hparam, minval, maxval, kneeval):
        '''
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        kneeval: float
            Kneww value of prior (x0)
        '''
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval
        self.kneeval = kneeval
        
        assert self.minval < self.maxval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
        
    
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Modified Jeffrey prior on the paramter {}, with minimum and maximum values of x = ({}, {}) and knee = {}").format(self.hparam, self.minval, self.maxval, self.kneeval)
        print(message)
        return message
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : float
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -10e5
            return logprob
        else:
            normalisation = 1./(np.log((self.maxval+self.kneeval)/(self.kneeval)))
            prob = normalisation * 1./(x-self.kneeval)
            logprob = np.log(prob)
            return logprob


# Uniform Prior

class Uniform(Prior):
    ''' Uniform prior
    
    Args:
        hparam (string): parameter label
        minval (float): minimum allowed value
        maxval (float): maximum allowed value
    '''
    
    
    def __init__(self, hparam, minval, maxval):
        '''
        Parameters
        ----------
        hparam : string
            Label of chosen parameter
        minval : float
            Mininum value of x
        maxval : float
            Maximum vlaue of x
        '''
        self.hparam = hparam
        self.minval = minval
        self.maxval = maxval
    
        assert self.maxval > self.minval, "Minimum value {} must be smaller than the maximum value {}".format(self.minval, self.maxval)
    
    
    @property
    def __repr__(self):
        '''
        Returns
        -------
        message : string
            Description of the prior
        '''
        message = ("Uniform prior on the paramter {}, with minimum and maximum values of x = ({}, {})").format(self.hparam, self.minval, self.maxval)
        print(message)
        return message
    
    
    def logprob(self, x):
        '''
        Parameters
        ----------
        x : array
            Assumed value of the parameter (number space over which to sample)

        Returns
        -------
        logprob : float
            Natural  logarithm of the probabiliy of x being the best fit
        '''
        if x < self.minval or x > self.maxval:
            logprob = -10e5
            return logprob
        else:
            logprob = 0
            return logprob


    
'''
GP likelihood calculation for the gaussian model
'''

# Contains:
#     GP Likelihood class
#
# Author: Federica Rescigno, Bryce Dixon
# Version 22.08.2023


import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

import magpy_rv.parameters as par
import magpy_rv.kernels as ker
import magpy_rv.models as mod
from magpy_rv.mcmc_aux import get_model 


#######################################
############ GP Likelihood ############
#######################################


class GPLikelihood:
    '''
    Gaussian Process likelihood
    
    Parameters
    ----------
    x : array or list, floats
        Time series of the radial velocity
    y : array or list, floats
        Radial velocity values
    yerr : array or list, floats
        Radial velocity errors
    hparamters: dictionary
        dictionary of all hyper parameters considered
    kernel_name: string
        name of the used kernel
    model_y: array, floats, optional
        Array of y (rv) values from the model chosen
        Default to None
    model_param: dictionary, optional
        dictionary of all model parameters considered
        Default to None
        
    Raises
    ------
    Assertion:
        Raised if the hyperparameters are not a dictionary
    Assertion:
        Raised if the a model is in use and the model parameters are not a dictionary
    KeyError:
        Raised if a model is in use and model_y is not given
    KeyError:
        Raised if a model_y is given but no parameters are specified
    KeyError:
        Raised if a model_y is given but no model is in use
    '''
    
    def __init__(self, x, y, yerr, hparameters, kernel_name, model_y = None, model_param = None):
        '''
        Parameters
        ----------
        x : array or list, floats
            Time series of the radial velocity
        y : array or list, floats
            Radial velocity values
        yerr : array or list, floats
            Radial velocity errors
        hparamters: dictionary
            dictionary of all hyper parameters considered
        kernel_name: string
            name of the used kernel
        model_y: array, floats, optional
            Array of y (rv) values from the model chosen
            Default to None
        model_param: dictionary, optional
            dictionary of all model parameters considered
            Default to None
            
        Raises
        ------
        Assertion:
            Raised if the hyperparameters are not a dictionary
        Assertion:
            Raised if the a model is in use and the model parameters are not a dictionary
        KeyError:
            Raised if a model is in use and model_y is not given
        KeyError:
            Raised if a model_y is given but no parameters are specified
        KeyError:
            Raised if a model_y is given but no model is in use
        '''
        self.x = np.array(x)    #Time series (must be array)
        self.y = np.array(y)    #Radial velocity array (must be array)
        
        if yerr is None:
            # if no errors are provided, create an array of error equal to 0.1 times the y value
            yerr = np.array(y)*0.1
            
        self.yerr = np.array(yerr) #Radial velocity error values (must be array)
        err_check = not np.any(self.yerr)
        if err_check is True:
            # raise an error if an array of zeros is given as yerr
            raise KeyError("Uncertainties should not be zero")
        
        assert type(hparameters) == dict, "hyperparameters should be a dictionary generated by par.parameter"
        self.hparameters = hparameters #Dictionary of all parameter each of Parameter class as: value, vary, mcmc scale 
        self.hparam_names = hparameters.keys()
        self.hparam_values = []
        for key in hparameters.keys():
            self.hparam_values.append(hparameters[key].value)
        
        self.kernel_name = kernel_name
        
        if type(model_y) == np.ndarray:
            model_y = model_y.tolist()
        
        # check if a model is required, if not create a blank model
        
        if model_y == None:
            if model_param == None or "no" in model_param.keys():
                # run the get_model function from the MCMC file, this will likely be changed when that file is re-written
                model_list = ["no"]
                model_param = mod.mod_create(model_list)
                model_param["no"]=par.parameter(value=0., error=0., vary=False)
                model_y = get_model(model_list, self.x, model_param, to_ecc=False)
                self.model_y = np.array(model_y)
            else:
                raise KeyError("model_y must be provided when using a model")
        else:
            if model_param == None:
                raise KeyError("model parameters must be specified when using a model")  
            else:
                self.model_y = np.array(model_y)
        
        assert type(model_param) == dict, "Model parameters should be a dictionary generated by par.parameter"
        self.model_param = model_param #Dictionary of all parameters of the model
        self.model_param_names = model_param.keys()
        self.model_param_values = []
        for key in model_param.keys():
            self.model_param_values.append(model_param[key].value)
        
    
    def __repr__(self):
        '''
        Returns
        -------
        message : string
        parameters : string
            List of all parameters with values
        '''
        message = "Gaussian Process Likelihood object, computed with a {} kernel \n".format(self.kernel_name)
        
        parameters = "Kernel parameters: \n"
        for i in range(len(self.hparam_values)):
            parameters += ("{} with initial value {} \n").format(self.hparam_names[i], self.hparam_values[i])
        
        model_parameters = "Model parameters: \n"
        for i in range(len(self.model_param_values)):
            model_parameters += ("{} with initial value {} \n").format(self.model_param_names[i], self.model_param_values[i])
        
        print(message)
        print(parameters)
        print(model_parameters)
        return message, parameters, model_parameters
    
    
    def compute_kernel(self, x1, x2):
        '''
        Parameters
        ----------
        x1 : array or list, floats
            array to be used in calculation of covariance matrix
        x2 : array or list, floats
            other array to be used in the calculations of covariance matrix
            

        Returns
        -------
        covmatrix : array, floats
            Covariance matrix of the chosen set of varaibles x1 and x2
        '''
        
        # call the kernels list
        KERNELS = ker.defKernelList()
        
        x1=np.array(x1)
        x2=np.array(x2)
        yerr = self.yerr
        
        # call the relevant kernel class and set the name to be exactly what appears in the kernels list so it can be checked
        kernel_name = self.kernel_name
        if kernel_name.startswith("Cos") or kernel_name.startswith("cos"):
            self.kernel = ker.Cosine(self.hparameters)
            kernel_name = "Cosine"
        if kernel_name.startswith("ExpSquare") or kernel_name.startswith("expsquare") or kernel_name.startswith("Expsquare") or kernel_name.startswith("expSquare"):
            self.kernel = ker.ExpSquared(self.hparameters)
            kernel_name = "ExpSquared"
        if kernel_name.startswith("Periodic") or kernel_name.startswith("periodic") or kernel_name.startswith("ExpSin") or kernel_name.startswith("Expsin") or kernel_name.startswith("expsin"):
            self.kernel = ker.ExpSinSquared(self.hparameters)
            kernel_name = "ExpSinSquared"
        if kernel_name.startswith("QuasiPer") or kernel_name.startswith("quasiper") or kernel_name.startswith("Quasiper"):
            self.kernel = ker.QuasiPer(self.hparameters)
            kernel_name = "QuasiPer"
        if kernel_name.startswith("jit") or kernel_name.startswith("Jit"):
            self.kernel = ker.JitterQuasiPer(self.hparameters)
            kernel_name = "JitterQuasiPer"
        if kernel_name.startswith("Matern5") or kernel_name.startswith("matern5") or kernel_name.startswith("Matern 5") or kernel_name.startswith ("matern 5"):
            self.kernel = ker.Matern5(self.hparameters)
            kernel_name = "Matern5/2"
        if kernel_name.startswith("Matern3") or kernel_name.startswith("matern3") or kernel_name.startswith("Matern 3") or kernel_name.startswith("matern 3"):
            self.kernel = ker.Matern3(self.hparameters)
            kernel_name = "Matern3/2"
        
        # check if the kernel is in the list of implemented kernels
        assert kernel_name in KERNELS.keys(), 'Kernel not yet implemented. Pick from available kernels: ' + str(KERNELS.keys())
        
        # run the relevant functions to compute the covariance matrix of the chosen kernel
        dist_e, dist_se = ker.compute_distances(x1, x2)    
        if np.array_equal(x1,x2) is True and x1.all() == self.x.all():
            covmatrix = self.kernel.compute_covmatrix(dist_e, dist_se, yerr)
        else:
            covmatrix = self.kernel.compute_covmatrix(dist_e, dist_se, 0.)

        return covmatrix
            
    
    
    def residuals(self):
        '''
        Residuals internal to the computation:
            RV - RV_model

        Returns
        -------
        res : array
            New RVs for internal calculations
        '''
        self.new_y = self.y - self.model_y
        res = self.new_y
        return res



    def logprob(self):
        '''
        Computes the natural logarith of the likelihood of the gaussian fit.
        Following the equation:
        
        .. math::
        
            ln(L) = -\\frac{n}{2} ln(2\\pi) \\cdot -\\frac{1}{2} ln(det(K)) -\\frac{1}{2} Y^T \\cdot (K-1) \\cdot Y
        
        Returns
        -------
        logL : float
            Ln of the likelihood
        '''
        # Compute kernel covariance matrix and the y (rvs) to model
        K = self.compute_kernel(self.x, self.x)
        Y = self.residuals()
        # Compute likelihood, formula 2.28 in Raphie Thesis
        # Part 1: get ln of determinant
        sign, logdetK = np.linalg.slogdet(K)
        #Part 2: compute Y.T * K-1 * Y
        A = cho_solve(cho_factor(K), Y)
        alpha = np.dot(Y, A)
        # Part 3: all together
        N = len(Y)
        logprob = - (N/2)*np.log(2*np.pi) - 0.5*logdetK - 0.5*alpha
        self.logprob = logprob

        return logprob



    def priors(self, param_name, prior_name, prior_parameters):
        '''Fuction to compute the natural logarithm of the prior likelihood
        
        Parameters
        ----------
        hparam_name : string
            Name of the hyperparameter on which to impose the prior
        prior_name : string
            Name of the prior to apply
        prior_parameters : dictionary
            Dictionary containing all necessary prior parameters
            (get with pri_create and assign values)
        
        Raises
        ------
        Assertion:
            Raised if the selected prior is not in the list of currently available priors    

        Returns
        -------
        prior_logprob : float
            Natural logarithm of the prior likelihood
        '''
        
        PRIORS = par.defPriorList()
        
        try:
            # check for priors on the hyperparameters
            self.prior_param = self.hparameters[param_name].value
        except KeyError:
            # check for priors on the model parameters
            self.prior_param = self.model_param[param_name].value
        
        self.prior_name = prior_name
        self.prior_parameters = prior_parameters
        
        # check which prior is in use, assign relevant values from the prior and apply them to the prior class then call the relevant logprob function.
        if prior_name.startswith("Gaussian") or prior_name.startswith("gaussian"):
            # set the prior name to be exactly the same a sit appears on the priors list
            self.prior_name = "Gaussian"
            self.mu = self.prior_parameters["mu"]
            self.sigma = self.prior_parameters["sigma"]
            self.prior = par.Gaussian(param_name, self.mu, self.sigma)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        if prior_name.startswith("Jeffrey") or prior_name.startswith("jeffrey"):
            self.prior_name = "Jeffrey"
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.prior = par.Jeffrey(param_name, self.minval, self.maxval)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        if prior_name.startswith("Modified") or prior_name.startswith("modified"):
            self.prior_name = "Modified_Jeffrey"
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.kneeval = self.prior_parameters["kneeval"]
            self.prior = par.Modified_Jeffrey(param_name, self.minval, self.maxval, self.kneeval)
            prior_logprob = self.prior.logprob(self.prior_param)

        
        if prior_name.startswith("Uni") or prior_name.startswith("uni"):
            self.prior_name = "Uniform"
            self.minval = self.prior_parameters["minval"]
            self.maxval = self.prior_parameters["maxval"]
            self.prior = par.Uniform(param_name, self.minval, self.maxval)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        # check if the prior is in the list of available priors
        assert self.prior_name in PRIORS.keys(), 'Prior not yet implemented. Pick from available priors: ' + str(PRIORS.keys())
        
        return prior_logprob
    
    
    
    def LogL(self, prior_list):
        '''
        Function to compute the log likelihood after applying the priors
        
        Parameters
        ----------
        prior_list : list of sets of 3 objects
            List of the priors applied. Each item in the list should countain the following
            3 objects:
                String of the name of the parameter the prior is applied to
                
                String of the name of the prior
                
                pri_create dictionary of the prior
        
        Raises
        ------
        Assertion:
            Raised if the prior_list is not a list
        Assertion:
            Raised if the name of the parameter is not a string
        Assertion:
            Raised if the name of the prior is not a string
        Assertion:
            Raised if the prior parameters are not a dictionary

        Returns
        -------
        LogL : float
            Final ln of likelihood after applying all posteriors from priors
        '''
        
        # check if prior_list is a list
        assert type(prior_list) == list, "prior_list should be a list of sets of 3 objects: name of parameter, name of prior, dictionary of prior parameters"
        
        LogL = self.logprob()
        for i in range(len(prior_list)):
            hparam = prior_list[i][0]
            assert type(hparam) == str, "Name of parameter should be a string"
            name_prior = prior_list[i][1]
            assert type(name_prior) == str, "Name of prior should be a string"
            prior_param = prior_list[i][2]
            assert type(prior_param) == dict, "Prior parameters must be in the form of a dictionary created using pri_create"
            LogL += self.priors(hparam, name_prior, prior_param)
        
        return LogL
                     
            
         
    def predict(self, xpred, FullCov = False):
        '''
        Parameters
        ----------
        xpred : array, floats
            X array over which to do the prediction
        FullCov : True/False, optional
            Want to return the full covariance? The default is False.

        Returns
        -------
        pred_mean: array, floats
            Predicted values of the y axis
        stdev: array, floats
            Standard deviation of the y points, to be used as error
        OR
        np.array(pred_cov): array, floats
            Full covariance of the data set
        '''
   
        Y = self.residuals()
        y = Y.T
        
        K = self.compute_kernel(self.x, self.x)
        Ks = self.compute_kernel(xpred, self.x)
        Kss = self.compute_kernel(xpred, xpred)
        
        # Predicted mean = Ks * K-1 * y
        alpha = cho_solve(cho_factor(K), y)
        pred_mean = np.dot(Ks, alpha).flatten()
        pred_mean = np.array(pred_mean)
            
        #Predicted errors = Kss - Ks * K-1 * Ks.T
        beta = cho_solve(cho_factor(K), Ks.T)
        pred_cov = Kss - np.dot(Ks, beta)
        
        
        # Turn plot to true for in-depth checks on health of matrixes
        plots = False
        if plots is True:
            
            from matplotlib.colors import LogNorm
            
            fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1, 1.7]})
            
            vmin = 0
            vmax1 = max([max(l) for l in K])
            vmax2 = max([max(o) for o in Ks])
            vmax3 = max([max(p) for p in Kss])
            vmaxs = np.array([vmax1, vmax2, vmax3])
            vmaxs.sort()
            vmax = vmaxs[-1]
            
            ax1.imshow(K, vmin=vmin, vmax=vmax)
            ax1.title.set_text('K')
            ax2.imshow(Ks, vmin=vmin, vmax=vmax)
            ax2.title.set_text('Ks')
            im3 = ax3.imshow(Kss, vmin=vmin, vmax=vmax)
            ax3.title.set_text('Kss')
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            fig.colorbar(im3, cax=cax)
            plt.show()
        
        
        if FullCov is True:
            return pred_mean, np.array(pred_cov)
        else:
            var = np.array(np.diag(pred_cov)).flatten()
            
            stdev = np.sqrt(var)
            
            stdev = np.array(stdev)
            return pred_mean, stdev
        
        
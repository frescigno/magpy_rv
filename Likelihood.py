#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Gaussian Process Likelyhood calculator

Contains:
    GPLikelihood

Author: Federica Rescigno
Version: 27.06.2022
'''

import numpy as np
import Kernels
import Priors
from scipy.linalg import cho_factor, cho_solve

class GPLikelihood:
    ''' Gaussian Process Likelyhood
    Computed as:
        see Eq. ???? in Rescigno et al. in prep b'''
    
    def __init__(self, x, y, yerr, model_y, mod_params, hparams, kernel_name):
        '''
        Parameters
        ----------
        x : array or list, floats
            Time series of observations
        y : array or list, floats
            Y-axis values of the observations
        yerr : array or list, floats
            Y-axis data errors
        model_y: array, floats
            Array of y values from the models chosen
        mod_params: dictionary
            Dictionary of all the model parameters considered
        hparams: dictionary
            Cictionary of all the kernel hyperparameters considered
        kernel_name: string
            name of the used kernel
        '''
        
        # Observational data and model
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        
        # Kernel information
        self.hparams = hparams
        self.hparams_names = hparams.keys()
        self.hparams_values = []
        for key in hparams.keys():
            self.hparams_values.append(hparams[key].value)
        self.kernel_name = kernel_name
        KERNELS = Kernels.Kernel_list()
        assert self.kernel_name in KERNELS.keys(), 'Kernel not yet implemented. Pick from available kernels: ' + str(KERNELS.keys())
        
        # Model information
        self.model_y = np.array(model_y)
        self.mod_params = mod_params
        self.mod_params_names = mod_params.keys()
        self.mod_params_values = []
        for key in mod_params.keys():
            self.mod_params_values.append(mod_params[key].value)
        
    def __repr__(self):
        '''
        Returns
        -------
        message : string
        parameters : string
            List of all parameters with values
        '''
        
        message = "Gaussian Process Likelyhood object, computed with a {} kernel \n".format(self.kernel_name)
        
        hparameters = "Kernel parameters: \n"
        for i in range(len(self.hparams_values)):
            hparameters += ("{} = {} \n").format(self.hparams_names[i], self.hparams_values[i])
        
        model_parameters = "Model parameters: \n"
        for i in range(len(self.mod_params_values)):
            model_parameters += ("{} = {} \n").format(self.mod_params_names[i], self.mod_params_values[i])
        
        print(message)
        print(hparameters)
        print(model_parameters)
        return message, hparameters, model_parameters
    
    
    def compute_kernel(self, x1, x2):
        '''
        Parameters
        ----------
        x1 : array or list, floats
        x2 : array or list, floats

        Returns
        -------
        covmatrix : array, floats
            Covariance matrix of the chosen set of varaibles x1 and x2
        '''
                
        x1=np.array(x1)
        x2=np.array(x2)
        
        # In future this will become a get_kernel function with a numb_param_perkernel counter
        if self.kernel_name.startswith("Quasi") or self.kernel_name.startswith("quasi"):
            self.kernel = Kernels.QuasiPer(self.hparams)
        if self.kernel_name.startswith("Periodic") or self.kernel_name.startswith("periodic") or self.kernel_name.startswith("ExpSin") or self.kernel_name.startswith("Expsin") or self.kernel_name.startswith("expsin"):
            self.kernel = Kernels.ExpSinSquared(self.hparams)
        if self.kernel_name.startswith("Cos") or self.kernel_name.startswith("cos"):
            self.kernel = Kernels.Cosine(self.hparams)
        if self.kernel_name.startswith("ExpSqu") or self.kernel_name.startswith("expsqu") or self.kernel_name.startswith("Expsqu"):
            self.kernel = Kernels.ExpSquared(self.hparams)
        if self.kernel_name.startswith("Matern5") or self.kernel_name.startswith("matern5"):
            self.kernel = Kernels.Matern5(self.hparams)

        
        self.kernel.compute_distances(x1, x2)
        if np.array_equal(x1,x2) is True and x1.all() == self.x.all():
            covmatrix = self.kernel.compute_covmatrix(self.yerr)
        else:
            covmatrix = self.kernel.compute_covmatrix(0.)
        
        return covmatrix
    
    
    def GPonly_residuals(self):
        '''
        Residuals internal to the computation, they represent the data modeled by the GP:
            Observations - Model

        Returns
        -------
        GPres : array, floats
            New data for internal calculations
        '''
        self.new_y = self.y - self.model_y
        GPres = self.new_y
        return GPres

    def initial_logprob(self):
        '''
        Computes the natural logarith of the likelihood of the gaussian fit.
        Following the equation:
            ln(L) = -n/2 ln(2pi) * -1/2 ln(det(K)) -1/2 Y.T dot K-1 dot Y
        
        Returns
        -------
        logprob : float
            Natural logarith of the likelihood
        '''
        # Compute kernel covariance matrix and the y to model
        K = self.compute_kernel(self.x, self.x)
        Y = self.GPonly_residuals()
        # Part 1: get ln of determinant
        sign, logdetK = np.linalg.slogdet(K)
        #Part 2: compute Y.T * K-1 * Y
        A = cho_solve(cho_factor(K), Y)
        alpha = np.dot(Y, A)
        # Part 3: all together
        N = len(Y)
        self.logprob = - (N/2)*np.log(2*np.pi) - 0.5*logdetK - 0.5*alpha
        
        return self.logprob
    
    def priors(self, param_name, prior_name, prior_info):
        '''
        Parameters
        ----------
        param_name : string
            Name of the hyperparameter on which to impose the prior
        prior_name : string
            Name of the prior to apply
        prior_info : dictionary
            Dictionary containing all necessary prior parameters
            (get with Prior_Par_Creator and assign values)

        Returns
        -------
        prior_logprob : float
            Natural logarith of the prior likelihood
        '''
        # First check that this is an implemented prior
        self.prior_name = prior_name
        PRIORS = Priors.Prior_list()
        assert self.prior_name in PRIORS.keys(), 'Prior not yet implemented. Pick from available priors: ' + str(PRIORS.keys())
        self.prior_info = prior_info

        # Get the value of the parameter over which the prior is being implemented
        try:
            self.prior_param = self.hparams[param_name].value
        except KeyError:
            try:
                self.prior_param = self.mod_params[param_name].value
            except KeyError:
                raise KeyError("This is not an included parameter in the list of kernel and model parameters")
        
        # Compute the likelihood 
        if prior_name.startswith("Gauss") or prior_name.startswith("gauss"):
            mu = self.prior_info["mu"]
            sigma = self.prior_info["sigma"]
            self.prior = Priors.Gaussian(param_name, mu, sigma)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        if self.prior_name.startswith("Jeff") or self.prior_name.startswith("jeff"):
            minval = self.prior_info["minval"]
            maxval = self.prior_info["maxval"]
            self.prior = Priors.Jeffreys(param_name, minval, maxval)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        if self.prior_name.startswith("Modified") or self.self.prior_name.startswith("modified"):
            minval = self.prior_info["minval"]
            maxval = self.prior_info["maxval"]
            kneeval = self.prior_info["kneeval"]
            self.prior = Priors.Modified_Jeffreys(param_name, minval, maxval, kneeval)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        if self.prior_name.startswith("Uni") or self.prior_name.startswith("uni"):
            minval = self.prior_info["minval"]
            maxval = self.prior_paraprior_infoeters["maxval"]
            self.prior = Priors.Uniform(param_name, minval, maxval)
            prior_logprob = self.prior.logprob(self.prior_param)
        
        return prior_logprob
    
    
    def LogL(self, prior_list):
        '''
        Parameters
        ----------
        prior_list : list of sets of 3 objects
            List of the priors applied. Each item in the list should countain the following
            3 objects:
                String of the name of the parameter the prior is applied to
                String of the name of the prior
                Prior_Par_Creator dictionary of the prior

        Returns
        -------
        LogL : float
            Final ln of likelyhood after applying all posteriors from priors
        '''
        
        LogL = self.initial_logprob()
        for i in range(len(prior_list)):
            param = prior_list[i][0]
            name_prior = prior_list[i][1]
            prior_info = prior_list[i][2]
            LogL += self.priors(param, name_prior, prior_info)
        
        return LogL
    
    def predict(self, xpred, plot_covmatrix=False):
        '''
        Parameters
        ----------
        xpred : array, floats
            X array over which to do the prediction
        plot_covmatrix: True/False, optional
            Plot the covariance matrix? Defaults is False

        Returns
        -------
        pred_mean: array, floats
            Predicted values of the y axis
        stdev: array, floats
            Standard deviation of the y points, to be used as error
        '''
        
        Y = self.GPonly_residuals()
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
        
        if plot_covmatrix is True:
            import matplotlib.pyplot as plt
                        
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
        
        var = np.array(np.diag(pred_cov)).flatten()
        stdev = np.array(np.sqrt(var))
        return pred_mean, stdev
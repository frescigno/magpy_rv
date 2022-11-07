#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
MCMC affine base code

Contains:
    MCMC class code
    ...

Author: Federica Rescigno
Version: 27.06.2022
'''

import numpy as np
import random
import time

import Likelihood as lk
import Get as get
import Kernels
import Models
import Check
import Parameter
import Auxiliary as aux
import Check as check



class MCMC:
    
    def __init__(self, x, y, y_err, hparam0, kernel_names, mod_par0, model_names, prior_list, numb_chains=100):
        '''
        Parameters
        ----------
        x : array, floats
            X-array, most often time array, of the observations
        y : array, floats
            Y-array, can be radial velocity array, of the observations
        y_err : array, floats
            Errors on the radial velocity array
        hparam0 : dictionary
            Set of hyper-parameters for the kernel, with value, error and vary 
        kernel_names : string
            Name of chosen kernel
        mod_par0 : dictionary
            Set of parameters for the models, with value, error and vary
        model_names : list, strings
            Names of chosen models
        prior_list : list
            List inlcuding in order the name of parameter, the name of the prior and the dictionary of the prior parameters (see posterior function in GP_solar.py for better description)
        numb_chains : integer, optional
            Number of chains requested. The default is 100.
        '''
        
        
        # Save unchanging inputs
        self.x = x
        self.y = y
        self.y_err = y_err
        self.numb_chains = numb_chains
        self.prior_list = prior_list
        
        # Check if the kernal names are properly given
        if isinstance(self.model_names, str):
            self.kernel_names = kernel_names
        else:
            raise TypeError("kernel_names must be a string or a list of strings")
        
        # Check if model_names are properly given
        if isinstance(self.model_names, list):
            self.numb_models = len(self.model_names)
        elif isinstance(self.model_names, str):
            self.numb_models = 1
            # Put it into list for easy computation in loops
            self.model_names = [model_names]
        else:
            raise TypeError("model_name must be a string or a list of strings")
        
        # Set up storing arrays
        self.hparameter_list = []
        self.model_parameter_list = []
        self.logL_list = []
        self.accepted = []

        
        # Get initial guesses for hyperparameters and save them as 0
        # Save also errors for chains initial positions population
        # Also include a list of the names of the parameters, in the proper order (useful when have multiple kernels)
        self.single_hp0 = []
        self.hp_err = []
        self.hp_vary = []
        self.hp_names = []
        for key in hparam0.keys():
            self.single_hp0.append(hparam0[key].value)
            self.hp_err.append(hparam0[key].error)
            self.hp_vary.append(hparam0[key].vary)
            self.hp_names.append(hparam0[key])
            
        # Do the same for model parameters
        # Also include a list of the names of the parameters, in the proper order (useful when have multiple models)
        self.single_modpar0 = []
        self.modpar_err = []
        self.modpar_vary = []
        self.modpar_names = []
        for key in mod_par0.keys():
            self.single_modpar0.append(mod_par0[key].value)
            self.modpar_err.append(mod_par0[key].error)
            self.modpar_vary.append(mod_par0[key].vary)
            self.modpar_names.append(key)
       
        # Extend self.modpar_names to inlcude a 2nd row with info on which model do they belong to and a 3rd with the name of the model
        model_number = []       # This is the number of the model (depends on the order of the model list of strings)
        name_model = []         # This is the name of the current model (the number above does not correspond to the number in the parameter naming)
        current_mod_numb = 0
        for mod in self.model_names:
            # Get the number of parameters in each model as we go through them
            current_param_numb = get.get_mod_numb_params(mod)
            for a in range(current_param_numb):
                model_number.append(current_mod_numb)
                name_model.append(mod)
            current_mod_numb +=1
        # Create array of informations
        self.modpar_info = [[],[],[]]
        self.modpar_info[0] = self.modpar_names
        self.modpar_info[1] = model_number
        self.modpar_info[2] = name_model
        
        
        # Save the number of parameters
        self.k_numb_param = len(self.single_hp0)
        self.mod_numb_param = len(self.single_modpar0)
        self.tot_numb_param = len(self.single_hp0) + len(self.single_modpar0)
        
        
        # If model is keplerian, substitute eccentricity and omega with Sk and Ck
        # Ford et al. 2006
        # ecc = Sk^2 + Ck^2
        # omega = arctan(Sk/Ck)
        # Sk = sqrt(ecc) * sin(omega), Ck = sqrt(ecc) * cos(omega)
        for g in range(len(self.single_modpar0)):
            if self.modpar_info[0][g].startswith('ecc') and (self.modpar_info[2][g].startswith('kepl') or self.modpar_info[2][g].startswith('Kepl')):
                self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1] = aux.to_SkCk(self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1])
        

        # Check that you have a reasonable number of chains
        if self.numb_chains < (len(self.single_hp0)+len(self.single_modpar0))*2:
            return RuntimeWarning("It is not advisable to conduct this analysis with less chains than double the amount of free parameters")
        
        
         # Populate the positions based on those values for the starting point of all the chains
        self.hp0 = aux.initial_dist_creator(self.single_hp0, self.hp_err, self.numb_chains, self.hp_vary)
        self.modpar0 = aux.initial_dist_creator(self.single_modpar0, self.modpar_err, self.numb_chains, self.modpar_vary)
        
        # Append these first guesses (and their chains) to the storing arrays
        self.hparameter_list.append(self.hp0)
        self.model_parameter_list.append(self.modpar0)
        
        # And correct to get the proper shape
        self.hparameter_list = np.array(self.hparameter_list[0])
        self.hparameter_list.tolist()
        self.model_parameter_list = np.array(self.model_parameter_list[0])
        self.model_parameter_list.tolist()
        # Expected output: 2d array with nrows=numb_chains, ncols=number of hyperparameters
        # We will then use np.concatenate(a.b) to make it into a 3d array with ndepth=steps
        
        # Now let's get the first round of likelyhoods
        # Initialise list
        self.logL0 = []
        # Loop over all chains and get the likelyhood
        for chain in range(self.numb_chains):
            # Pick one row at a time
            hp_chain = self.hp0[chain]
            modpar_chain = self.modpar0[chain]
            
            # Generate parameter objects
            hparam_chain = Parameter.Kernel_Par_Creator(self.kernel_names)
            for i, key in zip(range(len(hp_chain)), hparam_chain.keys()):
                hparam_chain[key] = Parameter.Parameter(value=hp_chain[i], error=self.hp_err[i], vary=self.hp_vary[i])
            
            modelparam_chain = Parameter.Model_Par_Creator(self.model_names)
            for i, key in zip(range(len(modpar_chain)), modelparam_chain.keys()):
                modelparam_chain[key] = Parameter.Parameter(value=modpar_chain[i], error=self.modpar_err[i], vary=self.modpar_vary[i])

            # Compute the model (and need to go from Sk Ck to ecc and omega)
            self.model_y0 = get.get_model(self.model_names, self.x, modelparam_chain, to_ecc=True)
            
            # Get likelihood and append to initial guess array
            self.likelyhood = lk.GPLikelihood(self.x,self.y,self.model_y0, self.y_err, hparam_chain, modelparam_chain, self.kernel_names)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            self.logL0.append(logL_chain)
            
            # Clean up for next round
            hparam_chain, modelparam_chain, logL_chain = None, None, None
        
        # Save this set of likelihoods as the first ones of the overall likelihood array
        self.logL_list.append(self.logL0)
        # Do some shape rearranging
        transp1 = np.array(self.logL_list).T
        self.logL_list = transp1.tolist()
        
        # As well as the accepted step 0
        acc =  [True for i in range(self.numb_chains)]
        self.accepted.append(acc)
        transp2 = np.array(self.accepted).T
        self.accepted = transp2.tolist()


        # Verbose
        print("Initial hyper-parameter guesses: ")
        print(self.single_hp0)
        print()
        print("Initial model parameter guesses (ecc and omega are replaced by Sk and Ck): ")
        print(self.single_modpar0)
        print()
        print("Initial Log Likelihood: ", self.logL0[0])
        print()
        print("Number of chains: ", self.numb_chains)
        print()
    
    

    def split_step(self, n_splits=2, a=1.25, check_list=None):
        '''
        Parameters
        ----------
        n_splits : integer, optional
            Number of subsplits of the total number of chains. The default is 2.
        a : float, optional
            Adjustable scale parameter. The default is 1.25.
        '''

        # Split the chains into subgroups (Foreman-Mackey et al. 2013)
        all_inds = np.arange(self.numb_chains)
        # Use modulo operator
        inds = all_inds % n_splits
        # Shuffle to allow for flexible statistics
        random.shuffle(inds)
        
        # Create empty array for the new steps
        self.hp = np.zeros_like(self.hp0)
        self.modpar = np.zeros_like(self.modpar0)
        self.logz = np.zeros(self.numb_chains)
        
        # Create empty arrays for considered split part = 1, and the rest = 2
        S1_hp = []
        S2_hp = []
        
        S1_mod = []
        S2_mod = []
        
        # Repeat how many splits you want
        for split in range(n_splits):
            # Do the separation
            for walker in range(len(inds)):
                if inds[walker] == split:
                    S1_hp.append(self.hp0[walker])
                    S1_mod.append(self.modpar0[walker])
                else:
                    S2_hp.append(self.hp0[walker])
                    S2_mod.append(self.modpar0[walker])
            S1_hp = np.array(S1_hp)
            S2_hp = np.array(S2_hp)
            S1_mod = np.array(S1_mod)
            S2_mod = np.array(S2_mod)
            
            # Get the lengths of the two sets
            N_chains1 = len(S1_hp)
            N_chains2 = len(S2_hp)

            # Start looping over all chains in the set
            for chain in range(N_chains1):
                # Pick a random chain in the second set
                rint = random.randint(0, N_chains2-1)
                Xj_hp = S2_hp[rint]
                # Use same chain for model parameters as well
                Xj_mod = S2_mod[rint]
                # Compute the step and apply it
                # The random number is the same within a chain but different between chains
                z = (np.random.rand()*(a-1) + 1)**2 / a
                Xk_new = Xj_hp + (S1_hp[chain] - Xj_hp) * z
                Xk_new_mod = Xj_mod + (S1_mod[chain] - Xj_mod) * z
                
                start_check = check.Parameter_Check(Xk_new,self.kernel_names, Xk_new_mod, self.model_names, self.x)
                for i in range(len(check_list)):
                    base = check_list[i][0]
                    check_name = check_list[i][1]
                    check_params = check_list[i][2]
                    
                    
                    
                
                # check_list will be in the form of ["kernel", "no_negative", params (if needed)]
                

        




# Currently don't know how/if to split this
# should I just keep everything together?
# Don't need to add any extra stuff for multiple kernels here, can add to likelihood.py
# Maybe do a separate file for the parameter check funtion: (worth it if user can add their own criteria)
    # none less than zero
    # if keplerian, planet not into star
    # also sum of squares of Sk and CK must be less than 1
    # if multiple keplerians, orbits must not intersect and be stable

# For the computation of mass, no point in doing it in this file
# better to create a separate stuff with a function that takes the proper parameters
# and compute all the planet masses for all iterations of all chains separately and then
# get the value and the error
# OR do a straightforwad error formula

# for convergence criteria, do a fully separate file

# Final current plan
# MCMC.py with [MCMC class and run_MCMC]
# convergence tests.py with Gelman Rubin (and in future more and autocorr time calc)
# parameter checks.py with [check function ]
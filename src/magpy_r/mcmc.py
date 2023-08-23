'''
MCMC code for the gaussian model

Contains:
    MCMC class
    run_mcmc function
    

Author: Federica Rescigno, Bryce Dixon
Version 22.08.2023
'''

import random
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import magpy_r.parameters as par
import magpy_r.models as modl
import magpy_r.gp_likelihood as gp
import magpy_r.auxiliary as aux
import magpy_r.mcmc_aux as mcmcx
get_model = mcmcx.get_model
numb_param_per_model = mcmcx.numb_param_per_model
parameter_check = mcmcx.parameter_check




class MCMC:
    
    def __init__(self, t, rv, rv_err, hparam0, kernel_name, model_par0, model_name, prior_list, numb_chains=100, flags=None, Mstar = 0):
        '''
        Parameters
        ----------
        t : array, floats
            Time array of the observations
        rv : array, floats
            Radial velocity array of the observations
        rv_err : array, floats
            Errors on the radial velocity array
        hparam0 : dictionary
            Set of hyper-parameters for the kernel, with value, error and vary 
        kernel_name : string
            Name of chosen kernel
        model_par0 : dictionary
            Set of parameters for the model, with value, error and vary
        model_name : string
            Name of chosen model
        prior_list : list
            List inlcuding in order the name of parameter, the name of the prior and the dictionary of the prior parameters (see posterior function in GP_solar.py for better description)
        numb_chains : integer, optional
            Number of chains requested. The default is 100.
        Mstar : float
            mass of the host star in solar masses
        '''
        
        # Save unchanging inputs
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.kernel_name = kernel_name
        self.model_name = model_name
        self.Mstar = Mstar
        if flags is not None:
            self.flags=flags
        else:
            self.flags = None
        # Check how many models are included
        if isinstance(self.model_name, list):
            self.numb_models = len(self.model_name)
        elif isinstance(self.model_name, str):
            self.numb_models = 1
            # Put it into list for easy computation in loops
            self.model_name = [model_name]
        else:
            raise TypeError("model_name must be a string or a list of strings")
        self.prior_list = prior_list
        self.numb_chains = int(numb_chains)
        

        hlen = []
        for key in hparam0.keys():
            hlen.append(hparam0[key].value)
        self.hlen = len(hlen)
        plen = []
        for key in model_par0.keys():
            plen.append(model_par0[key].value)
        self.plen = len(plen)
        
        # Set up storing arrays
        self.hparameter_list = np.zeros(shape = (1, self.numb_chains, self.hlen)) # don't think these need to be setup here
        self.model_parameter_list = np.zeros(shape = (1, self.numb_chains, self.plen))
        self.logL_list = np.zeros(shape = (1, self.numb_chains, 1))
        self.accepted = np.zeros(shape = (1, self.numb_chains, 1))
        
        # par_list counts how many keplerians are being used which can be used as the number of columns (planets)
        par_list = []
        try:
            model_par0['P'].value
            self.mass_list = np.zeros(shape = (1, self.numb_chains, 1)) # mass array with rows = chains, columns = planets, dimensions = iterations, just one planet for this one
        except KeyError:
            for i in range(len(self.model_name)):
                try:
                    model_par0['P_'+str(i)]
                    par_list.append(i)
                except KeyError:
                    continue
            self.mass_list = np.zeros(shape = (1, self.numb_chains, len(par_list))) # mass array with rows = chains, columns = planets, dimensions = iterations
                        
                    
        # Get initial guesses for hyperparameters and save them as 0
        # Save also errors for chains initial positions population
        self.single_hp0 = []
        self.hp_err = []
        self.hp_vary = []
        for key in hparam0.keys():
            self.single_hp0.append(hparam0[key].value)
            self.hp_err.append(hparam0[key].error)
            self.hp_vary.append(hparam0[key].vary)
        # Do the same for model parameters
        self.single_modpar0 = []
        self.modpar_err = []
        self.modpar_vary = []
        self.modpar_names = []
        # Also include a list of the names of the parameters, in the proper order (useful when have multiple models)
        for key in model_par0.keys():
            self.single_modpar0.append(model_par0[key].value)
            self.modpar_err.append(model_par0[key].error)
            self.modpar_vary.append(model_par0[key].vary)
            self.modpar_names.append(key)
        
        for i in range(len(prior_list)):
            assert prior_list[i][0] in self.modpar_names or prior_list[i][0] in hparam0.keys(), "mispelt or incorrect parameter name: " + str(prior_list[i][0])
        # Extend self.modpar_names to inlcude a 2nd row with info on which model do they belong to and a 3rd with the name of the model
        which_model = []
        name = []
        b = 0
        for mod in self.model_name:
            a = numb_param_per_model(mod)
            for a in range(a):
                which_model.append(b)
                name.append(mod+str(b))
            b +=1
        self.modpar_info = [[],[],[]]
        self.modpar_info[0] = self.modpar_names
        self.modpar_info[1] = which_model
        self.modpar_info[2] = name
        
        
        # Save number of parameters
        self.k_numb_param = len(self.single_hp0)
        self.mod_numb_param = len(self.single_modpar0)
        self.numb_param = len(self.single_hp0) + len(self.single_modpar0)
        
        
        # If model is keplerian, substitute eccentricity and omega with Sk and Ck
        # Ford et al. 2006
        # ecc = Sk^2 + Ck^2
        # omega = arctan(Sk/Ck)
        # Sk = sqrt(ecc) * sin(omega), Ck = sqrt(ecc) * cos(omega)
        for g in range(len(self.single_modpar0)):
            if self.modpar_info[0][g].startswith('ecc') and (self.modpar_info[2][g].startswith('kep') or self.modpar_info[2][g].startswith('Kep')):
                self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1] = aux.to_SkCk(self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1])

        
        # Check that you have a reasonable number of chains
        if self.numb_chains < (len(self.single_hp0)+len(self.single_modpar0))*2:
            return RuntimeWarning("It is not advisable to conduct this analysis with less chains than double the amount of free parameters")
        
        
        
        # Populate the positions based on those values for the starting point of all the chains
        self.hp0 = mcmcx.initial_pos_creator(self.single_hp0,self.hp_err, self.numb_chains)
        self.modpar0 = mcmcx.initial_pos_creator(self.single_modpar0, self.modpar_err, self.numb_chains)
        
        # Append these first guesses (and their chains) to the storing arrays (check for right shape)
        self.hparameter_list[0] = self.hp0 # set first dimension in hparameter array to initial guesses
        self.model_parameter_list[0] = self.modpar0 # set first dimension in model_parameter array to initial guesses
        # Expected output: 3d array with nrows=numb_chains, ncols=number of hyperparameters, ndepth=iterations
        
        self.logL0 = np.zeros(shape = (1, self.numb_chains, 1)) # set up initial logL list
        self.mass0_list = np.zeros_like(self.mass_list) # set up initial mass list
        
        
        # Start looping over all chains to get initial models and logLs
        for chain in range(self.numb_chains):
            # Pick each row one at a time
            hp_chain = self.hparameter_list[0,chain,]
            modpar_chain = self.model_parameter_list[0,chain,]

            
            # Re-create parameter objects, kernel and model
            hparam_chain = par.par_create(self.kernel_name)
            for i, key in zip(range(len(hp_chain)), hparam_chain.keys()):
                hparam_chain[key] = par.parameter(value=hp_chain[i], error=self.hp_err[i], vary=self.hp_vary[i])
            model_par_chain = modl.mod_create(self.model_name)
            for i, key in zip(range(len(modpar_chain)), model_par_chain.keys()):
                model_par_chain[key] = par.parameter(value=modpar_chain[i], error=self.modpar_err[i], vary=self.modpar_vary[i])
            
            # try for one keplerian model, if there is not then loop through the number of columns in the initial mass list    
            try:
                mass_chain = aux.mass_calc(model_par_chain["P"].value, model_par_chain["K"].value, model_par_chain["omega"].value, model_par_chain["ecc"].value, self.Mstar)
                self.mass0_list[0, chain, 0] = mass_chain
            except KeyError:
                for i in range(len(self.mass0_list[0,0,])):
                    mass_chain = aux.mass_calc(model_par_chain["P_"+str(i)].value, model_par_chain["K_"+str(i)].value, model_par_chain["omega_"+str(i)].value, model_par_chain["ecc_"+str(i)].value, self.Mstar)
                    self.mass0_list[0, chain, i] = mass_chain
                
            
            
            # Compute model y as sum of models
            if flags is None:
                self.model_y0 = get_model(self.model_name, self.t, model_par_chain, to_ecc = True)
            if flags is not None:
                self.model_y0 = get_model(self.model_name, self.t, model_par_chain, flags=self.flags, to_ecc = True)
            
            #for priors in ecc and omega need to go back momentarely
            
            self.likelihood = gp.GPLikelihood(self.t, self.rv, self.rv_err, hparam_chain, self.kernel_name, self.model_y0, model_par_chain)
            logL_chain = self.likelihood.LogL(self.prior_list)
            
            # logL is initial likelihood of this chain, append it to logL0 of all chains
            self.logL0[0,chain,0] = logL_chain
            
            #### ATTENTION!!!! FOR SOME REASON AFTER GET_MODEL THE VALUES IN MODEL_PAR_CHAIN BECOME OMEGA AND ECCENTRICITY
            hparam_chain, model_par_chain, logL_chain, mass_chain = None, None, None, None

        
        # Save this set of likelihoods as the first ones of the overall likelihood array
        self.logL_list[0] = self.logL0 # set first dimension in logL array to the initial logL
        acc =  [True for i in range(self.numb_chains)]
        self.accepted[0,:,0] = acc # numpy array reads 1.0 as true and 0.0 as false, all initial values are accepted
        self.mass_list[0] = self.mass0_list # set first dimension in mass array to initial masses
        
        
        # Verbose
        print("Initial hyper-parameter guesses: ")
        print(self.single_hp0)
        print()
        print("Initial model parameter guesses (ecc and omega are replaced by Sk and Ck): ")
        print(self.single_modpar0)
        print()
        print("Initial Log Likelihood: ", self.logL_list[0,0,0])
        print()
        print("Number of chains: ", self.numb_chains)
        print()
        
        
    
    
    def split_step(self, n_splits=2, a=2., Rstar=None, Mstar=None):
        '''        
        Parameters
        ----------
        n_splits : integer, optional
            Number of subsplits of the total number of chains. The default is 2.
        a : float, optional
            Adjustable scale parameter. The default is 2.
        Rstar : float, optional
            Radius of the host star in solar radii. The default is None.
        Mstar : float, optional
            Mass of the star in solar masses. The default is None.
        '''
        
        # Split the chains into subgroups (Foreman-Mackey et al. 2013)
        all_inds = np.arange(self.numb_chains)
        # Use modulo operator (think clock)
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
                    S1_hp.append(self.hp0[0,walker,])
                    S1_mod.append(self.modpar0[0,walker,])
                else:
                    S2_hp.append(self.hp0[0,walker,])
                    S2_mod.append(self.modpar0[0,walker,])
            S1_hp = np.array(S1_hp)
            S2_hp = np.array(S2_hp)
            S1_mod = np.array(S1_mod)
            S2_mod = np.array(S2_mod)
            N_chains1 = len(S1_hp)
            N_chains2 = len(S2_hp)

            row = len(S1_hp)
            column = len(S1_hp[0])
            
            
            # Start looping over all chains in the set
            for chain in range(N_chains1):
                # Pick a random chain in the second set
                rint = random.randint(0, N_chains2-1)
                Xj = S2_hp[rint]
                # Use same chain for model parameters as well
                Xj_mod = S2_mod[rint]
                # Compute the step and apply it
                # The random number is the same within a chain but different between chains
                z = (np.random.rand()*(a-1) + 1)**2 / a
                Xk_new = Xj + (S1_hp[chain] - Xj) * z
                Xk_new_mod = Xj_mod + (S1_mod[chain] - Xj_mod) * z
                
                
                if Rstar is not None and Mstar is not None:
                    model_param_check = parameter_check(Xk_new_mod, self.model_name, Rstar, Mstar)
                else:
                    model_param_check = parameter_check(Xk_new_mod, self.model_name)
                    
                while (np.min(Xk_new) < 0) or (not model_param_check):
                    # Compute the step and apply it
                    # The random number is the same within a chain but different between chains
                    z = (np.random.rand()*(a-1) + 1)**2 / a
                    Xk_new = Xj + (S1_hp[chain] - Xj) * z
                    Xk_new_mod = Xj_mod + (S1_mod[chain] - Xj_mod) * z
                    
                    if Rstar is not None and Mstar is not None:
                        model_param_check = parameter_check(Xk_new_mod, self.model_name, Rstar, Mstar)
                    else:
                        model_param_check = parameter_check(Xk_new_mod, self.model_name)
                                
            
                        
                log_z = np.log(z)
                    
                # Save as the new value for the chain, by finding where the initial step was positioned and putting it in the same position
                for o in range(self.numb_chains):
                    # Kernel and model parameters are still kept together after shuffling
                    if (self.hp0[0, o, 0] == S1_hp[chain, 0]) and (self.modpar0[0, o, 0] == S1_mod[chain, 0]):
                        position = o
                
                for v in range(len(self.hp_vary)):
                    if self.hp_vary[v]:
                        self.hp[0, position, v] = Xk_new[v]
                    else:
                        self.hp[0, position, v] = S1_hp[chain, v]
                for v in range(len(self.modpar_vary)):
                    if self.modpar_vary[v]:
                        self.modpar[0, position, v] = Xk_new_mod[v]
                        #print("nope")
                    else:
                        self.modpar[0, position, v] = S1_mod[chain, v]
                self.logz[position] = log_z
                
            
            # Once all elements of the first section have taken the step, reinitialise the S1,2 arrays
            
            S1_hp = []
            S2_hp = []
            
            S1_mod = []
            S2_mod = []
            
        # Final result is self.hp which should be a 3d array, nrow=numb chains, ncol=numb parameters, dimensions=1
                
                

    
    def compute(self):
        '''
        Computes the current logL and mass
        '''
        
        # create empty arrays for current step
        self.logL = np.zeros(shape = (1, self.numb_chains, 1))
        self.mass = np.zeros_like(self.mass0_list)

        
        # Start by going chain by chain
        for chain in range(self.numb_chains):
            #print("chain", chain)
            # Reset dictionary for the kernel hyperparameters and store new values
            param = None
            param = par.par_create(self.kernel_name)
            for i, key in zip(range(self.k_numb_param), param.keys()):
                param[key] = par.parameter(value=self.hp[0, chain, i], error=self.hp_err[i], vary=self.hp_vary[i])
            
            # Do the same for the model parameters
            model_param = None
            model_param = modl.mod_create(self.model_name)
            
            
            for i, key in zip(range(self.mod_numb_param), model_param.keys()):
                model_param[key] = par.parameter(value=self.modpar[0, chain, i], error=self.modpar_err[i], vary=self.modpar_vary[i])
            
            # same method as the previous mass calculation
            try:
                mass_chain = aux.mass_calc(model_param["P"].value, model_param["K"].value, model_param["omega"].value, model_param["ecc"].value, self.Mstar)
                self.mass[0, chain, 0] = mass_chain
            except KeyError:
                for i in range(len(self.mass[0,0,])):
                    mass_chain = aux.mass_calc(model_param["P_"+str(i)].value, model_param["K_"+str(i)].value, model_param["omega_"+str(i)].value, model_param["ecc_"+str(i)].value, self.Mstar)
                    self.mass[0, chain, i] = mass_chain
            
            
            # Get new model
            if self.flags is None:
                self.model_y = get_model(self.model_name, self.t, model_param, flags=None, to_ecc = True)
            if self.flags is not None:
                self.model_y = get_model(self.model_name, self.t, model_param, flags=self.flags, to_ecc = True)
            
            #For some reason after going through get model we get the ecc and omega instead???
            
            self.likelihood = None
            # Use current hp and model to compute the logL
            self.likelihood = gp.GPLikelihood(self.t, self.rv, self.rv_err, param, self.kernel_name, self.model_y, model_param)
            logL_chain = self.likelihood.LogL(self.prior_list)
            
            self.logL[0,chain,0] = logL_chain
            
        
        # Final output: a logL 3d array, ncols = 1, nrows = numb_chains, dimensions = 1        
        

        
    def compare(self):
        """
        Compares current step with previous step, appends current or previous step to storing array based on the difference in logL 
        """
        
        # Create empty array to save decisions in to then concatenate to the list arrays
        hp_decision = np.zeros(shape = (1, self.numb_chains, self.hlen))
        modpar_decision = np.zeros(shape = (1, self.numb_chains, self.plen))
        logL_decision = np.zeros(shape = (1, self.numb_chains, 1))
        self.acceptance_chain = np.zeros(shape = (1,self.numb_chains,1))
        self.mass_decision = np.zeros_like(self.mass)
        
        
        
        # Loop over all chains
        for chain in range(self.numb_chains):
            # Compute the difference between the current and the previous likelihood (include affine invariant normalisation)
            
            diff_logL_z = self.logL[0, chain, 0] - self.logL0[0, chain, 0] #+ self.logz[chain] * (self.numb_param - 1)
            hp = self.hp
            hp0 = self.hp0
            modpar = self.modpar
            modpar0 = self.modpar0
            
            # If the logL is larger than one, then it's exponent will definitely be larger than 1 and will automatically be accepted
            if diff_logL_z > 1:
                logL_decision[0, chain, 0] = self.logL[0, chain, 0]
                hp_decision[0, chain,] = hp[0, chain,]
                modpar_decision[0, chain,] = modpar[0, chain,]
                self.acceptance_chain[0,chain,0] = True
                self.mass_decision[0, chain,] = self.mass[0, chain,]
                
            # If the logL is ver small (eg. smaller than -35), automatic refusal
            if diff_logL_z < -35.:
                logL_decision[0, chain, 0] = self.logL0[0, chain, 0]
                hp_decision[0, chain,] = hp0[0, chain,]
                modpar_decision[0, chain,] = modpar0[0, chain,]
                self.acceptance_chain[0,chain,0] = False
                self.mass_decision[0, chain,] = self.mass0_list[0, chain,]
            
            if (diff_logL_z >= -35.) and (diff_logL_z <= 1.):
                # Generate random number from uniform distribution
                MH_rand = random.uniform(0,1)
                # if diff_Lz is larger than the number, accept the step
                if MH_rand <= (np.exp(diff_logL_z)):
                    logL_decision[0, chain, 0] = self.logL[0, chain, 0]
                    hp_decision[0, chain,] = hp[0, chain,]
                    modpar_decision[0, chain,] = modpar[0, chain,]
                    self.acceptance_chain[0,chain,0] = True
                    self.mass_decision[0, chain,] = self.mass[0, chain,]
                # if it is smaller than the number reject the step
                else:
                    logL_decision[0, chain, 0] = self.logL0[0,chain,0]
                    hp_decision[0, chain,] = hp0[0,chain,]
                    modpar_decision[0, chain,] = modpar0[0,chain,]
                    self.acceptance_chain[0,chain,0] = False
                    self.mass_decision[0, chain,] = self.mass0_list[0, chain,]
            
    
            

        
        # Now concatenate the storing arrays with the decision arrays to add the current step as a new dimension
        self.logL_list = np.concatenate((self.logL_list, logL_decision)) # nrows = chains, ndimensions = iterations
        self.mass_list = np.concatenate((self.mass_list, self.mass_decision)) # nrows = chains, ncolumns = planets, ndimensions = iterations
        self.accepted = np.concatenate((self.accepted, self.acceptance_chain)) # nrows = chains, ndimensions = iterations
        self.hparameter_list = np.concatenate((self.hparameter_list, hp_decision)) # nrows = chains, ncolumns = parameters, ndimensions = iterations
        self.model_parameter_list = np.concatenate((self.model_parameter_list, modpar_decision)) # nrows = chains, ncolumns = parameters, ndimensions = iterations
        logL_decision, hp_decision, modpar_decision = None, None, None        
        

    
    
    def reset(self):
        '''
        Returns
        -------
        logL_list : 3D array
            List of the logL of all accepted steps, nrow = n_chains, ncol = 1, ndep = iterations
        hparameters_list : 3D array
            List of the hyperparameters of all accepted steps, nrow = n_chains, ncol = n_parameters, ndep = iterations
        modelel_paramater_list : 3D array
            List of the model parameters of all accepted steps, nrow = n_chains, ncol = n_parameters, ndep = iterations
        accepted : 3D array
            List of True and Falses regarding the acceptance of each MCMC step, nrow = n_chains, ncol = 1, ndep = iterations
        '''
        # Don't need burn in
        
        # Set the zero values for nex step
        for chain in range(self.numb_chains):
            if self.acceptance_chain[0, chain, 0] == True:
                # if the chain was accepted, set the new initial value to the accepted one
                self.logL0[0, chain, 0] = self.logL[0, chain, 0]
                self.hp0[0, chain,] = self.hp[0, chain,]
                self.modpar0[0, chain,] = self.modpar[0, chain,]
                self.mass0_list[0, chain,] = self.mass[0, chain,]
            # if the chain was rejected, leave the initial value as the current initial value
        
        # IMPORTANT!! In model_parameter_list, if the model is a keplerian we have Sk and Ck, not ecc and omega
        
        return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted, self.mass_list
    
    
    
    def gelman_rubin_calc(self, burn_in):
        """
        Computes the Gelman-Rubin convergence statistic.
        Must be calculated for each parameter independently.
        
        Parameters
        -------
        burn_in : int
            length in steps of the chosen burn in phase
        
        Returns
        -------
        all_R : array, floats
            array of Gelman-Rubin values for each parameter
        
        """

        all_R = []
        
        try:
            J = np.shape(self.hparameter_list)[0]
            P = np.shape(self.hparameter_list)[1]
            L = np.shape(self.hparameter_list)[2] - burn_in
        except:
            import pdb

            pdb.set_trace()
        
        #  Calculate for hyperparams
        hp=True
        if hp:
            for hyper_param in range(P):
                chain_means = []
                intra_chain_vars = []
                for chain in range(J):
                    # Calculate chain mean
                    param_chain = self.hparameter_list[burn_in:, chain, hyper_param]

                    chain_means.append(np.nanmean(param_chain))
                    intra_chain_var = np.nanvar(param_chain, ddof=1)
                    intra_chain_vars.append(intra_chain_var)
                chain_means = np.array(chain_means)
                grand_mean = np.mean(chain_means)
                intra_chain_vars = np.array(intra_chain_vars)
                inter_chain_var = L / (J - 1) * np.sum(np.square(chain_means - grand_mean))
                W = np.mean(intra_chain_vars)

                R = (1 - 1 / L) * W + inter_chain_var / L
                R /= W
                all_R.append(R)
        
        # Redefine for model params - Others should be unchanged
        P = np.shape(self.model_parameter_list)[1]

        #  Calculate for model_params
        for param in range(P):
            chain_means = []
            intra_chain_vars = []
            if (
                np.nanmax(self.model_parameter_list[:, param, :])
                - np.nanmin(self.model_parameter_list[:, param, :])
                == 0.0
            ):
                all_R.append(1.0)
                continue
            for chain in range(J):
                # Calculate chain mean
                param_chain = self.model_parameter_list[burn_in:, chain, param]

                chain_means.append(np.nanmean(param_chain))
                intra_chain_var = np.nanvar(param_chain, ddof=1)
                intra_chain_vars.append(intra_chain_var)
            chain_means = np.array(chain_means)
            grand_mean = np.mean(chain_means)
            intra_chain_vars = np.array(intra_chain_vars)
            inter_chain_var = L / (J - 1) * np.sum(np.square(chain_means - grand_mean))
            W = np.mean(intra_chain_vars)

            R = (1 - 1 / L) * W + inter_chain_var / L
            R /= W

            all_R.append(R)
        
        all_R = np.array(all_R)
        try:
            assert np.all(all_R >= 1.0)
        except:
            import pdb

            pdb.set_trace()
        
        return all_R
    
    
    
    









def run_MCMC(iterations, t, rv, rv_err, hparam0, kernel_name, model_param0 = None, model_name = ["no_model"], prior_list = [], numb_chains=None, n_splits=None, a=None, Rstar=None, Mstar=None, flags=None, plot_convergence=False, saving_folder=None, gelman_rubin_limit = 1.1):
    """Function to run the MCMC to obtain posterior distributions for hyperparameters and model parameters

    Parameters
    ----------
    iterations : int
        number of iterations to run the MCMC for
    t : array of floats
        array of the time data
    rv : array of floats
        array of the rv data
    rv_err : array of floats
        array of the rv errors
    hparam0 : dictionary
        dictionary of all hyperparameters
    kernel_name : string
        name of the chosen Kernel
    model_param0 : dictionary, optional
        dictionary of all model parameters, not required for no model. Defaults to None
    model_name : list of strings, optional
        list of the names of the chosen models, not required for no model. Defaults to ["no_model"]
    prior_list : list of dictionaries, optional
        list of prior parameters in the form of dictionaries set up by pri_create, not required for no priors. Defaults to []
    numb_chains : int
        number of MCMC chains to run, no input will run 100 chains. Defaults to None
    n_splits : int, optional
        number of subsplits of the total number of chains, no input will use 2 splits. Defaults to None
    a : float, optional
        adjustable scale parameter, no input will use a=2. Defaults to None
    Rstar : float, optional
        radius of the host star in solar radii. Defaults to None
    Mstar : float, optional
        rmass of the star in solar masses. Defaults to None
    flags: array of floats, optional
        array of flags representing which datapoints in the time array are related to which telescope and so will have which offset. Defaults to false
    saving_folder : string, optional
        folder location to save outputs to. Defaults to None
    gelman_rubin_limit : float, optional
        Convergence cut for the Gelman_Rubin statistic. Default is 1.1

    Raises
    ------
    KeyError
        Raised if a model is in use and no model parameters have been provided
    Assertion
        Raised if the number of chains is less than double the number of free parameters

    Returns:
    logL_list : array of floats
        array of the log likelihoods of every iteration
    hparameter_list : array of floats
        array of the hyperparameters of every iteration
    model_parameter_list : array of floats
        array of the model parameters of every iteration
    mass : array of floats
        array of calculated masses from the model for each iteraiton, where the columns represent the different planets
    completed_iterations : integer
        number of completed iterations
    """
    
    if model_param0 == None:
        if model_name[0].startswith("no") or model_name[0].startswith("No"):
            model_param0 = modl.mod_create(["no_model"])
            model_param0["no"] = par.parameter(value=0., error=0., vary=False)
        else:
            raise KeyError("model parameters and model list must be provided when using a model")
    
    if numb_chains is not None:
        assert numb_chains > (len(model_param0)+len(hparam0)) * 2, "number of chains should be larger than the number of free parameters"
    
    if Mstar is None:
        if flags is None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains)
        if flags is not None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags)
    else:
        if flags is None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, Mstar = Mstar)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, Mstar = Mstar)
        if flags is not None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, Mstar = Mstar)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, Mstar = Mstar)
    
    # Initialise progress bar
    print("Start Iterations")
    print()
    
    start = time.time()
    burn_in = min(200, iterations // 10)
    
    
    if plot_convergence:
        conv_f, ax = plt.subplots(figsize=(15,15))
        hparam_names = aux.hparam_names(kernel_name)
        model_param_names = aux.model_param_names(model_name)
        all_param_names = hparam_names + model_param_names
        conv_vals = {i: [] for i in all_param_names}
        conv_iters = []
        ax.set_prop_cycle("color", sns.color_palette("plasma",len(all_param_names)))

        if saving_folder is None:
            saving_folder = os.getcwd()
    
    
    for iteration in range(iterations):
        if n_splits is None and a is None:
            _.split_step()
        elif n_splits is not None and a is None:
            _.split_step(n_splits=n_splits)
        elif n_splits is None and a is not None:
            _.split_step(a=a)
        elif n_splits is not None and a is not None:
            _.split_step(n_splits=n_splits, a=a)
        
        _.compute()
        
        if Rstar is not None and Mstar is not None:
            _.compare(Rstar=Rstar, Mstar=Mstar)
        elif Rstar is None or Mstar is None:
            _.compare()
        
        logL_list, hparameter_list, model_parameter_list, accepted, mass = _.reset()
        if (iteration % 2==0) or iteration == iterations-1:
            aux.printProgressBar(iteration, iterations-1, length=50)

        if plot_convergence:
            if (iteration >= burn_in) and (iteration - burn_in) % 20 == 0:
                # Check whether the chains have converged...
                R_list = _.gelman_rubin_calc(burn_in)
                if plot_convergence:
                    assert len(R_list) == len(conv_vals)
                    for i, R in enumerate(R_list):
                        conv_vals[all_param_names[i]].append(R)
                    conv_iters.append(iteration)


                if np.all(R_list < gelman_rubin_limit):
                    # Convergence reached...
                    completed_iterations = iteration + 1
                    print("CONVERGENCE REACHED!! at iteration {}".format(iteration))
                    break
    else:
        completed_iterations = iterations
                    

    if plot_convergence:
        for param in conv_vals:
            ax.plot(conv_iters, conv_vals[param], label=param)
        ax.legend()
        ax.axhline(gelman_rubin_limit, c="k", ls="--")
        ax.set_xlim(left=burn_in)
        ax.set_ylim(bottom=1.0)
        ax.set_yscale("log")
        ax.set_ylim(0,10)
        conv_f.savefig(str(saving_folder) + "/convergence_plot.png")
    
    print()
    print()
    if numb_chains is not None:
        print("{} iterations have been completed with {} contemporaneous chains".format(iterations, numb_chains))
    if numb_chains is None:
        print("{} iterations have been completed with {} contemporaneous chains".format(iterations, 100))
    print()
    
    # Compute the acceptance rate
    passed = 0
    rejected = 0
    accepted_flat = accepted.flatten()
    for m in range(len(accepted_flat)):
        if accepted_flat[m]:
            passed += 1
        if not accepted_flat[m]:
            rejected += 1
    
    ratio = passed/(iterations*numb_chains + numb_chains)
    print("Acceptance Rate = ", ratio)
    
    print(" ---- %s minutes ----" % ((time.time() - start)/60))
    
    # Mixing plots
    if Mstar is None:
        return logL_list, hparameter_list, model_parameter_list, completed_iterations
    else:
        return logL_list, hparameter_list, model_parameter_list, completed_iterations, mass
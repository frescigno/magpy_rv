'''
MCMC affine Base Code for multiple models
Made specifically for GP_solar_multi.py
Contains:
    Body of MCMC
    #Planetary Mass Calculator
    Running function for MCMC

Author: Federica Rescigno
Version: 20-04-2022'''

import numpy as np
import random
import GP_solar_multi as gp
import math
import plotting as plot
import auxiliary as aux
import time



def numb_param_per_model(model_name):
    ''' Function to give the number of expected parameters per model
    '''
    if model_name.startswith("no") or model_name.startswith("No"):
        model_param_number = 1
    if model_name.startswith("off") or model_name.startswith("Off"):
        model_param_number = 1
    if model_name.startswith("kepl") or model_name.startswith("Kepl"):
        model_param_number = 5
    return model_param_number




def get_model(model_name, time, model_par, to_ecc=True):
    '''
    Parameters
    ----------
    model_name : list of strings
        Name of model used
    time : array, floats
        Time array over which to calculate the model
    model_par : dictionary
        Set of parameters (within the parameter object) with which to compute the model

    Returns
    -------
    model_y : array, floats
        Radial velocity of the model
    '''
    
    
    model_y = np.zeros(len(time))
    i=0
    a=0
    for mod in model_name:
        numb_param_mod = numb_param_per_model(mod)
        parameters ={key: value for key, value in model_par.items() if (list(model_par).index(key) >= i and list(model_par).index(key) < i+numb_param_mod)}
        if mod.startswith("no") or mod.startswith("No"):
            model = gp.No_Model(time, parameters)
            model_y += model.model()
        if mod.startswith("off") or mod.startswith("Off"):
            model = gp.Offset(time, parameters)
            model_y += model.model()
        if mod.startswith("kepl") or mod.startswith("Kepl"):
            if to_ecc:
                if len(model_name) == 1:
                    parameters['ecc'].value, parameters['omega'].value = aux.to_ecc(parameters['ecc'].value, parameters['omega'].value)
                else:
                    parameters['ecc_'+str(a)].value, parameters['omega_'+str(a)].value = aux.to_ecc(parameters['ecc_'+str(a)].value, parameters['omega_'+str(a)].value)
            #print('check', parameters)
            model = gp.Keplerian(time, parameters)
            model_y += model.model()
            a +=1
        i += numb_param_mod
        
    
    return model_y




def parameter_check(parameters, names, Rstar=None, Mstar=None):
    ''' Function to check if the parameters are within the bounds
    
    Parameters
    ----------
    parameters : array, floats
        Array of parameters for all models
    names : list of strings
        Names of all the models used, can be one or more
    Rstar : float, optional
        Radius of the star in solar radii. Default is None. Needed for the orbit check
    Mstar : float, optional
        Mass of the star in solar masses. Default is None. Needed for the orbit check

    Returns
    -------
    check : bool
        Are all paramaters physically possible?
    '''
    check = True
    o = 0
    for name in names:
        numb_params = numb_param_per_model(name)
        if name.startswith('off') or name.startswith('Off'):
            if parameters[o]< 0.:
                check = False
                return check
            
        if name.startswith('kepl') or name.startswith('Kepl'):
            # Period, amplitude and t0 must be positive
            if parameters[o]< 0. or parameters[o+1]<0. or parameters[o+4]<0.:
                check = False
                return check
            
            # Sk and Ck can be negative, but the sum of their squares must be less than 1
            if (parameters[o+2]**2 + parameters[o+3]**2) > 1.:
                check = False
                return check
            
            if Rstar is not None and Mstar is not None:
                # Check planet does not fall into the star
                orbit_check = aux.orbit_check(parameters[o+2], parameters[o+3], Rstar, parameters[o], Mstar)
                if not orbit_check:
                    check = False
                    return check
            
        o += numb_params
    
    return check
            





class MCMC:
    
    def __init__(self, t, rv, rv_err, hparam0, kernel_name, model_par0, model_name, prior_list, numb_chains=100):
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
        '''
        
        # Save unchanging inputs
        self.t = t
        self.rv = rv
        self.rv_err = rv_err
        self.kernel_name = kernel_name
        self.model_name = model_name
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
        
        # Set up storing arrays
        self.hparameter_list = []
        self.model_parameter_list = []
        self.logL_list = []
        self.accepted = []
        
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
            if self.modpar_info[0][g].startswith('ecc') and (self.modpar_info[2][g].startswith('kepl') or self.modpar_info[2][g].startswith('Kepl')):
                self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1] = aux.to_SkCk(self.single_modpar0[g], self.single_modpar0[g+1], self.modpar_err[g], self.modpar_err[g+1])
        
        # Check that you have a reasonable number of chains
        if self.numb_chains < (len(self.single_hp0)+len(self.single_modpar0))*2:
            return RuntimeWarning("It is not advisable to conduct this analysis with less chains than double the amount of free parameters")
        
        
        
        # Populate the positions based on those values for the starting point of all the chains
        self.hp0 = aux.initial_pos_creator(self.single_hp0,self.hp_err, self.numb_chains)
        self.modpar0 = aux.initial_pos_creator(self.single_modpar0, self.modpar_err, self.numb_chains)
        
        # Append these first guesses (and their chains) to the storing arrays (check for right shape)
        self.hparameter_list.append(self.hp0)
        self.model_parameter_list.append(self.modpar0)
        self.hparameter_list = np.array(self.hparameter_list[0])
        self.hparameter_list.tolist()
        self.model_parameter_list = np.array(self.model_parameter_list[0])
        self.model_parameter_list.tolist()
        #print("check shape", self.hparameter_list)
        row = len(self.hparameter_list)
        column = len(self.hparameter_list[0])
        row = len(self.model_parameter_list)
        column = len(self.model_parameter_list[0])
        # Expected output: 2d array with nrows=numb_chains, ncols=number of hyperparameters
        # We will then use np.concatenate(a.b) to make it into a 3d array with ndepth=steps
        
        self.logL0 = []
        
        
        # Start looping over all chains to get initial models and logLs
        for chain in range(self.numb_chains):
            # Pick each row one at a time
            hp_chain = self.hp0[chain]
            modpar_chain = self.modpar0[chain]

            
            # Re-create parameter objects, kernel and model
            hparam_chain = gp.Par_Creator.create(self.kernel_name)
            for i, key in zip(range(len(hp_chain)), hparam_chain.keys()):
                hparam_chain[key] = gp.Parameter(value=hp_chain[i], error=self.hp_err[i], vary=self.hp_vary[i])
            Model_Par_Creator = gp.Model_Par_Creator()
            model_par_chain = Model_Par_Creator.create(self.model_name)
            for i, key in zip(range(len(modpar_chain)), model_par_chain.keys()):
                model_par_chain[key] = gp.Parameter(value=modpar_chain[i], error=self.modpar_err[i], vary=self.modpar_vary[i])
            
            # Compute model y as sum of models
            self.model_y0 = get_model(self.model_name, self.t, model_par_chain)
            self.likelyhood = gp.GPLikelyhood(self.t,self.rv,self.model_y0, self.rv_err, hparam_chain, model_par_chain, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            
            # logL is initial likelihood of this chain, append it to logL0 of all chains
            self.logL0.append(logL_chain)
            
            hparam_chain, model_par_chain, logL_chain = None, None, None

        
        # Save this set of likelihoods as the first ones of the overall likelihood array
        self.logL_list.append(self.logL0)
        acc =  [True for i in range(self.numb_chains)]
        self.accepted.append(acc)
        self.logL_list = aux.transpose(self.logL_list)
        self.accepted = aux.transpose(self.accepted)
        
        
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
        
        
    
    
    def split_step(self, n_splits=2, a=1.25, Rstar=None, Mstar=None):
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
                    S1_hp.append(self.hp0[walker])
                    S1_mod.append(self.modpar0[walker])
                else:
                    S2_hp.append(self.hp0[walker])
                    S2_mod.append(self.modpar0[walker])
            S1_hp = np.array(S1_hp)
            S2_hp = np.array(S2_hp)
            S1_mod = np.array(S1_mod)
            S2_mod = np.array(S2_mod)
            N_chains1 = len(S1_hp)
            N_chains2 = len(S2_hp)

            row = len(S1_hp)
            column = len(S1_hp[0])
            #print("row = {}, column = {}".format(row, column))
            
            
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

                #print("new", Xk_new)
                
                
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
                    if (self.hp0[o][0] == S1_hp[chain][0]) and (self.modpar0[o][0] == S1_mod[chain][0]):
                        position = o
                
                #print(position)
                
                for v in range(len(self.hp_vary)):
                    if self.hp_vary[v]:
                        self.hp[position][v] = Xk_new[v]
                    else:
                        self.hp[position][v] = S1_hp[chain][v]
                for v in range(len(self.modpar_vary)):
                    if self.modpar_vary[v]:
                        self.modpar[position][v] = Xk_new_mod[v]
                        #print("nope")
                    else:
                        self.modpar[position][v] = S1_mod[chain][v]
                self.logz[position] = log_z
                
            
            # Once all elements of the first section have taken the step, reinitialise the S1,2 arrays
            #print(len(S1_mod), len(S2_mod))
            
            S1_hp = []
            S2_hp = []
            
            S1_mod = []
            S2_mod = []
            
            #print("hp", self.hp)
        
        #print(self.modpar)
        #print()
        #print()
        # Final result is self.hp which should be a 2d array, nrow=numb chains, ncol=numb parameters
        #print("full z", self.logz)
                
                

    
    def compute(self):
        '''

        '''
        self.logL = []
        #print("hp", self.hp)
        #print("select", self.hp[0][1])

        
        # Start by going chain by chain
        for chain in range(self.numb_chains):
            #print("chain", chain)
            # Reset dictionary for the kernel hyperparameters and store new values
            param = None
            param = gp.Par_Creator.create(self.kernel_name)
            for i, key in zip(range(self.k_numb_param), param.keys()):
                param[key] = gp.Parameter(value=self.hp[chain][i], error=self.hp_err[i], vary=self.hp_vary[i])
            
            # Do the same for the model parameters
            model_param = None
            Model_Par_Creator = gp.Model_Par_Creator()
            model_param = Model_Par_Creator.create(self.model_name)
            
            
            for i, key in zip(range(self.mod_numb_param), model_param.keys()):
                model_param[key] = gp.Parameter(value=self.modpar[chain][i], error=self.modpar_err[i], vary=self.modpar_vary[i])
            
            
            #print("param", param)
            #print("model", model_param)
            
            '''self.model_y0 = get_model(self.model_name, self.t, model_par_chain)
            self.likelyhood = gp.GPLikelyhood(self.t,self.rv,self.model_y0, self.rv_err, hparam_chain, model_par_chain, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            '''
            
            #print("par model", model_param)
            #print("hyperpar", param)
            
            #self.model_y = None
            # Get new model
            self.model_y = get_model(self.model_name, self.t, model_param)
            #print("model", self.model_y[0:5])
            
            self.likelyhood = None
            #logL_chain = None
            # Use current hp and model to compute the logL
            self.likelyhood = gp.GPLikelyhood(self.t, self.rv, self.model_y, self.rv_err, param, model_param, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            #print("h", logL_chain)
            
            self.logL.append(logL_chain)
        
        # Final output: a logL 2d array, ncols = 1, nrows = numb_chains        
        

        
    def compare(self):

        
        # Create empty array to save decisions in to then concatenate to the list arrays
        hp_decision = []
        modpar_decision = []
        logL_decision = []
        self.acceptance_chain = []
        
        
        # Loop over all chains
        for chain in range(self.numb_chains):
            #print(chain)
            #print("z", self.logz[chain])
            #print("logL", self.logL[chain])
            #print("logL0", self.logL0[chain])
            # Compute the difference between the current and the previous likelyhood (include affine invariant normalisation)
            
            diff_logL_z = self.logL[chain] - self.logL0[chain] + self.logz[chain] * (self.numb_param - 1)
            #print(self.logL[chain])
            #diff_Lz = (np.exp(self.logz[chain]))**(self.numb_param - 1) * (np.exp(self.logL[chain])/np.exp(self.logL0[chain]))
            #print("diff", diff_Lz)
            
            hp = self.hp
            hp0 = self.hp0
            modpar = self.modpar
            modpar0 = self.modpar0
            
            # If the logL is larger than one, then it's exponent will definitely be larger than 1 and will automatically be accepted
            if diff_logL_z > 1:
                logL_decision.append(self.logL[chain])
                hp_decision.append(hp[chain])
                modpar_decision.append(modpar[chain])
                self.acceptance_chain.append(True)
            # If the logL is ver small (eg. smaller than -35), automatic refusal
            if diff_logL_z < -35.:
                logL_decision.append(self.logL0[chain])
                hp_decision.append(hp0[chain])
                modpar_decision.append(modpar0[chain])
                self.acceptance_chain.append(False)
            if (diff_logL_z >= -35.) and (diff_logL_z <= 1.):
                # Generate random number from uniform distribution
                MH_rand = random.uniform(0,1)
                # if diff_Lz is smaller than the number, accept the step
                if MH_rand <= (np.exp(diff_logL_z)):
                    logL_decision.append(self.logL[chain])
                    hp_decision.append(hp[chain])
                    modpar_decision.append(modpar[chain])
                    self.acceptance_chain.append(True)
                # if it is larger than the number reject the step
                else:
                    logL_decision.append(self.logL0[chain])
                    hp_decision.append(hp0[chain])
                    modpar_decision.append(modpar0[chain])
                    self.acceptance_chain.append(False)
    
            

        
        # Now concatenate all the 2D arrays into the 3D list arrays
        # Start with logL list and append, nrows = nchains, ncols = niterations
        self.logL_list = np.column_stack((self.logL_list, logL_decision))
        self.accepted = np.column_stack((self.accepted, self.acceptance_chain))
        # Rest of lists, nrows = nchains, ncols = nparam, ndepth = niterations
        self.hparameter_list = np.dstack((self.hparameter_list, hp_decision))
        self.model_parameter_list = np.dstack((self.model_parameter_list, modpar_decision))
        #print("before", hp_decision)
        #print("after", self.hparameter_list)
        
        

    
    
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
            if self.acceptance_chain[chain] is True:
                self.logL0[chain] = self.logL[chain]
                self.hp0[chain] = self.hp[chain]
                self.modpar0[chain] = self.modpar[chain]
            if self.acceptance_chain[chain] is False:
                self.logL0[chain] = self.logL0[chain]
                self.hp0[chain] = self.hp0[chain]
                self.modpar0[chain] = self.modpar0[chain]
        
        # IMPORTANT!! In model_parameter_list, if the model is a keplerian we have Sk and Ck, not ecc and omega
        
        return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted
    









def run_MCMC(iterations, t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains=None, n_splits=None, a=None, Rstar=None, Mstar=None):
    
    from MCMC_affine_multi import MCMC
    

    if numb_chains is None:
        _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list)
        numb_chains=100
    else:
        _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains)
    
    
    # Initialise progress bar
    print("Start Iterations")
    print()
    
    start = time.time()
    
    #aux.printProgressBar(0, iterations, length = 50)
    
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
        
        
        logL_list, hparameter_list, model_parameter_list, accepted = _.reset()
        if (iteration % 2==0) or iteration == iterations-1:
            aux.printProgressBar(iteration, iterations-1, length=50)
             
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
        #print(m)
        if accepted_flat[m]:
            passed += 1
        if not accepted_flat[m]:
            rejected += 1
    
    print(passed, rejected)
    ratio = passed/(iterations*numb_chains + numb_chains)
    print("Acceptance Rate = ", ratio)
    
    print(" ---- %s minutes ----" % ((time.time() - start)/60))
    
    # Mixing plots
    #plot.mixing_plot(iterations+4, hparam_chain, hparam_names, model_param_chain, model_param_names, LogL_chain)
    
    return logL_list, hparameter_list, model_parameter_list


    

        
        

    


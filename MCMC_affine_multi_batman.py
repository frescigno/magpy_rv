'''
MCMC affine Base Code for multiple models
Made specifically for GP_solar_multi.py
Contains:
    Body of MCMC
    #Planetary Mass Calculator
    Running function for MCMC

Author: Federica Rescigno
Version: 13-07-2023

Adding batman for simulataneous photometric analysis'''

import numpy as np
import random
import GP_solar_multi_batman as gp
import math
import plotting_batman as plot
import auxiliary_batman as aux
import time
import mass_calc as mc

import batman


def numb_param_per_model(model_name):
    ''' Function to give the number of expected parameters per model
    '''
    if model_name.startswith("no") or model_name.startswith("No"):
        model_param_number = 1
    if model_name.startswith("off") or model_name.startswith("Off"):
        model_param_number = 1
    if model_name.startswith("kepl") or model_name.startswith("Kepl"):
        model_param_number = 5
    if model_name.startswith("poly") or model_name.startswith("Poly"):
        model_param_number = 4
    if model_name.startswith("bat") or model_name.startswith("Bat"):
        model_param_number =  17
    return model_param_number




def get_model(model_name, time, model_par, to_ecc=True, flags=None):
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
            model = gp.Offset(flags, parameters)
            model_y += model.model()
        if mod.startswith("poly") or mod.startswith("Poly"):
            model = gp.Offset(time, parameters)
            model_y += model.model()
        if mod.startswith("bat") or mod.startswith("Bat"):
            model = gp.Batman2(time, parameters)
            model_y = model.model()
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
        
    parameters=None
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
            
        if name.startswith('kepl') or name.startswith('Kepl'):
            # Period, amplitude and t0 must be positive
            if parameters[o]< 0. or parameters[o+1]<0. or parameters[o+4]<0.:
                check = False
                return check
            
            # Sk and Ck can be negative, but the sum of their squares must be less than 1
            if ((parameters[o+2]**2 + parameters[o+3]**2) > 1.): # or (parameters[o+2] < 0.) or (parameters[o+3] < 0.):
                check = False
                return check
            
            if Rstar is not None and Mstar is not None:
                # Check planet does not fall into the star
                ####### NOW COLLED STAR CROSS!!!!!
                orbit_check = aux.orbit_check(parameters[o+2], parameters[o+3], Rstar, parameters[o], Mstar)
                if not orbit_check:
                    check = False
                    return check
            
        o += numb_params
    
    return check


def parameter_check_phot(parameters):
    check = True
    if parameters[0]<0 or parameters[3]<0 or parameters[4]<0 or parameters[5]<0 or parameters[6]<0 or parameters[8]<0 or parameters[10]<0 or parameters[11]<0 or parameters[12]<0 or parameters[14]<0:
        check = False
        return check
    if parameters[6]>1 or parameters[12]>1 or parameters[8]>1 or parameters[14]>1 or np.deg2rad(parameters[9])>(np.pi/2) or np.deg2rad(parameters[15])>(np.pi/2):
        check = False
        return check
    '''b0 =  parameters[4]*  np.cos(np.deg2rad(parameters[9]) * (1-parameters[6]**2) / (1+parameters[6]*np.sin(parameters[7])))
    if b0>(1+parameters[8]):
        check=False
        print("in here3")
        return check'''
    return check
            





class MCMC:
    
    def __init__(self, t, rv, rv_err, hparam0, kernel_name, model_par0, model_name, prior_list, numb_chains=100, flags=None, mass=False, x_phot=None, y_phot=None, yerr_phot=None, model_y_phot=None, model_param_phot=None):
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
        
        
        if x_phot is not None or (y_phot is not None) or (model_y_phot is not None) or (model_param_phot is not None) or (yerr_phot is not None):
            assert x_phot is not None and (y_phot is not None) and (model_y_phot is not None) and (model_param_phot is not None) and (yerr_phot is not None), "I need all info for batman"
        self.x_phot = x_phot
        self.y_phot = y_phot
        self.yerr_phot = yerr_phot
        self.model_y_phot = model_y_phot
        self.model_param_phot = model_param_phot
        
        
        # Set up storing arrays
        self.hparameter_list = []
        self.model_parameter_list = []
        self.batman_model_parameter_list = []
        self.logL_list = []
        self.accepted = []
        self.mass = mass
        if self.mass:
            self.mass0_list = []
            self.mass1_list = []
        
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
        
        # Do the same for batman model
        if self.x_phot is not None:
            self.single_batman_modpar0 = []
            self.batman_modpar_err = []
            self.batman_modpar_vary = []
            self.batman_modpar_names = []
            for key in model_param_phot.keys():
                self.single_batman_modpar0.append(model_param_phot[key].value)
                self.batman_modpar_err.append(model_param_phot[key].error)
                self.batman_modpar_vary.append(model_param_phot[key].vary)
                self.batman_modpar_names.append(key)
        
        
        # Save number of parameters
        self.k_numb_param = len(self.single_hp0)
        self.mod_numb_param = len(self.single_modpar0)
        self.numb_param = len(self.single_hp0) + len(self.single_modpar0)
        if x_phot is not None:
            self.batman_numb_param = len(self.single_batman_modpar0)
        
        
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
        if x_phot is not None:
            if self.numb_chains < (self.k_numb_param + self.mod_numb_param +self.batman_numb_param):
                return RuntimeWarning("It is not advisable to conduct this analysis with less chains than double the amount of free parameters")
        
        
        # Populate the positions based on those values for the starting point of all the chains
        self.hp0 = aux.initial_pos_creator(self.single_hp0,self.hp_err, self.numb_chains)
        self.modpar0 = aux.initial_pos_creator(self.single_modpar0, self.modpar_err, self.numb_chains) #, param_names=self.modpar_info[0])
        if x_phot is not None:
            self.batman_modpar0 = aux.initial_pos_creator(self.single_batman_modpar0, self.batman_modpar_err, self.numb_chains)
        
        # Append these first guesses (and their chains) to the storing arrays (check for right shape)
        self.hparameter_list.append(self.hp0)
        self.model_parameter_list.append(self.modpar0)
        self.hparameter_list = np.array(self.hparameter_list[0])
        self.hparameter_list.tolist()
        self.model_parameter_list = np.array(self.model_parameter_list[0])
        self.model_parameter_list.tolist()
        if x_phot is not None:
            self.batman_model_parameter_list.append(self.batman_modpar0)
            self.batman_model_parameter_list = np.array(self.batman_model_parameter_list[0])
            self.batman_model_parameter_list.tolist()
        #print("check shape", self.hparameter_list)
        row = len(self.hparameter_list)
        column = len(self.hparameter_list[0])
        row = len(self.model_parameter_list)
        column = len(self.model_parameter_list[0])
        # Expected output: 2d array with nrows=numb_chains, ncols=number of hyperparameters
        # We will then use np.concatenate(a.b) to make it into a 3d array with ndepth=steps
        
        self.logL0 = []
        if self.mass:
            self.mass0_0 = []
            self.mass1_0 = []
        
        
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
            
            if x_phot is not None:
                batman_modpar_chain = self.batman_modpar0[chain]
                batman_model_par_chain = Model_Par_Creator.create(["Batman2"])
                for i, key in zip(range(len(batman_modpar_chain)), batman_model_par_chain.keys()):
                    batman_model_par_chain[key] = gp.Parameter(value=batman_modpar_chain[i], error=self.batman_modpar_err[i], vary=self.batman_modpar_vary[i])
                
                
            if self.mass:
                mass0_chain = mc.mass_calc(model_par_chain["P_0"].value, model_par_chain["K_0"].value, model_par_chain["omega_0"].value, model_par_chain["ecc_0"].value, 0.743)
                mass1_chain = mc.mass_calc(model_par_chain["P_1"].value, model_par_chain["K_1"].value, model_par_chain["omega_1"].value, model_par_chain["ecc_1"].value, 0.743)
                #print("check", mass1_chain)
                #print(model_par_chain["P_1"].value, model_par_chain["K_1"].value, model_par_chain["omega_1"].value, model_par_chain["ecc_1"].value)
                self.mass0_0.append(mass0_chain)
                self.mass1_0.append(mass1_chain)
            
            
            # Compute model y as sum of models
            if flags is None:
                self.model_y0 = get_model(self.model_name, self.t, model_par_chain)
            if flags is not None:
                self.model_y0 = get_model(self.model_name, self.t, model_par_chain, flags=self.flags)
            if x_phot is not None:
                self.model_y0_batman = get_model(["Batman2"], self.x_phot, batman_model_par_chain)
            
            #for priors in ecc and omega need to go back momentarely
            if x_phot is not None:
                self.likelyhood = gp.GPLikelyhood(self.t,self.rv,self.model_y0, self.rv_err, hparam_chain, model_par_chain, self.kernel_name, x_phot=self.x_phot, y_phot=self.y_phot, model_y_phot=self.model_y0_batman, model_param_phot=batman_model_par_chain, yerr_phot=self.yerr_phot)
            else:
                self.likelyhood = gp.GPLikelyhood(self.t,self.rv,self.model_y0, self.rv_err, hparam_chain, model_par_chain, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            
            # logL is initial likelihood of this chain, append it to logL0 of all chains
            self.logL0.append(logL_chain)
            
            #### ATTENTION!!!! FOR SOME REASON AFTER GET_MODEL THE VALUES IN MODEL_PAR_CHAIN BECOME OMEGA AND ECCENTRICITY
            if self.mass:
                mass0_chain, mass1_chain = None, None
            hparam_chain, model_par_chain, logL_chain = None, None, None

        
        # Save this set of likelihoods as the first ones of the overall likelihood array
        self.logL_list.append(self.logL0)
        acc =  [True for i in range(self.numb_chains)]
        self.accepted.append(acc)
        self.logL_list = aux.transpose(self.logL_list)
        self.accepted = aux.transpose(self.accepted)
        if self.mass:
            self.mass0_list.append(self.mass0_0)
            self.mass0_list = aux.transpose(self.mass0_list)
            self.mass1_list.append(self.mass1_0)
            self.mass1_list = aux.transpose(self.mass1_list)
        
        
        # Verbose
        print("Initial hyper-parameter guesses: ")
        print(self.single_hp0)
        print()
        print("Initial model parameter guesses (ecc and omega are replaced by Sk and Ck): ")
        print(self.single_modpar0)
        print()
        if self.x_phot is not None:
            print("Modelling RVs and Photometry at the same time")
            print("Initial batman model parameter guesses: ")
            print(self.single_batman_modpar0)
        print("Initial Log Likelihood: ", self.logL0[0])
        print()
        print("Number of chains: ", self.numb_chains)
        print()
        
        
    
    
    def split_step(self, n_splits=2, a=2., Rstar=None, Mstar=None):
        '''
        self, n_splits=2, a=1.25, Rstar=None, Mstar=None
        
        Parameters
        ----------
        n_splits : integer, optional
            Number of subsplits of the total number of chains. The default is 2.
        a : float, optional
            Adjustable scale parameter. The default is 1.25.
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
        if self.x_phot is not None:
            self.batman_modpar = np.zeros_like(self.batman_modpar0)
            S1_phot = []
            S2_phot = []
        
        
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
                    if self.x_phot is not None:
                        S1_phot.append(self.batman_modpar0[walker])
                else:
                    S2_hp.append(self.hp0[walker])
                    S2_mod.append(self.modpar0[walker])
                    if self.x_phot is not None:
                        S2_phot.append(self.batman_modpar0[walker])
            S1_hp = np.array(S1_hp)
            S2_hp = np.array(S2_hp)
            S1_mod = np.array(S1_mod)
            S2_mod = np.array(S2_mod)
            N_chains1 = len(S1_hp)
            N_chains2 = len(S2_hp)
            if self.x_phot is not None:
                S1_phot = np.array(S1_phot)
                S2_phot = np.array(S2_phot)

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
                
                if self.x_phot is not None:
                    Xj_phot = S2_phot[rint]
                    Xk_new_phot=np.zeros_like(Xj_phot)
                    for i in range(len(Xj_phot)):
                        if i in (0,1,2,3,8,9,14,15,16):
                            Xk_new_phot[i] = Xj_phot[i] + (S1_phot[chain][i]-Xj_phot[i]) * z
                    Xk_new_phot[4]=Xk_new_mod[1]
                    Xk_new_phot[5]=Xk_new_mod[5]
                    Xk_new_phot[6],Xk_new_phot[7]= aux.to_ecc(Xk_new_mod[3],Xk_new_mod[4])
                    Xk_new_phot[10]=Xk_new_mod[6]
                    Xk_new_phot[11]=Xk_new_mod[10]
                    Xk_new_phot[12],Xk_new_phot[13]= aux.to_ecc(Xk_new_mod[8],Xk_new_mod[9])
                    phot_check = parameter_check_phot(Xk_new_phot)

                #print("new", Xk_new)
                
                
                if Rstar is not None and Mstar is not None:
                    model_param_check = parameter_check(Xk_new_mod, self.model_name, Rstar, Mstar)
                else:
                    model_param_check = parameter_check(Xk_new_mod, self.model_name)
                
                if self.x_phot is not None:
                    while (np.min(Xk_new) < 0) or (not model_param_check) or (not phot_check):
                        # Compute the step and apply it
                        # The random number is the same within a chain but different between chains
                        z = (np.random.rand()*(a-1) + 1)**2 / a
                        Xk_new = Xj + (S1_hp[chain] - Xj) * z
                        Xk_new_mod = Xj_mod + (S1_mod[chain] - Xj_mod) * z
                        Xk_new_phot = Xj_phot + (S1_phot[chain]-Xj_phot) * z
                    
                        if Rstar is not None and Mstar is not None:
                            model_param_check = parameter_check(Xk_new_mod, self.model_name, Rstar, Mstar)
                        else:
                            model_param_check = parameter_check(Xk_new_mod, self.model_name)
                        phot_check = parameter_check_phot(Xk_new_phot)
                            
                else:
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
                
                '''while (np.min(Xk_new) < 0) or (not model_param_check):
                    # Compute the step and apply it
                    # The random number is the same within a chain but different between chains
                    z = (np.random.rand()*(a-1) + 1)**2 / a
                    Xk_new = Xj + (S1_hp[chain] - Xj) * z
                    Xk_new_mod = Xj_mod + (S1_mod[chain] - Xj_mod) * z
                    
                    if Rstar is not None and Mstar is not None:
                        model_param_check = parameter_check(Xk_new_mod, self.model_name, Rstar, Mstar)
                    else:
                        model_param_check = parameter_check(Xk_new_mod, self.model_name)'''
                                
            
                        
                log_z = np.log(z)
                    
                # Save as the new value for the chain, by finding where the initial step was positioned and putting it in the same position
                if self.x_phot is not None:
                    for o in range(self.numb_chains):
                        # Kernel and model parameters are still kept together after shuffling
                        if (self.hp0[o][0] == S1_hp[chain][0]) and (self.modpar0[o][0] == S1_mod[chain][0]) and (self.batman_modpar0[o][0] == S1_phot[chain][0]):
                            position = o
                else:
                    for o in range(self.numb_chains):
                        # Kernel and model parameters are still kept together after shuffling
                        if (self.hp0[o][0] == S1_hp[chain][0]) and (self.modpar0[o][0] == S1_mod[chain][0]):
                            position = o
                '''for o in range(self.numb_chains):
                    # Kernel and model parameters are still kept together after shuffling
                    if (self.hp0[o][0] == S1_hp[chain][0]) and (self.modpar0[o][0] == S1_mod[chain][0]):
                        position = o'''
                
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
                if self.x_phot is not None:
                    for v in range(len(self.batman_modpar_vary)):
                        if self.batman_modpar_vary[v]:
                            self.batman_modpar[position][v] = Xk_new_phot[v]
                        else:
                            self.batman_modpar[position][v] = S1_phot[chain][v]
                            
                self.logz[position] = log_z
                
            
            # Once all elements of the first section have taken the step, reinitialise the S1,2 arrays
            #print(len(S1_mod), len(S2_mod))
            
            S1_hp = []
            S2_hp = []
            
            S1_mod = []
            S2_mod = []
            
            if self.x_phot is not None:
                S1_phot = []
                S2_phot = []
            
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
        self.mass0 = []
        self.mass1 =[]
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
                
            
            if self.x_phot is not None:
                batman_param = None
                batman_param = Model_Par_Creator.create(["Batman2"])
                for i, key in zip(range(len(self.batman_modpar_vary)), batman_param.keys()):
                    batman_param[key] = gp.Parameter(value=self.batman_modpar[chain][i], error=self.batman_modpar_err[i], vary=self.batman_modpar_vary[i])
                self.model_y_phot = get_model(["Batman2"], self.x_phot,batman_param)
            
            if self.mass:
                #print(model_param["omega_0"].value, model_param["ecc_0"].value, model_param["omega_1"].value, model_param["ecc_1"].value)
                mass0_chain = mc.mass_calc(model_param["P_0"].value, model_param["K_0"].value, model_param["omega_0"].value, model_param["ecc_0"].value, 0.743)
                mass1_chain = mc.mass_calc(model_param["P_1"].value, model_param["K_1"].value, model_param["omega_1"].value, model_param["ecc_1"].value, 0.743)
                self.mass0.append(mass0_chain)
                self.mass1.append(mass1_chain)
            
            
            #print("param", param)
            #print("par model", model_param)
            #print("hyperpar", param)
            
            #self.model_y = None
            # Get new model
            if self.flags is None:
                self.model_y = get_model(self.model_name, self.t, model_param, flags=None)
            if self.flags is not None:
                self.model_y = get_model(self.model_name, self.t, model_param, flags=self.flags)
            #print("model", self.model_y[0:5])
            
            #print(model_param)
            #For some reason after going through get model we get the ecc and omega instead???
            
            self.likelyhood = None
            #logL_chain = None
            # Use current hp and model to compute the logL
            if self.x_phot is not None:
                self.likelyhood = gp.GPLikelyhood(self.t, self.rv, self.model_y, self.rv_err, param, model_param, self.kernel_name, x_phot=self.x_phot, y_phot=self.y_phot, yerr_phot=self.yerr_phot, model_y_phot=self.model_y_phot, model_param_phot=batman_param)
            else:
                self.likelyhood = gp.GPLikelyhood(self.t, self.rv, self.model_y, self.rv_err, param, model_param, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)
            '''self.likelyhood = gp.GPLikelyhood(self.t, self.rv, self.model_y, self.rv_err, param, model_param, self.kernel_name)
            logL_chain = self.likelyhood.LogL(self.prior_list)'''
            #print("h", logL_chain)
            
            self.logL.append(logL_chain)
            
        
        # Final output: a logL 2d array, ncols = 1, nrows = numb_chains        
        

        
    def compare(self):

        
        # Create empty array to save decisions in to then concatenate to the list arrays
        hp_decision = []
        modpar_decision = []
        logL_decision = []
        self.acceptance_chain = []
        if self.mass:
            mass0_decision = []
            mass1_decision = []
        if self.x_phot is not None:
            batman_decision = []
        
        
        
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
            if self.x_phot is not None:
                batman_modpar = self.batman_modpar
                batman_modpar0 = self.batman_modpar0
            
            # If the logL is larger than one, then it's exponent will definitely be larger than 1 and will automatically be accepted
            if diff_logL_z > 1:
                logL_decision.append(self.logL[chain])
                hp_decision.append(hp[chain])
                modpar_decision.append(modpar[chain])
                self.acceptance_chain.append(True)
                if self.mass:
                    mass0_decision.append(self.mass0[chain])
                    mass1_decision.append(self.mass1[chain])
                if self.x_phot is not None:
                    batman_decision.append(batman_modpar[chain])
                
            # If the logL is ver small (eg. smaller than -35), automatic refusal
            if diff_logL_z < -35.:
                logL_decision.append(self.logL0[chain])
                hp_decision.append(hp0[chain])
                modpar_decision.append(modpar0[chain])
                self.acceptance_chain.append(False)
                if self.mass:
                    mass0_decision.append(self.mass0_0[chain])
                    mass1_decision.append(self.mass1_0[chain])
                if self.x_phot is not None:
                    batman_decision.append(batman_modpar0[chain])
            
            if (diff_logL_z >= -35.) and (diff_logL_z <= 1.):
                # Generate random number from uniform distribution
                MH_rand = random.uniform(0,1)
                # if diff_Lz is smaller than the number, accept the step
                if MH_rand <= (np.exp(diff_logL_z)):
                    logL_decision.append(self.logL[chain])
                    hp_decision.append(hp[chain])
                    modpar_decision.append(modpar[chain])
                    self.acceptance_chain.append(True)
                    if self.mass:
                        mass0_decision.append(self.mass0[chain])
                        mass1_decision.append(self.mass1[chain])
                    if self.x_phot is not None:
                        batman_decision.append(batman_modpar[chain])
                # if it is larger than the number reject the step
                else:
                    logL_decision.append(self.logL0[chain])
                    hp_decision.append(hp0[chain])
                    modpar_decision.append(modpar0[chain])
                    self.acceptance_chain.append(False)
                    if self.mass:
                        mass0_decision.append(self.mass0_0[chain])
                        mass1_decision.append(self.mass1_0[chain])
                    if self.x_phot is not None:
                        batman_decision.append(batman_modpar0[chain])
            
            
    
            

        
        # Now concatenate all the 2D arrays into the 3D list arrays
        # Start with logL list and append, nrows = nchains, ncols = niterations
        self.logL_list = np.column_stack((self.logL_list, logL_decision))
        if self.mass:
            self.mass0_list = np.column_stack((self.mass0_list, mass0_decision))
            self.mass1_list = np.column_stack((self.mass1_list, mass1_decision))
        self.accepted = np.column_stack((self.accepted, self.acceptance_chain))
        # Rest of lists, nrows = nchains, ncols = nparam, ndepth = niterations
        self.hparameter_list = np.dstack((self.hparameter_list, hp_decision))
        self.model_parameter_list = np.dstack((self.model_parameter_list, modpar_decision))
        if self.x_phot is not None:
            self.batman_model_parameter_list = np.dstack((self.batman_model_parameter_list,batman_decision))
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
                if self.mass:
                    self.mass0_0[chain] = self.mass0[chain]
                    self.mass1_0[chain] = self.mass1[chain]
                if self.x_phot is not None:
                    self.batman_modpar0[chain] = self.batman_modpar[chain]
            if self.acceptance_chain[chain] is False:
                self.logL0[chain] = self.logL0[chain]
                self.hp0[chain] = self.hp0[chain]
                self.modpar0[chain] = self.modpar0[chain]
                if self.mass:
                    self.mass0_0[chain] = self.mass0_0[chain]
                    self.mass1_0[chain] = self.mass1_0[chain]
                if self.x_phot is not None:
                    self.batman_modpar0[chain] = self.batman_modpar0[chain]
        
        # IMPORTANT!! In model_parameter_list, if the model is a keplerian we have Sk and Ck, not ecc and omega
        
        if self.mass:
            if self.x_phot is not None:
                return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted, self.mass0_list, self.mass1_list, self.batman_model_parameter_list
            else:
                return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted, self.mass0_list, self.mass1_list
            
            
        else:
            if self.x_phot is not None:
                return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted, self.batman_model_parameter_list
            else:
                return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted
    
    '''if self.mass:
        return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted, self.mass0_list, self.mass1_list
    else:
        return self.logL_list, self.hparameter_list, self.model_parameter_list, self.accepted
    '''
    
    
    def gelman_rubin_calc(self, burn_in):

        """
        Returns the Gelman-Rubin convergence statistic.

        Must be calculated for each parameter independently.
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
                if (
                np.nanmax(self.hparameter_list[:, hyper_param, :])
                - np.nanmin(self.hparameter_list[:, hyper_param, :])
                == 0.0
            ):
                    all_R.append(1.0)
                    continue
                for chain in range(J):
                    # Calculate chain mean
                    param_chain = self.hparameter_list[chain, hyper_param, burn_in:]

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
                param_chain = self.model_parameter_list[chain, param, burn_in:]

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
        
        if self.x_phot is not None:
            P = np.shape(self.batman_model_parameter_list)[1]
            for param in range(P):
                if self.batman_modpar_names[param] not in self.modpar_names:
                    chain_means = []
                    intra_chain_vars = []
                    if (
                        np.nanmax(self.batman_model_parameter_list[:, param, :])
                        - np.nanmin(self.batman_model_parameter_list[:, param, :])
                        == 0.0
                    ):
                        all_R.append(1.0)
                        continue
                    for chain in range(J):
                        # Calculate chain mean
                        param_chain = self.batman_model_parameter_list[chain, param, burn_in:]

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
            #assert len(all_R) == self.numb_param
            assert np.all(all_R >= 1.0)
        except:
            import pdb

            pdb.set_trace()
        
        return all_R
    
    
    
    









def run_MCMC(iterations, t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains=None, n_splits=None, a=None, Rstar=None, Mstar=None, flags=None, plot_convergence=False, saving_folder=None, mass=False, x_phot=None, y_phot=None, yerr_phot=None, model_y_phot=None, model_param_phot=None):
    
    from MCMC_affine_multi_batman import MCMC
    
    gelman_rubin_limit = 1.1
    

    if x_phot is not None:
        if mass:
            if flags is None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, mass=True, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, mass=True, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
            if flags is not None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, mass=True, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, mass=True, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
        else:
            if flags is None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
            if flags is not None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, x_phot=x_phot, y_phot=y_phot, yerr_phot=yerr_phot, model_y_phot=model_y_phot, model_param_phot=model_param_phot)
            
    else:
        if mass:
            if flags is None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, mass=True)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, mass=True)
            if flags is not None:
                if numb_chains is None:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, mass=True)
                    numb_chains=100
                else:
                    _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, mass=True)
        else:
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
    
    '''if mass:
        if flags is None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, mass=True)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, mass=True)
        if flags is not None:
            if numb_chains is None:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, flags=flags, mass=True)
                numb_chains=100
            else:
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags, mass=True)
    else:
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
                _ = MCMC(t, rv, rv_err, hparam0, kernel_name, model_param0, model_name, prior_list, numb_chains, flags=flags)'''
    
    # Initialise progress bar
    print("Start Iterations")
    print()
    
    start = time.time()
    burn_in = min(200, iterations // 10)
    
    
    if plot_convergence:
        import matplotlib.pyplot as plt

        conv_f, ax = plt.subplots(figsize=(15,15))
        hparam_names = aux.hparam_names(kernel_name)
        model_param_names = aux.model_param_names(model_name)
        all_param_names = hparam_names + model_param_names
        #all_param_names = model_param_names
        if x_phot is not None:
            batman_names = aux.batman_names()
            all_param_names = hparam_names + model_param_names + batman_names
        conv_vals = {i: [] for i in all_param_names}    
        conv_iters = []

        if saving_folder is None:
            import os

            saving_folder = os.getcwd()
    
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
        
        if x_phot is not None:
            if mass:
                logL_list, hparameter_list, model_parameter_list, accepted, mass0, mass1, batman_list = _.reset()
            else:
                logL_list, hparameter_list, model_parameter_list, accepted, batman_list = _.reset()
        else:
            if mass:
                logL_list, hparameter_list, model_parameter_list, accepted, mass0, mass1 = _.reset()
            else:
                logL_list, hparameter_list, model_parameter_list, accepted = _.reset()
            
        
        '''if mass:
            logL_list, hparameter_list, model_parameter_list, accepted, mass0, mass1 = _.reset()
        else:
            logL_list, hparameter_list, model_parameter_list, accepted = _.reset()'''
        
        if (iteration % 2==0) or iteration == iterations-1:
            aux.printProgressBar(iteration, iterations-1, length=50)

        if plot_convergence:
            if (iteration >= burn_in) and (iteration - burn_in) % 20 == 0:
                # Check whether the chains have converged...
                R_list = _.gelman_rubin_calc(burn_in)
                # print(iteration)
                if plot_convergence:
                    assert len(R_list) == len(conv_vals)
                    if x_phot is not None:
                        all_names = list(set(all_param_names))
                        for i, R in enumerate(R_list):
                            conv_vals[all_names[i]].append(R)
                    else:
                        for i, R in enumerate(R_list):
                            conv_vals[all_param_names[i]].append(R)
                    conv_iters.append(iteration)

                    # for param in conv_vals:
                    #     ax.plot(conv_iters, conv_vals[param], label=param)
                    # ax.legend()
                    # ax.axhline(gelman_rubin_limit, c='k', ls='--')
                    # ax.set_xlim(left=burn_in)
                    # ax.set_ylim(bottom = 1.)
                    # ax.set_yscale('log')
                    # conv_f.savefig(saving_folder+'/convergence_plot.png')

                if np.all(R_list < gelman_rubin_limit):
                    # Convergence reached...
                    completed_iterations = iteration + 1
                    print("CONVERGENCE REACHED!! at iteration {}".format(iteration))
                    break
    else:
        completed_iterations = iterations
                    

    if plot_convergence:
        print("conv_iters",conv_iters)
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
    if x_phot is not None:
        if mass:
            return logL_list, hparameter_list, model_parameter_list, mass0, mass1, batman_list, completed_iterations
        else:
            return logL_list, hparameter_list, model_parameter_list, batman_list, completed_iterations
    else:
        if mass:
            return logL_list, hparameter_list, model_parameter_list, mass0, mass1, completed_iterations
        else:
            return logL_list, hparameter_list, model_parameter_list, completed_iterations
    '''if mass:
        return logL_list, hparameter_list, model_parameter_list, mass0, mass1, completed_iterations
    else:
        return logL_list, hparameter_list, model_parameter_list, completed_iterations
    '''
        


    

        
        

    


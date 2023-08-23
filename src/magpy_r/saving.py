"""
Function to allow the user to save any or all outputs from the mcmc

Contains:
    saving function

    
Author: Federica Rescigno, Bryce Dixon
Version: 22.08.2023    
"""
import os

import numpy as np
import pandas as pd

import magpy_r.parameters as par
import magpy_r.models as mod
import magpy_r.gp_likelihood as gp
from magpy_r.mcmc_aux import get_model
from magpy_r.plotting import offset_subtract
import magpy_r.auxiliary as aux


# saving function


def save(folder_name, rv, time, rv_err, model_list = None, init_hparam = None, kernel = None, init_param = None, prior_list = [], fin_hparam_post = None, fin_param_post = None, logl_chain = None, masses = None, fin_param_values = None, fin_param_erru = None, fin_param_errd = None, flags = None, Mstar = None, fin_to_skck = False, burnin = None):
    """
    Saves offset subtracted and combined RVs and times, rv_error, kernel name, model list, initial hyperparameters and parameters, initial LogL, priors, final hyperparameter and parameter posteriors, final LogL posterior, mass posteriors, final hyperparameter, parameter and mass values along with errors, final logL value
    
    Parameters
    ----------
    folder_name: string
        file destination of the folder to save to
    rv: array of floats
        array of the rv values
    time: array of floats
        array of the time values
    rv_err: array of floats
        array of the rv error values
    model_list: list of strings, optional
        list of the names of the model in use, defaults to None
    init_hparam: dictionary, optional
        dictionary of the initial hyperparameters, defaults to None
    kernel: string, optional
        name of the chosen kernel, defaults to None
    init_param: dictionary, optional
        dictionary of the initial model parameters, defaults to None
    prior_list: dictionary, optional
        dictionary of the priors, defaults to empty list
    fin_hparam_post: array of floats, optional
        3d array of the hypermarater posteriors, defaults to None
    fin_param_post: array of floats, optional
        3d array of the model parameter posteriors, defaults to None
    logl_chain: array of floats, optional
        3d array of the logL posteriors, defaults to None
    masses: array of floats, optional
        3d array of the mass posteriors, defaults to None
    fin_param_values: list of floats, optional
        list of the final hyperparameter, parameter, and mass values from the posteriors, defaults to None
    fin_param_erru: list of floats, optional
        list of the final hyperparameter, parameter, and mass upper errors from the posteriors, defaults to None
    fin_param_errd: list of floats, optional
        list of the final hyperparameter, parameter, and mass lower errors form the posteriors, defaults to None
    flags: array of floats, optional
        array of floats representing the offsets, defaults to None
    Mstar: float, optional
        mass of the host star in solar masses
    fin_to_skck: bool, optional
        if True, returns final keplerian parameters with Sk and Ck, if False returns final keplerian parameters with ecc and omega, defaults to False
    burnin: integer, optional
        integer value to specify the length of the burn in, defaults to None for no burn in
    """
    
    # create new folder titled current run, at the moment the code will not run if the folder already exists
    if os.path.exists(folder_name):
        pass
    else:
        os.mkdir(folder_name)
    
    # create files for rv, time, and rv error
    rv_file = os.path.join(folder_name, "rv_data.txt")
    time_file = os.path.join(folder_name, "time_data.txt")
    rv_err_file = os.path.join(folder_name, "rv_err.txt")
    
    if flags is not None:
        
        # if offsets exist append them to a list
        offsets = []
        try:
            off = init_param['offset'].value
            offsets.append(off)
        except:
            for i in range(len(model_list)):
                try:
                    off = init_param['offset_'+str(i)].value
                    offsets.append(off)
                except:
                    continue
        
        # subtract the offsets and save the offset rv to a file        
        off_rv = offset_subtract(rv, flags, offsets)
        rv_off_file = os.path.join(folder_name, "rv_data_offset_sub.txt")
        np.savetxt(rv_off_file, off_rv)
    
    # save the times, rvs, and rv error to their files
    np.savetxt(time_file, time)
    np.savetxt(rv_file, rv)
    np.savetxt(rv_err_file, rv_err)
    
    # if a kernel has been entered, save all the hyperparameter information to an initial conditions file
    if kernel is not None:
        initial_file = os.path.join(folder_name, "initial_conditions.txt")
        initial_cond_file = open(initial_file, "w+")
        initial_cond_file.write("\nKernel Name:\n")
        initial_cond_file.write(kernel)
        initial_cond_file.write("\nInitial Hyperparameters:\n")
        initial_cond_file.write(init_hparam.__str__())
    # if a model has been entered, add the model parameters to this file
    if model_list is not None:
        initial_cond_file.write("\nModel List:\n")
        initial_cond_file.write(model_list.__str__())
        initial_cond_file.write("\nInitial Parameters:\n")
        initial_cond_file.write(init_param.__str__())
    # if Mstar is given, save that aswell
    if Mstar is not None:
        initial_cond_file.write("\nHost Star Mass:\n")
        initial_cond_file.write(Mstar.__str__())
        initial_cond_file.write(" Solar Masses")
        
        # generate the model to get the log likelihood
        model_y = get_model(model_list, time, init_param, to_ecc = False, flags = flags)    
        loglik = gp.GPLikelihood(time, rv, rv_err, init_hparam, kernel, model_y, init_param)
        logL = loglik.LogL(prior_list)
        
        # add the initial log likelihood to the file
        initial_cond_file.write("\nInitial LogL:\n")
        initial_cond_file.write(logL.__str__())
    initial_cond_file.close()
    
    # create a prior file and save the prior list to the file
    prior_file = os.path.join(folder_name, "priors.txt")
    prior_file = open(prior_file, "w+")
    prior_file.write("\nPriors:\n")
    prior_file.write(prior_list.__str__())
    prior_file.close()

    # if a hyperparameter posterior has been entered, save these to files based on hyperparameter as arrays with ncolumns = chains, and nrows = iterations
    if fin_hparam_post is not None:
        hparams = aux.hparam_names(kernel, plotting = False)
        for N,i in enumerate(hparams):
            hparam_post = os.path.join(folder_name, "{}_posteriors.txt".format(i))
            if burnin is not None:
                assert type(burnin) == int, "burnin should be an integer or None"
                post_hparam = fin_hparam_post[:,:,N]
                burn_hparam = post_hparam[burnin:, :]
                np.savetxt(hparam_post, burn_hparam)
            else:
                np.savetxt(hparam_post, fin_hparam_post[:,:,N])
    
    # do the same with model parameters    
    if fin_param_post is not None:
        params = aux.model_param_names(model_list, SkCk = True, plotting = False)
        for N,i in enumerate(params):
            param_post = os.path.join(folder_name, "{}_posteriors.txt".format(i))
            if burnin is not None:
                assert type(burnin) == int, "burnin should be an integer or None"
                post_param = fin_param_post[:,:,N]
                burn_param = post_param[burnin:, :]
                np.savetxt(param_post, burn_param)
            else:
                np.savetxt(param_post, fin_param_post[:,:,N])
    
    # do the same with the logL chain
    if logl_chain is not None:
        logl_post = os.path.join(folder_name, "logL_posteriors.txt")
        if burnin is not None:
            assert type(burnin) == int, "burnin should be an integer or None"
            post_logl = logl_chain[:,:,0]
            burn_logl = post_logl[burnin:, :]
            np.savetxt(logl_post, burn_logl)
        else:
            np.savetxt(logl_post, logl_chain[:,:,0])
    
    # do the same with the masses
    if masses is not None:
        mass_list = []
        if len(masses[0,0,:]) == 1 and len(model_list) == 1:
            name = "mass"
            mass_list.append(name)
        else:
            for i in range(len(masses[0,0,:])):
                name = "mass_{}".format(i)
                mass_list.append(name)
        
        for N,i in enumerate(mass_list):
            mass_post = os.path.join(folder_name, "{}_posteriors.txt".format(i))
            if burnin is not None:
                assert type(burnin) == int, "burnin should be an integer or None"
                post_mass = masses[:,:,N]
                burn_mass = post_mass[burnin:, :]
                np.savetxt(mass_post, burn_mass)
            else:
                np.savetxt(mass_post, masses[:,:,N])
    
    # if final parameter values have been entered, start by getting a list of all the existing parameters, should match the length of the final parameter values list        
    if fin_param_values is not None:
        try:
            hparams = aux.hparam_names(kernel, plotting = False)
            # SkCk will acocunt for whether the user wants ecc and omega returned or Sk and Ck returned
            params = aux.model_param_names(model_list, SkCk = fin_to_skck, plotting = False)
            mass_list = []
            if len(fin_param_values) > (len(hparams)+len(params)):
                try:
                    init_param['P'].value
                    mass_list.append('mass')
                except:
                    for i in range(len(model_list)):
                        try:
                            init_param['P_'+str(i)].value
                            mass_list.append('mass_'+str(i))
                        except:
                            continue
            # mass list will just be empty if there are no masses
            param_list = hparams + params + mass_list
        except:
            # if there is no model the list will just be hyperparameters
            param_list = hparams

        # create the final parameter file
        fin_param_file = os.path.join(folder_name, "final_parameter_values.txt")
        fin_param_file = open(fin_param_file, "w+")
        fin_param_table = os.path.join(folder_name, "final_parameter_table.txt")
        fin_param_table = open(fin_param_table, "w+")
        value_list = []
        erru_list = []
        errd_list = []
        
        # if the user wants to return Sk and Ck, read the values as they are form the fin_param_values list, naming the files after the parameter in param_list
        if fin_to_skck is True:
            for N,i in enumerate(param_list):
                value_list.append(fin_param_values[N])
                fin_param_file.write("\n{}:\n".format(i))
                fin_param_file.write(fin_param_values[N].__str__())
                try:
                    # if errors are given try to include the errors
                    fin_param_file.write("\n+")
                    # noting the errors given in fin_param_err and fin_param_errd do not have the value subtracted from them so this must be done
                    erru = fin_param_erru[N] - fin_param_values[N]
                    erru_list.append(erru)
                    fin_param_file.write(erru.__str__())
                except:
                    continue
                try:
                    fin_param_file.write("\n")
                    errd = fin_param_errd[N] - fin_param_values[N]
                    errd_list.append(errd)
                    fin_param_file.write(errd.__str__())
                except:
                    continue
            
        if fin_to_skck is False:
            # if the user wants to return ecc and omega, Sk and Ck in fin_param_values will need converting
            for N,i in enumerate(param_list):
                if i.startswith('ecc'):
                    # if we are on ecc in the list, this corresponds to ck in fin_param_values, pull that out along with the next value, sk
                    sk = fin_param_values[N]
                    ck = fin_param_values[N+1]
                    # run them through to_ecc to get ecc and omega
                    ecc, omega = aux.to_ecc(sk, ck)
                    # add ecc to the file
                    value_list.append(ecc)
                    fin_param_file.write("\n{}:\n".format(i))
                    fin_param_file.write(ecc.__str__())
                elif i.startswith('omega'):
                    # omega is checked next so can just be added to the file
                    value_list.append(omega)
                    fin_param_file.write("\n{}:\n".format(i))
                    fin_param_file.write(omega.__str__())
                else:
                    # the rest added as normal
                    value_list.append(fin_param_values[N])
                    fin_param_file.write("\n{}:\n".format(i))
                    fin_param_file.write(fin_param_values[N].__str__())
                try:
                    if i.startswith('ecc'):
                        # perform similar steps for errors
                        sk_erru = fin_param_erru[N]
                        ck_erru = fin_param_erru[N+1]
                        ecc_erru, omega_erru = aux.to_ecc(sk_erru, ck_erru)
                        fin_param_file.write("\n+")
                        ecc_erru = ecc_erru - ecc
                        erru_list.append(ecc_erru)
                        fin_param_file.write(ecc_erru.__str__())
                    elif i.startswith('omega'):
                        fin_param_file.write("\n+")
                        omega_erru = omega_erru - omega
                        erru_list.append(omega_erru)
                        fin_param_file.write(omega_erru.__str__())
                    else:
                        fin_param_file.write("\n+")
                        erru = fin_param_erru[N] - fin_param_values[N]
                        erru_list.append(erru)
                        fin_param_file.write(erru.__str__())
                except:
                    continue
                try:
                    if i.startswith('ecc'):
                        sk_errd = fin_param_errd[N]
                        ck_errd = fin_param_errd[N+1]
                        ecc_errd, omega_errd = aux.to_ecc(sk_errd, ck_errd)
                        fin_param_file.write("\n")
                        ecc_errd = ecc_errd - ecc
                        errd_list.append(ecc_errd)
                        fin_param_file.write(ecc_errd.__str__())
                    elif i.startswith('omega'):
                        fin_param_file.write("\n")
                        omega_errd = omega_errd - omega
                        errd_list.append(omega_errd)
                        fin_param_file.write(omega_errd.__str__())
                    else:
                        fin_param_file.write("\n")
                        errd = fin_param_errd[N] - fin_param_values[N]
                        errd_list.append(errd)
                        fin_param_file.write(errd.__str__())
                except:
                    continue
        fin_param_file.close()
        value_list = [ '%.3f' % elem for elem in value_list ]
        erru_list = [ '%.3f' % elem for elem in erru_list ]
        errd_list = [ '%.3f' % elem for elem in errd_list ]
            
        # try to create a final logl value if we have the correct inputs
        try:
            # require a new hparam and param list from the fin_param_values
            hparams_list = aux.hparam_names(kernel, plotting = False)
            params_list = aux.model_param_names(model_list, SkCk = False, plotting = False)
            new_hparam = par.par_create(kernel)
            for N,i in enumerate(hparams_list):
                # new hparam list can be created by taking the first few values from fin_param_values (based on the length of hparams_list)
                value = fin_param_values[N]
                error = fin_param_erru[N] - fin_param_values[N]
                new_hparam[i] = par.parameter(value, error, True)
            new_param = mod.mod_create(model_list)
            for N,i in enumerate(params_list):
                # new param list can be made similarly but requires ck and sk to be converted into ecc and omega, done in a similar way to above
                if i.startswith('ecc'):
                    # read the correct fin_param_values value by adding the length of hparams
                    sk = fin_param_values[N + len(hparams)]
                    ck = fin_param_values[N + len(hparams) +1]
                    sk_err = fin_param_erru[N + len(hparams)]
                    ck_err = fin_param_erru[N + len(hparams) +1]
                    # convert to ecc and omega
                    ecc, omega = aux.to_ecc(sk, ck)
                    ecc_err, omega_err = aux.to_ecc(sk_err, ck_err)
                    ecc_err = ecc_err - ecc
                    new_param[i] = par.parameter(ecc, ecc_err, True)
                elif i.startswith('omega'):
                    omega_err = omega_err - omega
                    new_param[i] = par.parameter(omega, omega_err, True)
                else:
                    # create the rest as normal
                    value = fin_param_values[N + len(hparams)]
                    error = fin_param_erru[N + len(hparams)] - value
                    new_param[i] = par.parameter(value, error, True)

            # create a new model y based on the new parameters in order to get a new logL based on the new parameters
            model_y = get_model(model_list, time, new_param, to_ecc = False, flags = flags)    
            loglik = gp.GPLikelihood(time, rv, rv_err, new_hparam, kernel, model_y, new_param)
            logL = loglik.LogL(prior_list)
            # save the final logL to a final logL file
            fin_logl_file = os.path.join(folder_name, "final_info.txt")
            fin_logl_file = open(fin_logl_file, "w+")
            fin_logl_file.write("\nFinal LogL:\n")
            fin_logl_file.write(logL.__str__())
            try:
            # if a hyperparameter posterior chain has been entered, add the number of chains and iterations to the initial conditions file
                num_chains = len(fin_hparam_post[0,:,0])
                iterations = len(fin_hparam_post[:,0,0])-1
                fin_logl_file.write("\nNumber of Chains:\n")
                fin_logl_file.write(num_chains.__str__())
                fin_logl_file.write("\nCompleted Iterations:\n")
                fin_logl_file.write(iterations.__str__())
            except:
                pass
            fin_logl_file.close()
        except:
            pass
        
        try:
            hparams = aux.hparam_names(kernel, plotting = True)
            # SkCk will acocunt for whether the user wants ecc and omega returned or Sk and Ck returned
            params = aux.model_param_names(model_list, SkCk = fin_to_skck, plotting = True)
            mass_list = []
            if len(fin_param_values) > (len(hparams)+len(params)):
                try:
                    init_param['P'].value
                    mass_list.append('mass')
                except:
                    for i in range(len(model_list)):
                        try:
                            init_param['P_'+str(i)].value
                            mass_list.append('mass_'+str(i))
                        except:
                            continue
            # mass list will just be empty if there are no masses
            param_list = hparams + params + mass_list
        except:
            # if there is no model the list will just be hyperparameters
            param_list = hparams
            
        try:
            # build the latex tabel using the plot names
            logL = "%.3f" % logL
            param_list = param_list + ['Final LogL', 'Num Iterations', 'Num Chains']
            value_list = value_list + [logL, iterations, num_chains]
            erru_list = erru_list + [0,0,0]
            errd_list = errd_list + [0,0,0]
            param_tab = np.array([errd_list, erru_list, value_list, param_list])
            param_tab = np.rot90(param_tab, 1, axes = (1,0))
            param_tab = pd.DataFrame(param_tab)
            param_tab = param_tab.to_latex(index = False, header = ['Parameter', 'Value', 'Error Up', 'Error Down'])
            fin_param_table.write(param_tab.__str__())
            fin_param_table.close()
        except:
            try:
                param_tab = np.array([errd_list, erru_list, value_list, param_list])
                param_tab = np.rot90(param_tab, 1, axes = (1,0))
                param_tab = pd.DataFrame(param_tab)
                param_tab = param_tab.to_latex(index = False, header = ['Parameter', 'Value', 'Error Up', 'Error Down'])
                fin_param_table.write(param_tab.__str__())
                fin_param_table.close()
            except:
                pass
    
        
    
    
    
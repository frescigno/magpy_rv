"""


Contains:

    
Author: Bryce Dixon
Version: 08.08.2023    
"""

import numpy as np
import scipy as sc
import Parameters as par
import Models as mod
import GP_Likelihood as gp
from MCMC_aux import get_model
from new_plotting import offset_subtract
import new_plotting as plot
import auxiliary as aux
import os


# saving function


def save(folder_name, rv, time, rv_err, model_list = None, init_hparam = None, kernel = None, init_param = None, prior_list = None, fin_hparam_post = None, fin_param_post = None, logl_chain = None, masses = None, fin_param_values = None, fin_param_erru = None, fin_param_errd = None, flags = None):
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
    """
    current = "current_run"
    path = os.path.join(folder_name, current)
    path = os.mkdir(path)
    folder_name = folder_name+'/current_run/'
    
    rv_file = os.path.join(folder_name, "rv_data.txt")
    time_file = os.path.join(folder_name, "time_data.txt")
    rv_err_file = os.path.join(folder_name, "rv_err.txt")
    
    if flags is not None:
        
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
                
        off_rv = plot.offset_subtract(rv, flags, offsets)
        rv_off_file = os.path.join(folder_name, "rv_data_offset_sub.txt")
        np.savetxt(rv_off_file, off_rv)
    
    np.savetxt(time_file, time)
    np.savetxt(rv_file, rv)
    np.savetxt(rv_err_file, rv_err)
    
    if kernel is not None:
        initial_file = os.path.join(folder_name, "initial_conditions.txt")
        initial_cond_file = open(initial_file, "w+")
        initial_cond_file.write("\nKernel Name:\n")
        initial_cond_file.write(kernel)
        initial_cond_file.write("\nInitial Hyperparameters:\n")
        initial_cond_file.write(init_hparam.__str__())
    if model_list is not None:
        initial_cond_file.write("\nModel List:\n")
        initial_cond_file.write(model_list.__str__())
        initial_cond_file.write("\nInitial Parameters:\n")
        initial_cond_file.write(init_param.__str__())
        
        model_y = get_model(model_list, time, init_param, to_ecc = False, flags = flags)    
        loglik = gp.GPLikelihood(time, rv, rv_err, init_hparam, kernel, model_y, init_param)
        logL = loglik.LogL(prior_list)
        
        initial_cond_file.write("\nInitial LogL:\n")
        initial_cond_file.write(logL.__str__())
        try:
            num_chains = len(fin_hparam_post[0,:,0])
            iterations = len(fin_hparam_post[:,0,0])-1
            initial_cond_file.write("\nNumber of Chains:\n")
            initial_cond_file.write(num_chains)
            initial_cond_file.write("\nIterations:\n")
            initial_cond_file.write(iterations)
        except:
            pass
        initial_cond_file.close()
    
    prior_file = os.path.join(folder_name, "priors.txt")
    prior_file = open(prior_file, "w+")
    prior_file.write("\nPriors:\n")
    prior_file.write(prior_list.__str__())

    if fin_hparam_post is not None:
        hparams = aux.hparam_names(kernel)
        for N,i in enumerate(hparams):
            hparam_post = os.path.join(folder_name, "{}_posteriors.txt".format(i))
            np.savetxt(hparam_post, fin_hparam_post[:,:,N])
        
    if fin_param_post is not None:
        params = aux.model_param_names(model_list, SkCk = True, plotting = False)
        for N,i in enumerate(params):
            param_post = os.path.join(folder_name, "{}_posteriors.txt".format(i))
            np.savetxt(param_post, fin_param_post[:,:,N])
    
    if logl_chain is not None:
        logl_post = os.path.join(folder_name, "logL_posteriors.txt")
        np.savetxt(logl_post, logl_chain[:,:,0])
    
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
            np.savetxt(mass_post, masses[:,:,N])
        
    
        
    
    
    
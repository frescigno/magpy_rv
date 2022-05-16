"""
Template of the launcher for GP_solar and MCMC, also calls plotting

Author: Federica Rescigno
Version: 27-01-2022
"""


import numpy as np
import GP_solar as gp
from MCMC import run_MCMC
import matplotlib.pyplot as plt
import plotting



# Select where data will be called from

rootdir = '/Users/frescigno/Desktop'
datadir = rootdir + '/stellar RV data'
scriptdir = rootdir + '/GP code'

filename = '/data.rdb'
myinput = datadir + filename
data = np.genfromtxt(myinput, skip_header=0)

time = 
rv = 
rv_err = 


print("Step 0, Data visualisation")
# PLOT DATA
plot.basic_data_plot(time, rv, xlabel, ylabel, legend)



print("Step 1, Basics initialisation")

# INSTRUCTIONS

# KERNEL:
#   CHOOSE THE WANTED KERNEL AND CREATE HYPER-PARAMETER LIST
hparam = gp.Par_Creator.create("name of kernel")

#   ASSIGN A STARTING VALUE TO EACH HYPERPARAMETER
hparam['hyperparameter name'] = gp.Parameter(value=value number, error=error, vary=True)
...


# PRIORS:
prior_list = []
#   SPECIFY PRIORS FOR THE HYPERPARAMETERS AND ASSIGN VALUES
prior_param = gp.Prior_Par_Creator.create("name of prior")  
prior_param["parameter name"] = parameter value
...

#   ADD ALL PRIORS FOR EACH HYPERPARAMETER
prior_list.append(("name of hyperparameter", "name of prior", prior_param))

# REPEAT IF NEEDED




# MODEL:
#   CHOOSE MODEL AND INITIALISE IT
model_par = gp.Model_Par_Creator.create("Kepler")
model_par['model parameter name'] = gp.Parameter(value=value number, error=error, vary=True)
...

#   PRIORS
prior_param = gp.Prior_Par_Creator.create("name of prior")  
prior_param["parameter name"] = parameter value
...
prior_list.append(("name of hyperparameter", "name of prior", prior_param))


model = gp.NAME OF MODEL(rv, model_par)
model_y = model.model()

# REPEAT IF NEEDED

# PLOT MODEL
plot.basic_data_plot(time, model_y, xlabel, ylabel, legend)




print('Step 2, Compute likelyhood and preditions and plot initial guess')

# INSTRUCTIONS

# LOGARITHMIC LIKELYHOOD:
#   COMPUTE AND APPLY POSTERIORS
loglik = gp.GPLikelyhood(time, rv, model_y, rv_err, hparam, model_par, "name of kernel")
logL = loglik.LogL(prior_list)

# PREDICTIONS:      
#   CREATE ARRAY OVER WHICH TO PREDICT AND COMPUTE
xpred = np.arange(time[0]-50., time[-1]+50., 0.5)
GP_rv, GP_err = loglik.predict(xpred)


# PLOT INITIAL PREDICTION AND ERRORS AGAINST THE RV-MODEL
plot.GP_plot(time_obs, rv_obs, rv_err, model_rv, pred_time, pred_rv, pred_err, residuals=False, legend=None, xlabel=None, ylabel=None)




print("")
print("Step 3, Running MCMC for optimisation")

# INSTRUCTIONS

# PICK MAX ITERATIONS FOR BURN IN
iterations = number

# RUN MCMC
# If Keplerian
Mstar = number
logL_chain, hparams, model_param, Mpl_list, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, time, rv, err_rv, model_par, "Name Kernel", "Name Model", prior_list, Mstar)

# If no Keplerian (and therefore no mass calculation)
logL_chain, hparams, model_param, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, time, rv, err_rv, model_par, "Name Kernel", "Name Model", prior_list)

# THIS INCLUDES MIXING AND CORNER PLOTS




print("")
print("Step 4, Final plotting")

# GET FINAL RESULTS FROM MCMC (REQUIRES THAT THE FINAL CORNER WORKS)

hparam_fin = gp.Par_Creator.create("name of kernel")
hparam_fin['hyperparameter name'] = gp.Parameter(value=value number, error=error, vary=True)
...

model_pa_fin = gp.Model_Par_Creator.create("Kepler")
model_par_fin['model parameter name'] = gp.Parameter(value=value number, error=error, vary=True)
...
model = gp.NAME OF MODEL(rv, model_par_fin)
model_fin = model.model()


loglik = gp.GPLikelyhood(time, rv, model_fin, rv_err, hparam_fin, model_par_fin, "name of kernel")
logL = loglik.LogL(prior_list)

xpred = np.arange(time[0]-50., time[-1]+50., 0.5)
GP_rv, GP_err = loglik.predict(xpred)


# PLOT INITIAL PREDICTION AND ERRORS AGAINST THE RV-MODEL
plot.GP_plot(time_obs, rv_obs, rv_err, model_rv, pred_time, pred_rv, pred_err, residuals=False, legend=None, xlabel=None, ylabel=None)









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:27:17 2022

@author: frescigno

Launcher for TOI-2134 analysis
"""


import numpy as np
#import GP_solar as gp
import GP_solar_multi as gp
import matplotlib.pyplot as plt
import plotting
from pathlib import Path
import auxiliary
#from MCMC_affine import run_MCMC as run
from MCMC_affine_multi import run_MCMC as run
from MCMC_affine_multi import get_model


inputfilenameDACE = 'TOI2134_HARPN_DRS-3-7_last'
myinput = Path('/Users/frescigno/Desktop/stellar RV data/TOI-2134/{}.rdb'.format(inputfilenameDACE))
#inputfile = open(myinput, "r")
DACE_TOI_all = np.genfromtxt(myinput, delimiter=None, skip_header=3)
# Skipp 2 header lines and the first row (bad datapoint)
JD = DACE_TOI_all[:,0] + 2400000
rv = DACE_TOI_all[:,1]
err_rv = DACE_TOI_all[:,2]

#last point has bad data
JD = JD[:-1]
rv = rv[:-1]
err_rv = err_rv[:-1]

rv_offset = np.mean(rv)
rv = rv - rv_offset



plotting.data_plot(JD, rv, err_y=err_rv, seasonbreaks=[2459550.])

hparam = gp.Par_Creator.create("QuasiPer")
# Radius of the sun in km
Rsun = 6.957e5
Rstar = 0.769698 * Rsun
vsini = 1 #km/s
Prot_s = np.pi*2 * Rstar / vsini #seconds
# convert between seconds and days
Prot_days = Prot_s / 86400
hparam['gp_per'] = gp.Parameter(value=Prot_days, error=1.)
print("################################################################,", Prot_days)

##### IDEA OF ROATION PERIOD, ANDREW, PERIODOGRAM, AND VSIN FOR UPPER LIMIT (FORMULA)
########### ASK ANDREW FOR RADIUS AND I
hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.05)
hparam['gp_explength'] = gp.Parameter(value=2*Prot_days, error=4.)
######### APRROXIMATE ACC2019 PAPER TO SAY FACULAE DOMINATED, ABOUT 2X ROTATION PERIOD
##### giles et al. 2017
hparam['gp_amp'] = gp.Parameter(value=10., error=2.)



prior_list=[]

prior_param3_b = gp.Prior_Par_Creator.create("Jeffrey")  
prior_param3_b["minval"] = 0.1
prior_param3_b["maxval"] = 500.
prior_list.append(("gp_explength", "Jeffrey", prior_param3_b))

prior_param2_b = gp.Prior_Par_Creator.create("Jeffrey")  
prior_param2_b["minval"] = 0.1
prior_param2_b["maxval"] = 500.
prior_list.append(("gp_per", "Jeffrey", prior_param2_b))

prior_param_b = gp.Prior_Par_Creator.create("Uniform")  
prior_param_b["minval"] = 0.
prior_param_b["maxval"] = 1.
prior_list.append(("gp_perlength", "Uniform", prior_param_b))

prior_param4_b = gp.Prior_Par_Creator.create("Gaussian")  
prior_param4_b["mu"] = 0.5
prior_param4_b["sigma"] = 0.05
prior_list.append(("gp_perlength", "Gaussian", prior_param4_b))

models_list = ["Kepler"]
Model_Par_Creator = gp.Model_Par_Creator()
model_par = Model_Par_Creator.create(models_list)
########## ASK ANDREW FOR TESS CURVE ##########

model_par['P'] = gp.Parameter(value=9.2292309, error=0.0000371)
model_par['K'] = gp.Parameter(value=2., error=1.)
###### COULD TAKE THE RMS ########
model_par['ecc'] = gp.Parameter(value=0., error=0.1, vary = True)
model_par['omega'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)

t_tr = 2459407.546692
t_0 = auxiliary.transit_to_periastron(t_tr, model_par['P'].value, model_par['ecc'].value, model_par['omega'].value)
model_par['t0'] = gp.Parameter(value=t_0, error=0.0000371)

model_y = get_model(models_list, JD, model_par, to_ecc=False)
'''model = gp.Keplerian(JD, model_par)
model_y = model.model()'''


prior_param5 = gp.Prior_Par_Creator.create("Gaussian")  
prior_param5["mu"] = 9.2292309
prior_param5["sigma"] = 0.0000371
prior_list.append(("P", "Gaussian", prior_param5))

prior_param7 = gp.Prior_Par_Creator.create("Gaussian")  
prior_param7["mu"] = t_0
prior_param7["sigma"] = 0.0002
prior_list.append(("t0", "Gaussian", prior_param7))




loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0]-10., JD[-1]+10., 0.5)
GP_rv, GP_err = loglik.predict(xpred)



#plotting.GP_plot(JD, rv, err_rv, model_y, xpred, GP_rv, GP_err, residuals=True)


smooth_model_y = get_model(models_list, xpred, model_par, to_ecc=False)
smooth_model_end = smooth_model_y+GP_rv

#plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_y, seasonbreaks=[2459550.])
#plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_y, model_y=model_y, residuals=True)

phase = auxiliary.phasefold(JD, model_par["P"].value, model_par["t0"].value)
smooth_phase = auxiliary.phasefold(xpred, model_par["P"].value, model_par["t0"].value)

import scipy.interpolate as interp
f = interp.interp1d(xpred, GP_rv, kind='cubic')
new_pred_y = f(JD)
planet_only_rv = (rv-new_pred_y)

#plotting.phase_plot(phase, planet_only_rv, err_rv, model_y=model_y, smooth_model_phase=smooth_phase, smooth_model_y=smooth_model_y, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')




iterations = 3
logL_chain, fin_hparams, fin_model_param = run(iterations, JD, rv, err_rv, hparam, "QuasiPer", model_par, models_list, prior_list, numb_chains=100)

print(fin_model_param)

hparam_names = ("gp_per", "gp_perlength", "gp_explength", "gp_amp")
model_param_names = ("P", "K", "ecc", "omega", "t0")
plotting.mixing_plot(iterations, 100, fin_hparams, "QuasiPer", fin_model_param, "Keplerian", logL_chain)

final_param_values = plotting.corner_plot(fin_hparams, "QuasiPer", fin_model_param, "Keplerian")



hparam2 = gp.Par_Creator.create("QuasiPer")
hparam2['gp_per'] = gp.Parameter(value=final_param_values[0], error=np.std(final_param_values[0]))
hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1], error=np.std(final_param_values[1]))
hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2], error=np.std(final_param_values[2]))
hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3], error=np.std(final_param_values[3]))

final_param_values[6], final_param_values[7] = auxiliary.to_ecc(final_param_values[6], final_param_values[7])

model_par2 = Model_Par_Creator.create(models_list)
model_par2['P'] = gp.Parameter(value=final_param_values[4], error=np.std(final_param_values[4]))
model_par2['K'] = gp.Parameter(value=final_param_values[5], error=np.std(final_param_values[5]))
model_par2['ecc'] = gp.Parameter(value=final_param_values[6], error=np.std(final_param_values[6]))
model_par2['omega'] = gp.Parameter(value=final_param_values[7], error=np.std(final_param_values[7]))
model_par2['t0'] = gp.Parameter(value=final_param_values[8], error=np.std(final_param_values[8]))

model = gp.Keplerian(JD, model_par2)
model_fin = model.model()
loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
GP_rv, GP_err = loglik.predict(xpred)

smooth_model = gp.Keplerian(xpred, model_par2)
smooth_model_y = smooth_model.model()
smooth_model_end = smooth_model_y+GP_rv


plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=True)
plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_y, model_y=model_fin, residuals=True)

phase = auxiliary.phasefold(JD, model_par2["P"].value, model_par2["t0"].value)
smooth_phase = auxiliary.phasefold(xpred, model_par2["P"].value, model_par2["t0"].value)

import scipy.interpolate as interp
f = interp.interp1d(xpred, GP_rv, kind='cubic')
new_pred_y = f(JD)
planet_only_rv = (rv-new_pred_y)

plotting.phase_plot(phase, planet_only_rv, err_rv, model_y=model_fin, smooth_model_phase=smooth_phase, smooth_model_y=smooth_model_y, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')

print()
print("Final LogL: ", logL)
print()
print("Final hyperparameters: ", hparam2)
print()
print("Final model parameters: ", model_par2)

plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_fin, seasonbreaks=[2459550.])


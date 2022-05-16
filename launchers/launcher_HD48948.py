#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:27:17 2022

@author: frescigno

Launcher for HD489848 analysis
"""


import numpy as np
import GP_solar_multi as gp
import matplotlib.pyplot as plt
import plotting
from pathlib import Path
import auxiliary
from MCMC_affine_multi import run_MCMC as run
from MCMC_affine_multi import get_model



inputfilenameDACE = 'HD48948_yarara_v1_timeseries'
myinput = Path('/Users/frescigno/Desktop/stellar RV data/HD-48948/{}.rdb'.format(inputfilenameDACE))
#inputfile = open(myinput, "r")
DACE_TOI_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
# Skipp 2 header lines and the first row (bad datapoint)
JD = DACE_TOI_all[:,0] + 2400000
rv = DACE_TOI_all[:,1]
err_rv = DACE_TOI_all[:,2]

rv_offset = np.mean(rv)
rv = rv - rv_offset



#plotting.data_plot(JD, rv, err_y=err_rv)


hparam = gp.Par_Creator.create("QuasiPer")
hparam['gp_per'] = gp.Parameter(value=55., error=10.)
#print("################################################################,", Prot_days)

##### IDEA OF ROATION PERIOD, ANDREW, PERIODOGRAM, AND VSIN FOR UPPER LIMIT (FORMULA)
########### ASK ANDREW FOR RADIUS AND I
hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.05)
#hparam['gp_explength'] = gp.Parameter(value=2*Prot_days, error=4.)
hparam['gp_explength'] = gp.Parameter(value=2*55., error=10.)
######### APRROXIMATE ACC2019 PAPER TO SAY FACULAE DOMINATED, ABOUT 2X ROTATION PERIOD
##### giles et al. 2017
hparam['gp_amp'] = gp.Parameter(value=np.nanstd(rv), error=2.)


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

'''prior_param4_b = gp.Prior_Par_Creator.create("Gaussian")  
prior_param4_b["mu"] = 0.5
prior_param4_b["sigma"] = 0.05
prior_list.append(("gp_perlength", "Gaussian", prior_param4_b))'''


models_list = ["Kepler", "Kepler"]
Model_Par_Creator = gp.Model_Par_Creator()
model_par = Model_Par_Creator.create(models_list)
########## ASK ANDREW FOR TESS CURVE ##########

model_par['P_0'] = gp.Parameter(value=7.34, error=0.05)
model_par['K_0'] = gp.Parameter(value=2., error=0.5)
###### COULD TAKE THE RMS ########
model_par['ecc_0'] = gp.Parameter(value=0., error=0.01, vary = True)
model_par['omega_0'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)

model_par['t0_0'] = gp.Parameter(value=JD[0], error=1.)


model_par['P_1'] = gp.Parameter(value=38.0, error=0.5)
model_par['K_1'] = gp.Parameter(value=1.7, error=0.5)
###### COULD TAKE THE RMS ########
model_par['ecc_1'] = gp.Parameter(value=0., error=0.01, vary = True)
model_par['omega_1'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)

model_par['t0_1'] = gp.Parameter(value=JD[0], error=1.)
print(model_par)

model_y = get_model(models_list, JD, model_par, to_ecc=False)




loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0]-10., JD[-1]+10., 1.)
GP_rv, GP_err = loglik.predict(xpred)



#plotting.GP_plot(JD, rv, err_rv, model_y, xpred, GP_rv, GP_err, residuals=True)

smooth_model_y = get_model(models_list, xpred, model_par, to_ecc=False)
smooth_model_end = smooth_model_y+GP_rv
#plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_y)



model_par_pl0 = Model_Par_Creator.create(["kepler"])
model_par_pl0['P'] = model_par['P_0']
model_par_pl0['K'] = model_par['K_0']
###### COULD TAKE THE RMS ########
model_par_pl0['ecc'] = model_par['ecc_0']
model_par_pl0['omega'] = model_par['omega_0']
model_par_pl0['t0'] = model_par['t0_0']
planet_0_model = get_model(["Kepler"], JD, model_par_pl0, to_ecc=False)
smooth_model_y0 = get_model(["kepler"], xpred, model_par_pl0, to_ecc=False)

model_par_pl1 = Model_Par_Creator.create(["kepler"])
model_par_pl1['P'] = model_par['P_1']
model_par_pl1['K'] = model_par['K_1']
###### COULD TAKE THE RMS ########
model_par_pl1['ecc'] = model_par['ecc_1']
model_par_pl1['omega'] = model_par['omega_1']
model_par_pl1['t0'] = model_par['t0_1']
planet_1_model = get_model(["Kepler"], JD, model_par_pl1, to_ecc=False)
smooth_model_y1 = get_model(["keple"], xpred, model_par_pl1, to_ecc=False)


#pl0 = plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv+smooth_model_y1, smooth_model_x=xpred, smooth_model_y=smooth_model_y0, model_y=planet_0_model, residuals=True)
#pl1 = plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv+smooth_model_y0, smooth_model_x=xpred, smooth_model_y=smooth_model_y1, model_y=planet_1_model, residuals=True)


phase0 = auxiliary.phasefold(JD, model_par["P_0"].value, model_par["t0_0"].value)
smooth_phase0 = auxiliary.phasefold(xpred, model_par["P_0"].value, model_par["t0_0"].value)

import scipy.interpolate as interp
f = interp.interp1d(xpred, GP_rv, kind='cubic')
new_pred_y0 = f(JD)
planet_only_rv0 = (rv-new_pred_y0-planet_1_model)

#plotting.phase_plot(phase0, planet_only_rv0, err_rv, model_y=planet_0_model, smooth_model_phase=smooth_phase0, smooth_model_y=smooth_model_y0, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')

phase1 = auxiliary.phasefold(JD, model_par["P_1"].value, model_par["t0_1"].value)
smooth_phase1 = auxiliary.phasefold(xpred, model_par["P_1"].value, model_par["t0_1"].value)
new_pred_y1 = f(JD)
planet_only_rv1 = (rv-new_pred_y1-planet_0_model)
#plotting.phase_plot(phase1, planet_only_rv1, err_rv, model_y=planet_1_model, smooth_model_phase=smooth_phase1, smooth_model_y=smooth_model_y1, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')


iterations = 500
logL_chain, fin_hparams, fin_model_param = run(iterations, JD, rv, err_rv, hparam, "QuasiPer", model_par, models_list, prior_list, numb_chains=400)

#print(fin_model_param)



#hparam_names = ("gp_per", "gp_perlength", "gp_explength", "gp_amp")
#model_param_names = ("P0", "K0", "ecc0", "omega0", "t00", "P1", "K1", "ecc1", "omega1", "t01")
plotting.mixing_plot(iterations, 400, fin_hparams, "QuasiPer", fin_model_param, models_list, logL_chain)

final_param_values = plotting.corner_plot(fin_hparams, "QuasiPer", fin_model_param, models_list)

print(np.mean(fin_hparams[:,0,:]))

hparam2 = gp.Par_Creator.create("QuasiPer")
hparam2['gp_per'] = gp.Parameter(value=np.mean(fin_hparams[:,0,:]))
hparam2['gp_perlength'] = gp.Parameter(value=np.mean(fin_hparams[:,1,:]))
hparam2['gp_explength'] = gp.Parameter(value=np.mean(fin_hparams[:,2,:]))
hparam2['gp_amp'] = gp.Parameter(value=np.mean(fin_hparams[:,3,:]))

fin_model_param[:,2,:], fin_model_param[:,3,:] = 0.0,np.pi*1/2
fin_model_param[:,7,:], fin_model_param[:,8,:] = 0.0,np.pi*1/2


model_par2 = Model_Par_Creator.create(models_list)
model_par2['P_0'] = gp.Parameter(value=np.mean(fin_model_param[:,0,:]))
model_par2['K_0'] = gp.Parameter(value=np.mean(fin_model_param[:,1,:]))
model_par2['ecc_0'] = gp.Parameter(value=np.mean(fin_model_param[:,2,:]))
model_par2['omega_0'] = gp.Parameter(value=np.mean(fin_model_param[:,3,:]))
model_par2['t0_0'] = gp.Parameter(value=np.mean(fin_model_param[:,4,:]))

model_par2['P_1'] = gp.Parameter(value=np.mean(fin_model_param[:,5,:]))
model_par2['K_1'] = gp.Parameter(value=np.mean(fin_model_param[:,6,:]))
model_par2['ecc_1'] = gp.Parameter(value=np.mean(fin_model_param[:,7,:]))
model_par2['omega_1'] = gp.Parameter(value=np.mean(fin_model_param[:,8,:]))
model_par2['t0_1'] = gp.Parameter(value=np.mean(fin_model_param[:,9,:]))


model_fin = get_model(models_list, JD, model_par2, to_ecc=False)
loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
GP_rv, GP_err = loglik.predict(xpred)


loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)
xpred = np.arange(JD[0]-10., JD[-1]+10., 1.)
GP_rv, GP_err = loglik.predict(xpred)



#plotting.GP_plot(JD, rv, err_rv, model_y, xpred, GP_rv, GP_err, residuals=True)

smooth_model_y = get_model(models_list, xpred, model_par2, to_ecc=False)
smooth_model_end = smooth_model_y+GP_rv
#plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_y)



model_par_pl0 = Model_Par_Creator.create(["kepler"])
model_par_pl0['P'] = model_par2['P_0']
model_par_pl0['K'] = model_par2['K_0']
###### COULD TAKE THE RMS ########
model_par_pl0['ecc'] = model_par2['ecc_0']
model_par_pl0['omega'] = model_par2['omega_0']
model_par_pl0['t0'] = model_par2['t0_0']
planet_0_model = get_model(["Kepler"], JD, model_par_pl0, to_ecc=False)
smooth_model_y0 = get_model(["kepler"], xpred, model_par_pl0, to_ecc=False)

model_par_pl1 = Model_Par_Creator.create(["kepler"])
model_par_pl1['P'] = model_par2['P_1']
model_par_pl1['K'] = model_par2['K_1']
###### COULD TAKE THE RMS ########
model_par_pl1['ecc'] = model_par2['ecc_1']
model_par_pl1['omega'] = model_par2['omega_1']
model_par_pl1['t0'] = model_par2['t0_1']
planet_1_model = get_model(["Kepler"], JD, model_par_pl1, to_ecc=False)
smooth_model_y1 = get_model(["keple"], xpred, model_par_pl1, to_ecc=False)


#pl0 = plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv+smooth_model_y1, smooth_model_x=xpred, smooth_model_y=smooth_model_y0, model_y=planet_0_model, residuals=True)
#pl1 = plotting.Keplerian_only_plot(JD, rv, err_rv, xpred, GP_rv+smooth_model_y0, smooth_model_x=xpred, smooth_model_y=smooth_model_y1, model_y=planet_1_model, residuals=True)


phase0 = auxiliary.phasefold(JD, model_par2["P_0"].value, model_par2["t0_0"].value)
smooth_phase0 = auxiliary.phasefold(xpred, model_par2["P_0"].value, model_par2["t0_0"].value)

import scipy.interpolate as interp
f = interp.interp1d(xpred, GP_rv, kind='cubic')
new_pred_y0 = f(JD)
planet_only_rv0 = (rv-new_pred_y0-planet_1_model)

plotting.phase_plot(phase0, planet_only_rv0, err_rv, model_y=planet_0_model, smooth_model_phase=smooth_phase0, smooth_model_y=smooth_model_y0, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')

phase1 = auxiliary.phasefold(JD, model_par2["P_1"].value, model_par2["t0_1"].value)
smooth_phase1 = auxiliary.phasefold(xpred, model_par2["P_1"].value, model_par2["t0_1"].value)
new_pred_y1 = f(JD)
planet_only_rv1 = (rv-new_pred_y1-planet_0_model)
plotting.phase_plot(phase1, planet_only_rv1, err_rv, model_y=planet_1_model, smooth_model_phase=smooth_phase1, smooth_model_y=smooth_model_y1, residuals=True, xlabel='Time [BJD]', ylabel='RV [km s-1]')

smooth_model_y = get_model(models_list, xpred, model_par2, to_ecc=False)
smooth_model_end = smooth_model_y+GP_rv
plotting.data_plot(JD, rv, err_y=err_rv, smooth_model_x=xpred, smooth_model_y=smooth_model_end, model_y=model_y)


'''smooth_model = gp.Keplerian(xpred, model_par2)
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
'''
print()
print("Final LogL: ", logL)
print()
print("Final hyperparameters: ", hparam2)
print()
print("Final model parameters: ", model_par2)




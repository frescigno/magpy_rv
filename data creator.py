#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:43:40 2022

@author: frescigno
"""

# to create fake data

import numpy as np
import GP_solar as gp
import matplotlib.pyplot as plt
import plotting
from pathlib import Path


'''time = np.arange(10,190,1.)
rv = np.ones_like(time)*5.
err_rv = np.ones_like(time)*0.001


hparam = gp.Par_Creator.create("QuasiPer")

hparam['gp_per'] = gp.Parameter(value=70., error=3.) 
hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.01)
hparam['gp_explength'] = gp.Parameter(value=200., error=1.)
hparam['gp_amp'] = gp.Parameter(value=10., error=0.1)

prior_list=[]

model_par = gp.Model_Par_Creator.create("No")
model_par['no'] = gp.Parameter(value=0., error=0.01, vary=False)

model = gp.No_Model(time, model_par)
model_y = model.model()

loglik = gp.GPLikelyhood(time, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
logL = loglik.LogL(prior_list)

xpred = np.arange(time[0]-10., time[-1]+10., 0.5)

GP_rv, GP_err = loglik.predict(xpred)

plt.plot(xpred, GP_rv)'''

inputfilename = 'Kepler-21_HARPN_DRS-3-7'
myinput = Path('simulated_data/{}.rdb'.format(inputfilename))
#inputfile = open(myinput, "r")
kepler_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
JD = kepler_all[:,0]
rv = kepler_all[:,1]

rv_offset = np.mean(rv)
rv = rv - rv_offset

plt.scatter(JD, rv)

model_par = gp.Model_Par_Creator.create("Kepler")
model_par['P'] = gp.Parameter(value=2.78574, error=0.0001)
model_par['K'] = gp.Parameter(value=2.12, error=0.66)
model_par['ecc'] = gp.Parameter(value=0.007, error=0.1)
model_par['omega'] = gp.Parameter(value=0.0349066, error=0.1)
model_par['t0'] = gp.Parameter(value=2456798.718, error=0.2*100)

model = gp.Keplerian(JD, model_par)
model_y = model.model()

plt.scatter(JD, model_y, color='orange')


act_rv = rv - model_y
data = np.column_stack((JD,act_rv))

filename = 'K-21'
myfile = Path('simulated_data/{}.rdb'.format(filename))
file = open(myfile, "wb+")

np.savetxt(file, data)
file.close()
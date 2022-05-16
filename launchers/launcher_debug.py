"""
Launcher for GP_solar and MCMC, also calls plotting
Used for debugging
Currently only K-21 from Lopez-Morales et al. 2016

Author: Federica Rescigno
Version: 27-01-2022
"""

import numpy as np
import GP_solar as gp
from MCMC import run_MCMC
import matplotlib.pyplot as plt
import plotting
from pathlib import Path
import auxiliary
from MCMC_affine import run_MCMC as run



K_21 = False

if K_21 is True:
    myinput = Path('simulated_data/K-21.rdb')
    #inputfile = open(myinput, "r")
    K21_all = np.genfromtxt(myinput)
    JD = K21_all[:,0]
    rv = K21_all[:,1]
    err_rv = np.ones_like(rv)*0.5
    
    plt.scatter(JD, rv)
    plt.title("Data")
    plt.show()
    
    hparam = gp.Par_Creator.create("QuasiPer")
    hparam['gp_per'] = gp.Parameter(value=12.60, error=0.1) 
    hparam['gp_perlength'] = gp.Parameter(value=0.42, error=0.05)
    hparam['gp_explength'] = gp.Parameter(value=24.04, error=0.1)
    hparam['gp_amp'] = gp.Parameter(value=6.7, error=1.)
    
    
    prior_list=[]
    '''prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] = 0.5
    prior_param["sigma"] = 0.05
    prior_list.append(("gp_perlength", "Gaussian", prior_param))
    
    prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 14.83
    prior_param2["sigma"] = 1.
    prior_list.append(("gp_per", "Gaussian", prior_param2))
    
    prior_param3 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param3["mu"] = 24.
    prior_param3["sigma"] = 1.
    prior_list.append(("gp_explength", "Gaussian", prior_param3))'''


    model_par = gp.Model_Par_Creator.create("No")
    model_par['no'] = gp.Parameter(value=0., error=0.1, vary=False)
    model = gp.No_Model(JD, model_par)
    model_y = model.model()
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.ylim(-20,10)
    plt.show()

    
    iterations = 50
    logL_chain, hparams, model_param, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, JD, rv, err_rv, model_par, "QuasiPer", "No", prior_list)


    hparam2 = gp.Par_Creator.create("QuasiPer")
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1])
    hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2])
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3])
    
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)



K_21_full = False

if K_21_full is True:
    
    inputfilename = 'Kepler-21_HARPN_DRS-3-7'
    myinput = Path('simulated_data/{}.rdb'.format(inputfilename))
    #inputfile = open(myinput, "r")
    kepler_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
    JD = kepler_all[:,0]
    rv = kepler_all[:,1]

    rv_offset = np.mean(rv)
    rv = rv - rv_offset
    
    err_rv = np.ones_like(rv)*0.5
    
    plt.scatter(JD, rv)
    plt.title("Data")
    plt.show()
    
    hparam = gp.Par_Creator.create("QuasiPer")
    hparam['gp_per'] = gp.Parameter(value=12.60, error=1.) 
    hparam['gp_perlength'] = gp.Parameter(value=0.42, error=0.1)
    hparam['gp_explength'] = gp.Parameter(value=24.04, error=1.)
    hparam['gp_amp'] = gp.Parameter(value=6.7, error=1.)
    
    
    prior_list=[]
    prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] = 0.42
    prior_param["sigma"] = 0.05
    prior_list.append(("gp_perlength", "Gaussian", prior_param))
    
    '''prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 14.83
    prior_param2["sigma"] = 1.
    prior_list.append(("gp_per", "Gaussian", prior_param2))
    
    prior_param3 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param3["mu"] = 24.
    prior_param3["sigma"] = 1.
    prior_list.append(("gp_explength", "Gaussian", prior_param3))'''


    model_par = gp.Model_Par_Creator.create("Kepler")
    model_par['P'] = gp.Parameter(value=2.7858212, error=0.0001)
    model_par['K'] = gp.Parameter(value=2.12, error=0.66)
    model_par['K'] = gp.Parameter(value=2.12, error=1.)
    model_par['ecc'] = gp.Parameter(value=0.007, error=0.1, vary = False)
    #model_par['ecc'] = gp.Parameter(value=0., error=0.1, vary=True)
    model_par['omega'] = gp.Parameter(value=0.0349066, error=0.1, vary= False)
    model_par['t0'] = gp.Parameter(value=2456798.0432434715, error=0.2*100)

    model = gp.Keplerian(JD, model_par)
    model_y = model.model()

    prior_param4 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param4["mu"] = 2.7858212
    prior_param4["sigma"] = 0.0000032
    prior_list.append(("P", "Gaussian", prior_param4))
    
    '''prior_param5 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param5["mu"] = 2.1
    prior_param5["sigma"] = 0.1
    prior_list.append(("K", "Gaussian", prior_param5))'''
    
    prior_param6 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param6["mu"] = 2456798.0432434715
    prior_param6["sigma"] = 0.00085
    prior_list.append(("t0", "Gaussian", prior_param6))
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.ylim(-20,10)
    plt.show()

    
    iterations = 50
    logL_chain, hparams, model_param, Mpl_list, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, JD, rv, err_rv, model_par, "QuasiPer", "Kepler", prior_list)


    hparam2 = gp.Par_Creator.create("QuasiPer")
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1])
    hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2])
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3])
    
    
    model_par2 = gp.Model_Par_Creator.create("Kepler")
    model_par2['P'] = gp.Parameter(value=final_param_values[4])
    model_par2['K'] = gp.Parameter(value=final_param_values[5])
    model_par2['ecc'] = gp.Parameter(value=final_param_values[6])
    model_par2['omega'] = gp.Parameter(value=final_param_values[7])
    model_par2['t0'] = gp.Parameter(value=final_param_values[8])

    model = gp.Keplerian(JD, model_par2)
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)



K_78_full = False

if K_78_full is True:
    
    inputfilename = 'Kepler-78_HARPN_DRS-3-7'
    myinput = Path('simulated_data/{}.rdb'.format(inputfilename))
    #inputfile = open(myinput, "r")
    kepler_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
    JD = kepler_all[:,0]
    rv = kepler_all[:,1]
    err_rv = kepler_all[:,2]

    rv_offset = np.mean(rv)
    rv = rv - rv_offset

    
    plt.scatter(JD, rv)
    plt.title("Data")
    plt.show()
    
    hparam = gp.Par_Creator.create("QuasiPer")
    hparam['gp_per'] = gp.Parameter(value=12.74, error=0.06) 
    hparam['gp_perlength'] = gp.Parameter(value=0.47, error=0.05)
    hparam['gp_explength'] = gp.Parameter(value=17, error=1.)
    hparam['gp_amp'] = gp.Parameter(value=8.78, error=2.)
    
    
    prior_list=[]
    '''prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] = 0.47
    prior_param["sigma"] = 0.05
    prior_list.append(("gp_perlength", "Gaussian", prior_param))
    
    prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 14.83
    prior_param2["sigma"] = 1.
    prior_list.append(("gp_per", "Gaussian", prior_param2))
    
    prior_param3 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param3["mu"] = 24.
    prior_param3["sigma"] = 1.
    prior_list.append(("gp_explength", "Gaussian", prior_param3))
    
    prior_param4 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param4["mu"] = 8.7
    prior_param4["sigma"] = 1.
    prior_list.append(("gp_amp", "Gaussian", prior_param4))'''
    
    prior_param3_b = gp.Prior_Par_Creator.create("Jeffrey")  
    prior_param3_b["minval"] = 0.1
    prior_param3_b["maxval"] = 25.
    prior_list.append(("gp_explength", "Jeffrey", prior_param3_b))
    
    prior_param2_b = gp.Prior_Par_Creator.create("Jeffrey")  
    prior_param2_b["minval"] = 0.1
    prior_param2_b["maxval"] = 25.
    prior_list.append(("gp_per", "Jeffrey", prior_param2_b))
    
    prior_param_b = gp.Prior_Par_Creator.create("Uniform")  
    prior_param_b["minval"] = 0.
    prior_param_b["maxval"] = 1.
    prior_list.append(("gp_perlength", "Uniform", prior_param_b))
    



    model_par = gp.Model_Par_Creator.create("Kepler")
    model_par['P'] = gp.Parameter(value=0.355, error=0.001)
    model_par['K'] = gp.Parameter(value=1.87, error=0.2)
    model_par['ecc'] = gp.Parameter(value=0., error=0.1, vary = True)
    model_par['omega'] = gp.Parameter(value=np.pi/2, error=0.1, vary= True)
    
    t_tr = 2454953.95995
    t_0 = auxiliary.transit_to_periastron(t_tr, model_par['P'].value, model_par['ecc'].value, model_par['omega'].value)
    model_par['t0'] = gp.Parameter(value=t_0, error=0.0002)

    model = gp.Keplerian(JD, model_par)
    model_y = model.model()

    prior_param5 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param5["mu"] = 0.355
    prior_param5["sigma"] = 0.001
    prior_list.append(("P", "Gaussian", prior_param5))
    
    '''prior_param6 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param6["mu"] = 1.87
    prior_param6["sigma"] = 0.2
    prior_list.append(("K", "Gaussian", prior_param6))'''
    
    prior_param7 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param7["mu"] = t_0
    prior_param7["sigma"] = 0.0002
    prior_list.append(("t0", "Gaussian", prior_param7))
    
    '''prior_param8 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param8["mu"] = 0.
    prior_param8["sigma"] = 0.01
    prior_list.append(("ecc", "Gaussian", prior_param8))
    
    prior_param9 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param9["mu"] = np.pi/2
    prior_param9["sigma"] = 0.01
    prior_list.append(("omega", "Gaussian", prior_param9))'''
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.ylim(-20,10)
    plt.show()

    
    iterations = 100
    logL_chain, hparams, model_param, Mpl_list, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, JD, rv, err_rv, model_par, "QuasiPer", "Kepler", prior_list)


    hparam2 = gp.Par_Creator.create("QuasiPer")
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1])
    hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2])
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3])
    
    
    model_par2 = gp.Model_Par_Creator.create("Kepler")
    model_par2['P'] = gp.Parameter(value=final_param_values[4])
    model_par2['K'] = gp.Parameter(value=final_param_values[5])
    model_par2['ecc'] = gp.Parameter(value=final_param_values[6])
    model_par2['omega'] = gp.Parameter(value=final_param_values[7])
    model_par2['t0'] = gp.Parameter(value=final_param_values[8])

    model = gp.Keplerian(JD, model_par2)
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)




random = False

if random is True:
    '''myinput = Path('simulated_data/random.rdb')
    #inputfile = open(myinput, "r")
    random_all = np.genfromtxt(myinput)
    JD = random_all[:,0]
    rv = random_all[:,1]
    err_rv = np.ones_like(rv)*0.5'''
    
    JD = np.arange(0,200,0.5)
    rv = [-1.29078372, -0.63757162,  0.03400026,  0.71050644,  1.37718682,  2.01845981,
      2.61851391,  3.16194356,  3.63439174,  4.02316186,  4.31776332,  4.51036014,
      4.59609863,  4.57329785,  4.44349516,  4.21134784,  3.8843998 ,  3.47272919,
      2.98849891,  2.44543569,  1.85826641,  1.24214083,  0.6120695 , -0.01759645,
     -0.63362039, -1.22425093, -1.77949418, -2.291279  , -2.75351238, -3.16202796,
     -3.51443631, -3.80989085, -4.04878769, -4.23242127, -4.36262049, -4.44139087,
     -4.4705884 , -4.451649  , -4.38539408, -4.27192829, -4.11063929, -3.90030229,
     -3.63928478, -3.32583858, -2.95845949, -2.53628806, -2.05952082, -1.52979837,
     -0.95053695, -0.32717288,  0.33270608,  1.01936017,  1.72103721,  2.42425107,
      3.11418787,  3.7752192 ,  4.39149377,  4.94757333,  5.4290754 ,  5.82328517,
      6.11970106,  6.31048344,  6.3907823 ,  6.35892762,  6.21647501,  5.96810724,
      5.62140085,  5.18647382,  4.67553585,  4.10236748,  3.48175621,  2.82891915,
      2.15894078,  1.48625259,  0.82417786,  0.18456072, -0.42250629, -0.98884766,
     -1.50826052, -1.97646334, -2.3909465 , -2.75073843, -3.05610563, -3.30820871,
     -3.50873867, -3.65955939, -3.76238177, -3.81849344, -3.82856469, -3.79254646,
     -3.70967042, -3.57855388, -3.39740479, -3.16431413, -2.87761586, -2.53628806,
     -2.14036446, -1.69132282, -1.19241694, -0.64892128, -0.06826279,  0.53997834,
      1.16420782,  1.79109551,  2.40598124,  2.99338772,  3.53761194,  4.02336077,
      4.43639335,  4.76413274,  4.99621125,  5.12491882,  5.14553043,  5.05649628,
      4.85948712,  4.55929559,  4.1636026 ,  3.68262478,  3.12866457,  2.51558916,
      1.85826641,  1.17198735,  0.47190383, -0.22749203, -0.9128232 , -1.57219867,
     -2.19548614, -2.77447743, -3.30294418, -3.77658664, -4.19288424, -4.55086173,
     -4.85078929, -5.09383848, -5.28171852, -5.41631878, -5.49938282, -5.53223807,
     -5.51560164, -5.44947825, -5.33316022, -5.1653322 , -4.94427606, -4.66816317,
     -4.33541411, -3.94509971, -3.49735235, -2.99375416, -2.43766881, -1.8344859,
     -1.19175259, -0.51917409,  0.17152576,  0.86688295,  1.55209941,  2.21155624,
      2.82940531,  3.39020521,  3.87956396,  4.28475091,  4.59524239,  4.80317042,
      4.90365044,  4.89497183,  4.77864349,  4.55929559,  4.24444623,  3.84414923,
      3.37054456,  2.83733757,  2.25923528,  1.65136919,  1.02873322,  0.40566352,
     -0.20461656, -0.79036719, -1.34160432, -1.85026487, -2.31026212, -2.71743422,
     -3.06939444, -3.36529711, -3.60553743, -3.79140714, -3.92473063, -4.00750713,
     -4.04158457, -4.02838903, -3.96873036, -3.86269994, -3.70967042, -3.50840041,
     -3.25723912, -2.95441854, -2.59841305, -2.18834032, -1.72437249, -1.20812439,
     -0.64298514, -0.0343626 ,  0.61018515,  1.28094922,  1.96620943,  2.65251272,
      3.32507927,  3.96831563,  4.56640636,  5.10394984,  5.56660091,  5.9416827,
      6.21873219,  6.38994874,  6.45052172,  6.39882087,  6.23644174,  5.96810724,
      5.60143413,  5.14658057,  4.61579643,  4.02290219,  3.38272508,  2.71052161,
      2.02141528,  1.32987608,  0.64926527, -0.00853571, -0.63339768 -1.21710931,
     -1.75343274, -2.23805239, -2.66842557, -3.04354871, -3.36365744, -3.62988268,
     -3.843887  , -4.00750713, -4.1224282 , -4.18991348, -4.21061035, -4.18444834,
     -4.11063929, -3.98778224, -3.8140685 , -3.5875741 , -3.30661969, -2.9701718,
     -2.57825432, -2.13233695, -1.6356672 , -1.09351502, -0.51330465,  0.0953846,
      0.72095756,  1.35008138,  1.96809137,  2.55950398,  3.10860811,  3.6001008,
      4.01972963,  4.35490439,  4.59524239,  4.73301694,  4.76348478,  4.68507624,
      4.49944069,  4.21134784,  3.82845427,  3.3609508 ,  2.82111276,  2.22277888,
      1.58078735,  0.91039831,  0.22673161, -0.45575369, -1.12371459, -1.7652951,
     -2.37039874, -2.93085394, -3.44046968, -3.89498418, -4.29191537, -4.63032702,
     -4.91052872, -5.13373172, -5.30168525, -5.41631878, -5.47941609, -5.49234483,
     -5.45586221, -5.37001296, -5.23412909, -5.04693467, -4.80675056, -4.51178666,
     -4.16050151, -3.75200328, -3.28646095, -2.76549251, -2.19249659, -1.57289685,
     -0.91427352, -0.22636381,  0.47907758,  1.18855692,  1.88724774,  2.55950398,
      3.18945174,  3.76162525,  4.26160962,  4.67665279,  4.99621125,  5.21239877,
      5.32031416,  5.3182318 ,  5.20764732,  4.99317933,  4.6823361 ,  4.28516337,
      3.81379482,  3.28193131,  2.70427715,  2.09596293,  1.47198348,  0.84667765,
      0.2332733 , -0.35648345, -0.91260049, -1.4270049 , -1.89359841, -2.30820586,
     -2.66842557, -2.97339523, -3.22349177, -3.4199871 , -3.56468419, -3.65955939,
     -3.70643624, -3.70671506, -3.66117855, -3.56988966, -3.43219135, -3.24681136,
     -3.0120669 , -2.72615689, -2.38752166, -1.99524389, -1.5494599 , -1.05174788,
     -0.50545964,  0.08403494,  0.70921628,  1.36041451,  2.02594885,  2.69240596,
      3.34504599,  3.96831563,  4.54643964,  5.06405659,  5.50686148,  5.86221741,
      6.11970106,  6.2715512 ,  6.31299622,  6.24244436,  6.06152915,  5.77501081,
      5.39054273,  4.91831892,  4.3706242 ,  3.76131314,  3.10524602,  2.41771133,
      1.71386346,  1.00820211,  0.31411693, -0.35648345, -0.99344412, -1.58852935,
     -2.13547839, -2.62995427, -3.06939444, -3.45277706, -3.78032116, -4.05314265,
     -4.27289083, -4.44139087, -4.56031806, -4.63092762, -4.65386061, -4.62904208,
     -4.55568116, -4.43237598, -4.25731876, -4.02858823, -3.74450955, -3.40405554,
     -3.00725815, -2.55559692, -2.05233092, -1.50274338, -0.91260049]
    err_rv = np.ones_like(JD)*0.5
    
    plt.scatter(JD, rv)
    plt.title("Data")
    plt.show()
    
    hparam = gp.Par_Creator.create("QuasiPer")
    hparam['gp_per'] = gp.Parameter(value=40., error=3.) 
    hparam['gp_perlength'] = gp.Parameter(value=0.5, error=0.01)
    hparam['gp_explength'] = gp.Parameter(value=200., error=1.)
    hparam['gp_amp'] = gp.Parameter(value=10., error=0.1)
    
    
    prior_list=[]
    prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] = 0.5
    prior_param["sigma"] = 0.01
    prior_list.append(("gp_perlength", "Gaussian", prior_param))
    
    '''prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 70.
    prior_param2["sigma"] = 1.
    prior_list.append(("gp_per", "Gaussian", prior_param2))'''
    
    prior_param3 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param3["mu"] = 200.
    prior_param3["sigma"] = 1.
    prior_list.append(("gp_explength", "Gaussian", prior_param3))
    
    prior_param4 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param4["mu"] = 10.
    prior_param4["sigma"] = 0.1
    prior_list.append(("gp_amp", "Gaussian", prior_param4))
    

    model_par = gp.Model_Par_Creator.create("No")
    model_par['no'] = gp.Parameter(value=0., error=0.1, vary=False)
    model = gp.No_Model(JD, model_par)
    model_y = model.model()
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.ylim(-20,10)
    plt.show()

    
    iterations = 50
    logL_chain, hparams, model_param, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, JD, rv, err_rv, model_par, "QuasiPer", "No", prior_list)


    hparam2 = gp.Par_Creator.create("QuasiPer")
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1])
    hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2])
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3])
    
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)




cos_only = False

if cos_only is True:
    
    JD = np.arange(0,200,0.5)
    A = 5.
    P = 10.
    pi = np.pi
    B = 2.*pi/P
    rv = A * np.cos(B*JD)
    err_rv = np.ones_like(JD)*0.5
    
    
    hparam = gp.Par_Creator.create("Cosine") 
    hparam['gp_amp'] = gp.Parameter(value=5., error=0.1)
    hparam['gp_per'] = gp.Parameter(value=10., error=0.2)
    
    
    prior_list=[]
    '''prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] =10
    prior_param["sigma"] = 0.1
    prior_list.append(("gp_per", "Gaussian", prior_param))
    
    prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 5.
    prior_param2["sigma"] = 0.1
    prior_list.append(("gp_amp", "Gaussian", prior_param2))'''
    

    

    model_par = gp.Model_Par_Creator.create("No")
    model_par['no'] = gp.Parameter(value=0., error=0.1, vary=False)
    model = gp.No_Model(JD, model_par)
    model_y = model.model()
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "Cosine")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.show()

    
    iterations = 50
    logL_chain, hparams, model_param, iterations, burnin, final_param_values = run_MCMC(iterations, hparam, JD, rv, err_rv, model_par, "Cosine", "No", prior_list)

    print(final_param_values)
    hparam2 = gp.Par_Creator.create("Cosine")
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[1])
    print(hparam2)
    
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "Cosine")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)



cos_affine = False


if cos_affine is True:
    
    JD = np.arange(0,200,0.5)
    A = 5.
    P = 10.
    pi = np.pi
    B = 2.*pi/P
    off = np.ones_like(JD)*5.
    #print(off)
    rv = off + A * np.cos(B*JD)
    #print(rv)
    err_rv = np.ones_like(JD)*0.5
    
    
    hparam = gp.Par_Creator.create("Cosine") 
    hparam['gp_amp'] = gp.Parameter(value=5.5, error=0.2)
    hparam['gp_per'] = gp.Parameter(value=11., error=0.1)
    

    prior_list=[]
    
    prior_param_uni1 = gp.Prior_Par_Creator.create("Uniform")
    prior_param_uni1["minval"] = 0
    prior_param_uni1["maxval"] = 20
    prior_list.append(("gp_per", "Uniform", prior_param_uni1))
    
    prior_param_uni2 = gp.Prior_Par_Creator.create("Uniform")
    prior_param_uni2["minval"] = 0
    prior_param_uni2["maxval"] = 50
    prior_list.append(("gp_amp", "Uniform", prior_param_uni2))
    
    '''
    prior_param = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param["mu"] =10
    prior_param["sigma"] = 0.5
    prior_list.append(("gp_per", "Gaussian", prior_param))'''
    '''
    prior_param2 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param2["mu"] = 5.
    prior_param2["sigma"] = 0.1
    prior_list.append(("gp_amp", "Gaussian", prior_param2))'''
    

    

    model_par = gp.Model_Par_Creator.create("Offset")
    model_par['offset'] = gp.Parameter(value=5.5, error=0.1, vary=True)
    
    
    prior_param_uni3 = gp.Prior_Par_Creator.create("Uniform")
    prior_param_uni3["minval"] = 0
    prior_param_uni3["maxval"] = 50
    prior_list.append(('offset', "Uniform", prior_param_uni3))
    
    print()
    print(model_par)
    
    model = gp.Offset(rv, model_par)
    #model = gp.No_Model(JD, model_par)
    model_y = model.model()
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "Cosine")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-50., JD[-1]+50., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.show()

    
    iterations = 2000
    logL_chain, fin_hparams, fin_model_param = run(iterations, JD, rv, err_rv, hparam, "Cosine", model_par, "Offset", prior_list)
    
    hparam_names = ("gp_amp", "gp_per")
    model_param_names = ("offset")
    plotting.mixing_plot(iterations, 100, fin_hparams, hparam_names, fin_model_param, model_param_names, logL_chain)
    
    plotting.corner_plot(fin_hparams, hparam_names, fin_model_param, model_param_names)
    

    '''print(final_param_values)
    hparam2 = gp.Par_Creator.create("Cosine")
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[1])
    print(hparam2)
    
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "Cosine")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)'''




K_78_affine = True

if K_78_affine:
    inputfilename = 'Kepler-78_HARPN_DRS-3-7'
    myinput = Path('simulated_data/{}.rdb'.format(inputfilename))
    #inputfile = open(myinput, "r")
    kepler_all = np.genfromtxt(myinput, delimiter=None, skip_header=2)
    JD = kepler_all[:,0]
    rv = kepler_all[:,1]
    err_rv = kepler_all[:,2]

    rv_offset = np.mean(rv)
    rv = rv - rv_offset

    
    plt.scatter(JD, rv)
    plt.title("Data")
    plt.show()
    
    hparam = gp.Par_Creator.create("QuasiPer")
    hparam['gp_per'] = gp.Parameter(value=12.75, error=0.06) 
    hparam['gp_perlength'] = gp.Parameter(value=0.47, error=0.05)
    hparam['gp_explength'] = gp.Parameter(value=17, error=1.)
    hparam['gp_amp'] = gp.Parameter(value=8.78, error=2.)
    
    

    prior_list=[]
    
    prior_param3_b = gp.Prior_Par_Creator.create("Jeffrey")  
    prior_param3_b["minval"] = 0.1
    prior_param3_b["maxval"] = 25.
    prior_list.append(("gp_explength", "Jeffrey", prior_param3_b))
    
    prior_param2_b = gp.Prior_Par_Creator.create("Jeffrey")  
    prior_param2_b["minval"] = 0.1
    prior_param2_b["maxval"] = 25.
    prior_list.append(("gp_per", "Jeffrey", prior_param2_b))
    
    prior_param_b = gp.Prior_Par_Creator.create("Uniform")  
    prior_param_b["minval"] = 0.
    prior_param_b["maxval"] = 1.
    prior_list.append(("gp_perlength", "Uniform", prior_param_b))
    
    
    model_par = gp.Model_Par_Creator.create("Kepler")
    model_par['P'] = gp.Parameter(value=0.355, error=0.001)
    model_par['K'] = gp.Parameter(value=1.87, error=0.2)
    model_par['ecc'] = gp.Parameter(value=0., error=0.001, vary = True)
    model_par['omega'] = gp.Parameter(value=np.pi/2, error=0.001, vary= True)
    
    t_tr = 2454953.95995
    t_0 = auxiliary.transit_to_periastron(t_tr, model_par['P'].value, model_par['ecc'].value, model_par['omega'].value)
    model_par['t0'] = gp.Parameter(value=t_0, error=0.0002)

    model = gp.Keplerian(JD, model_par)
    model_y = model.model()

    prior_param5 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param5["mu"] = 0.355
    prior_param5["sigma"] = 0.001
    prior_list.append(("P", "Gaussian", prior_param5))
    
    prior_param7 = gp.Prior_Par_Creator.create("Gaussian")  
    prior_param7["mu"] = t_0
    prior_param7["sigma"] = 0.0002
    prior_list.append(("t0", "Gaussian", prior_param7))
    
    
    loglik = gp.GPLikelyhood(JD, rv, model_y, err_rv, hparam, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    xpred = np.arange(JD[0]-10., JD[-1]+10., 0.5)
    GP_rv, GP_err = loglik.predict(xpred)

    plt.figure(figsize=(15,15))
    plt.plot(xpred, GP_rv, color='orange')
    plt.fill_between(xpred, GP_rv+GP_err*10, GP_rv-GP_err*10, alpha=0.5, color='orange')
    plt.scatter(JD, rv-model_y, color='black')
    plt.title("RV-Model and GP")
    plt.show()

    
    iterations = 400
    logL_chain, fin_hparams, fin_model_param = run(iterations, JD, rv, err_rv, hparam, "QuasiPer", model_par, "Kepler", prior_list, numb_chains=400)
    
    hparam_names = ("gp_per", "gp_perlength", "gp_explength", "gp_amp")
    model_param_names = ("P", "K", "ecc", "omega", "t0")
    plotting.mixing_plot(iterations, 100, fin_hparams, hparam_names, fin_model_param, model_param_names, logL_chain)
    
    final_param_values = plotting.corner_plot(fin_hparams, hparam_names, fin_model_param, model_param_names)

    
    
    hparam2 = gp.Par_Creator.create("QuasiPer")
    hparam2['gp_per'] = gp.Parameter(value=final_param_values[0])
    hparam2['gp_perlength'] = gp.Parameter(value=final_param_values[1])
    hparam2['gp_explength'] = gp.Parameter(value=final_param_values[2])
    hparam2['gp_amp'] = gp.Parameter(value=final_param_values[3])
    
    
    model_par2 = gp.Model_Par_Creator.create("Kepler")
    model_par2['P'] = gp.Parameter(value=final_param_values[4])
    model_par2['K'] = gp.Parameter(value=final_param_values[5])
    model_par2['ecc'] = gp.Parameter(value=final_param_values[6])
    model_par2['omega'] = gp.Parameter(value=final_param_values[7])
    model_par2['t0'] = gp.Parameter(value=final_param_values[8])

    model = gp.Keplerian(JD, model_par2)
    model_fin = model.model()
    loglik = gp.GPLikelyhood(JD, rv, model_fin, err_rv, hparam2, model_par, "QuasiPer")
    logL = loglik.LogL(prior_list)
    GP_rv, GP_err = loglik.predict(xpred)
    
    
    y = plotting.GP_plot(JD, rv, err_rv, model_fin, xpred, GP_rv, GP_err, residuals=False)








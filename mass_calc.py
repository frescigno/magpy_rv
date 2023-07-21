import numpy as np

def mass_calc(P,K,S,C, Mstar):
    '''
    Parameters
    ----------
    model_param : 2d array
        Array of all the model parameter in the mcmc (with Sk and Ck instead of ecc and omega)
    Mstar : float
        Stellar mass in solar masses

    Returns
    -------
    Mpl_sini : float
        Minimum mass of the planet in Jupiter masses

    '''
    
    ecc = C**2 + S**2
    Mpl_sini = 4.9191*10**(-3) * K * np.sqrt(1-ecc**2) * P**(1/3) * Mstar**(2/3)
    
    return Mpl_sini*317.907

M0 = mass_calc(7.34,1.92,0.,0.01,0.69)
M1 = mass_calc(38.07,1.64,0.01,0.01,0.69)

Mtoi = mass_calc(94.5, 8.3-2.,0.04,0.52,0.76)

Mhd1 = mass_calc(38.02, 1.71,0.0,0.0,0.686)
Mhd1p = mass_calc(38.02+0.40, 1.71+0.35,0.0,0.0,0.686)
Mhd1m = mass_calc(38.02-0.46, 1.71-0.41,0.0,0.0,0.686)

Mhd0 = mass_calc(7.34, 1.88,0.01,0.0,0.686)
Mhd0p = mass_calc(7.34+0.02, 1.88+0.38,0.01,0.0,0.686)
Mhd0m = mass_calc(7.34-0.05, 1.88-0.37,0.01,0.0,0.686)

Mhd2 = mass_calc(700.40, 1.92,0.01,0.0,0.686)
Mhd2p = mass_calc(700.40+20.65, 1.92+0.79,0.01,0.0,0.686)
Mhd2m = mass_calc(700.40-20.17, 1.92-0.76,0.01,0.0,0.686)

'''print(Mhd1*317.907, Mhd1p*317.907, Mhd1m*317.907)
print(Mhd0*317.907, Mhd0p*317.907, Mhd0m*317.907)
print(Mhd2*317.907, Mhd2p*317.907, Mhd2m*317.907)
print(Mhd2, Mhd2p, Mhd2m)
'''
Mtoib = mass_calc(9.23, 3.3, 0.01, 0.2, 0.74)
Mtoic = mass_calc(95,8.97,0.22, 0.09, 0.74)
print(Mtoib*317.907, Mtoic*317.907)

ecc= np.sqrt((0.54)**2+(0.03)**2)
ecc1= np.sqrt((0.54+0.12)**2+(0.03+0.03)**2)
ecc2= np.sqrt((0.54+0.12)**2+(0.03-0.02)**2)
ecc3= np.sqrt((0.54-0.13)**2+(0.03+0.03)**2)
ecc4= np.sqrt((0.54-0.13)**2+(0.03-0.02)**2)

#print(ecc,ecc1,ecc2,ecc3,ecc4)

#print(M0, M1)

#print(M0*317.907, M1*317.907)

print("hello")
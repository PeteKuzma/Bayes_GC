from multinest_def_NC_one_calking import PyNM as PyNM

print("Loaded succesfully")
PMN=PyNM("NGC5139",35,outbase_add="5_Jul31_11_20_tightCMD")
import numpy as np
import scipy.stats

def GaussianPrior(r,mu,sigma):
	"""Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
	from math import sqrt
	from scipy.special import erfcinv
	if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
		return -1.0e32
	else:
		return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r))



def prior(cube, ndim, nparams):
	#cube[0] = GaussianPrior(cube[0],-3.234,0.039)  # Prior for x_pm,cl is between -10 to 10.
	#cube[1] = GaussianPrior(cube[1],-6.719,0.039)  # Prior for y_pm,cl is between -10 to 10.
	cube[0] = scipy.stats.norm(-3.234,0.039).ppf(cube[0])
	cube[1] = scipy.stats.norm(-6.719,0.039).ppf(cube[1])
	cube[2] = 10**(4*cube[2] - 3) # Prior for x_disp,cl is between 10^-4 to 10.
	cube[3] = 10**(4*cube[3] - 3) # Prior for y_disp,cl is between 10^-4 to 10.
	cube[4] = cube[4]*20 - 10  # Prior for x_pm,MW is between -10 to 10.
	cube[5] = cube[5]*20 - 10  # Prior for y_pm,MW is between -10 to 10.
	cube[6] = 10**(4*cube[6] - 3) # Prior for x_disp,MW is between 10^-4 to 10.
	cube[7] = 10**(4*cube[7] - 3) # Prior for y_disp,M@ is between 10^-4 to 10.
	cube[8] = cube[8] # Prior for core radius from the pummber model between 10^-4 to 10.
	cube[9] = cube[9]
	cube[10] = 2*np.pi*cube[10]-np.pi
	cube[11] = 10**(12*cube[11]-10)
	#cube[12] = 10**(4*cube[12] - 3) 
	cube[12] = np.pi*cube[12]-np.pi/2.
	cube[13] = 10**(10*cube[13]-5)
	#cube[14] = scipy.stats.norm(0.0372,0.0033).ppf(cube[14])
	#cube[15] = scipy.stats.norm(0.8168,0.019).ppf(cube[15])

PMN.PyMultinest_setup(prior,0,50000,0.0372,0.8168,select=False,existing=True,Fadd="_5deg",rmax=5,pm_sel="gnom",live_points=200)
#PMN.PyMultinest_run()
PMN.PyMultinest_results()
#PMN.PyMultinest_plots()


PMN.Membership_after_PyNM(10000,PMN.rad_sel)

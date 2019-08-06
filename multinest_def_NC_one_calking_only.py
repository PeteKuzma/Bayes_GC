#---------------------------------------------
#
#        PyMultinest definitions
#
#---------------------------------------------

#---------------------------------------------
# Import required modules
#---------------------------------------------
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table
import numpy as np
import os
from pygaia.errors.photometric import gMagnitudeError, bpMagnitudeError, rpMagnitudeError
from pygaia.errors.photometric import gMagnitudeErrorEoM, bpMagnitudeErrorEoM, rpMagnitudeErrorEoM
from pygaia.photometry.transformations import gminvFromVmini
import astropy.units as u
import numpy.random as rand
from astropy.coordinates import Angle
from astropy.coordinates.sky_coordinate import SkyCoord
import astropy.coordinates as coord
from astropy.units import Quantity
#from astroquery.gaia import Gaia
#from astroquery.vizier import Vizier
from astropy import coordinates
import _pickle as cPickle
#from dustmaps.sfd import SFDQuery
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from astropy.modeling import models, fitting
from matplotlib import ticker
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import scipy.optimize as sop
from astropy.modeling import models, fitting
import pymultinest
import progressbar as PB
import time as time
import json
from numpy import log, exp, pi, random, linalg, array,matrix, zeros, sqrt,log10, arange, rad2deg, isnan,where
# Ignore warnings from TAP queries

# ---------------------------------------------------
# Definitions
# ---------------------------------------------------
class PyNM:
	def __init__(self,cluster,radius,version="1",outbase_add=None):
		if outbase_add!=None:
			self.outbase_add=outbase_add
			self.outbase_name="{0}_{1}_pymn_out_".format(cluster,outbase_add)
		else:
			self.outbase_name="{0}_pymn_out_".format(cluster)
		self.rad_sel=radius
		self.cluster=cluster
		self.Parameters=["x_pm,cl","y_pm,cl","x_dsp,cl","y_dsp,cl","x_pm,MW","y_pm,MW","x_dsp,MW","y_dsp,MW","f_cl","theta","k"]
		self.N_params = len(self.Parameters)
		self.cluster_F=cluster

	def PyMultinest_setup(self,prior,inner_radii,sample_size,cr,tr,select=True,pm_sel="norm",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False):
		self.Prior=prior
		self.rmin=inner_radii/60.
		self.rmax=rmax
		self.Live_Points=live_points
		if Fadd!=None:
			self.cluster_F="{0}{1}".format(self.cluster,Fadd)
		print("Changing to cluster directory:\nGaia/{0}/\n".format(self.cluster_F))
		os.chdir(self.cluster_F)
		if existing==True and os.path.isfile("{0}_{1}_sample_bayes.fits".format(self.cluster,sample_size)) == True: 
			print("Load sample fits file:  \n{0}_{1}_sample_bayes.fits\n".format(self.cluster,sample_size))		
			hdu3=fits.open("{0}_{1}_sample_bayes.fits".format(self.cluster,sample_size))
			M2_d=Table(hdu3[1].data)
			M2=M2_d[M2_d['dist']>=(inner_radii/60.)]
			M2=M2[np.isnan(M2['w_iso'])==False]
		if select==False:
			print("Load fits file: \n{0}_bays_ready.fits\n".format(self.cluster))
			hdu3=fits.open("{0}_bays_ready.fits".format(self.cluster))
			M2_d=Table(hdu3[1].data)
			M2=M2_d[M2_d['dist']>=(inner_radii/60.)]
			M2=M2[np.isnan(M2['w_iso'])==False]	
		else:
			print("Load fits file: \n{0}_bays_ready.fits\n".format(self.cluster))
			hdu3=fits.open("{0}_bays_ready.fits".format(self.cluster))
			M2_d=Table(hdu3[1].data)
			M2=M2_d[M2_d['dist']>=(inner_radii/60.)]
			M2=M2[np.isnan(M2['w_iso'])==False]
		if select==True and existing==False:
			print("Selecting sample from full data set\n")
			m2=hdu3[1].data	
			M2=Table(m2[np.random.choice(m2.shape[0], sample_size, replace=False)])
			self.SAMP=M2
			M2.write("{0}_{1}_sample_bayes.fits".format(self.cluster,sample_size),format="fits",overwrite=True)
		else:
			print("Selecting full sample\n")
			M2=M2
		M2=M2[M2['dist']>=self.rmin]
		M2=M2[M2['dist']<=rmax]
		self.x_ps=M2['ra_g'] # Spatial position in x_projection.
		self.y_ps=M2['dec_g'] # Spatial position in y_projection.
		if pm_sel=="norm":
			print("Selecting Gaia PMs\n")
			self.x_pm=M2['pmra'] # Proper motion in ra_projection.
			self.y_pm=M2['pmdec'] # Proper motion in dec_projection.
		else:
			print("Selecting projected PMs\n")
			self.x_pm=M2['pmra_g'] # Proper motion in x_projection.
			self.y_pm=M2['pmdec_g'] # Proper motion in y_projection.
		self.cv_pmraer=M2['pmra_error']# Proper Motion Covariance Matrix elements - error in pmra
		self.cv_pmdecer=M2['pmdec_error'] # Proper motion err in pmdec
		self.cv_coeff=M2['pmra_pmdec_corr'] # Proper motion correlation factor between pmra and pmdec
		self.w_par=M2['w_iso']
		self.tr=tr
		self.cr=cr
		self.King=where(M2['dist']<=tr,self.L_sat_king(self.x_ps,self.y_ps,cr,tr),-1e-40)
		#self.cv_raer=M2['ra_error']
		#self.cv_deer=M2['dec_error']
		#self.cv_radeccov=M2['ra_dec_corr']


	def PyMultinest_run(self):
		print("Run PyMultiNest")
		try:
			tstart=time.time()
			pymultinest.run(self.loglike_ndisp, self.Prior, self.N_params,outputfiles_basename=self.outbase_name, \
			resume = False, verbose = True,n_live_points=self.Live_Points)
			json.dump(self.Parameters, open("{0}_params.json".format(self.outbase_name), 'w')) # save parameter names
			tend=time.time()
			print("time taken: {0}".format(tend-tstart))
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")
	
	def PyMultinest_results(self,setup="complete"):
		try:
			result = pymultinest.solve(LogLikelihood=self.loglike_ndisp, Prior=self.Prior, 
			n_dims=self.N_params, outputfiles_basename=self.outbase_name, verbose=True,n_live_points=self.Live_Points)
			print('parameter values:')
			for name, col in zip(self.Parameters, result['samples'].transpose()):
				print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")
		
	def PyMultinest_plots(self,setup="complete",save_fig=True):
		try:
			a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename=self.outbase_name)
			s = a.get_stats()
			plt.clf()
			p = pymultinest.PlotMarginalModes(a)
			plt.figure(figsize=(5*self.N_params, 5*self.N_params))
			#plt.subplots_adjust(wspace=0, hspace=0)
			for i in range(self.N_params):
				plt.subplot(self.N_params, self.N_params, self.N_params * i + i + 1)
				p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
				plt.ylabel("Probability")
				plt.xlabel(self.Parameters[i])
	
				for j in range(i):
					plt.subplot(self.N_params, self.N_params, self.N_params * j + i + 1)
					#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
					p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
					plt.xlabel(self.Parameters[i])
					plt.ylabel(self.Parameters[j])
			plt.tight_layout()
			if save_fig==True:
				plt.savefig("{0}_{1}_post_dist.pdf".format(self.cluster,self.outbase_name),format='pdf')
			else:
				plt.show()
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")		


	def L_pm_GC(self,x_g,y_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
		'''
		Max log-likelihood from the proper motions of the Satellite. The variables are:
		x_g = cube for proper motion in x-profection
		y_g = cube for proper motion in y-profection
		sx_g = cube for dispersion in x-profection
		sy_g = cube for dispersion in y-profection
		x = proper motion in R.A. or x tangental projection
		y = proper motion in Dec. or y tangental projection
		cv_pmr = covariance matrix entry which is the uncertainty on the proper motion in R.A. or x
		cv_pmd = covariance matrix entry which is the uncertainty on the proper motion in Dec. or y
		cv_coeff = correlation between proper motion values.
		'''
		mc=1/(2*np.pi*cv_pmraer*cv_pmdecer*np.sqrt(1-cv_coeff**2))*np.exp((-0.5/(1-cv_coeff**2))*\
		(((x_g-x_pm)/(cv_pmraer))**2+((y_g-y_pm)/(cv_pmdecer))**2-\
		((2*cv_coeff*(x_g-x_pm)*(y_g-y_pm))/(cv_pmraer*cv_pmdecer))))
		return mc



	def L_pm_GC_old(self,x_g,y_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
		'''
		Max log-likelihood from the proper motions of the Satellite. The variables are:
		x_g = cube for proper motion in x-profection
		y_g = cube for proper motion in y-profection
		sx_g = cube for dispersion in x-profection
		sy_g = cube for dispersion in y-profection
		x = proper motion in R.A. or x tangental projection
		y = proper motion in Dec. or y tangental projection
		cv_pmr = covariance matrix entry which is the uncertainty on the proper motion in R.A. or x
		cv_pmd = covariance matrix entry which is the uncertainty on the proper motion in Dec. or y
		cv_coeff = correlation between proper motion values.
		''' 
		mc=exp(-1/2.*log(4*(pi**2)*(cv_pmraer**2*cv_pmdecer**2-(cv_coeff*cv_pmraer*cv_pmdecer)**2))        -(0.5*(1./(1-cv_coeff**2))*(((x_g-x_pm)/(cv_pmraer))**2+((y_g-y_pm)/(cv_pmdecer))**2-                                       ((2*cv_coeff*(x_g-x_pm)*(y_g-y_pm))/(cv_pmraer*cv_pmdecer)))))
		return mc

	def L_sat_spat_pow(self,xt_g,yt_g,ah,rmin,rmax,gam):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model and a constanct. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-light radii
		c = constant for the background.
		rmin = minimum radius in degrees
		rmax = minimum radius in degrees
		'''
		r = sqrt(xt_g**2+yt_g**2)
		n1 = -2*r*(1+(r/ah)**2)**(-gam/2.)	
		n2 =-((r*(-ah**2 - r**2)*((1 + (r/ah)**2)**(-1 - gam/2.))*gam)/(ah**2))
		d1 =(-ah**2. - rmax**2.)*(1 + (rmax/ah)**2)**(-gam/2.)
		d2 = (ah**2. + rmin**2.)*(1 + (rmin/ah)**2)**(-gam/2.)
		mc = (n1 + n2)/(d1+d2)	
		return mc



	def L_sat_spatial(self,xt_g,yt_g,ah):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-mass radii
		'''

		mc=((2*sqrt(xt_g**2+yt_g**2))/(ah**2))/    ((1+((xt_g**2+yt_g**2)/ah**2))**2)
		return mc

	def L_sat_king(self,xt_g,yt_g,ah,rt):
		r=sqrt(xt_g**2+yt_g**2)
		mc=r *( 1/(r*r+ah*ah)+1./(ah*ah+rt*rt)-2/(sqrt(ah*ah+r*r)*sqrt(ah*ah+rt*rt)))/\
		(pi*((self.rmax**2+4*(ah-sqrt(ah**2+self.rmax**2))*sqrt(ah**2+rt**2))/(ah**2+rt**2)\
		+log(1+self.rmax**2/ah**2)))       
		return mc


	def L_sat_spat_PL(self,xt_g,yt_g,ah,rmin,rmax):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model and a constanct. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-light radii
		c = constant for the background.
		rmin = minimum radius in degrees
		rmax = minimum radius in degrees
		'''
		r = sqrt(xt_g**2+yt_g**2)
		mc = (2. * r * (ah**2+rmax**2) * (ah**2+rmin**2))/\
			((rmax**2 - rmin**2)*((ah**2+r**2)**2)) 		
		return mc

	def L_sat_spat_PLt(self,xt_g,yt_g,ah,rmin,rmax):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model and a constanct. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-light radii
		c = constant for the background.
		rmin = minimum radius in degrees
		rmax = minimum radius in degrees
		'''
		r = sqrt(xt_g**2+yt_g**2)
		mc = ( r * (ah**2+rmax**2) * (ah**2+rmin**2))/\
		((rmax**2 - rmin**2)*((ah**2+r**2)**2)*np.pi)
		return mc

	def L_sat_const(self,xt_g,yt_g):
		'''
		Constantly likelihood.
		'''
		r=np.sqrt(xt_g**2+yt_g**2)
		mc=2.*r/(self.rmax**2)
		return mc

	def L_sat_constt(self,xt_g,yt_g):
		'''
		Constant likelihood over 2 dimensions
		'''
		r=np.sqrt(xt_g**2+yt_g**2)
		mc=r/(self.rmax**2*np.pi)
		return mc



	def L_sat_grad(self,xt_g,yt_g,the,a,b):
		z=(np.sqrt(xt_g**2+yt_g**2)*a+b*(np.sqrt(xt_g**2+yt_g**2))*np.cos(np.arctan2(yt_g,xt_g)-the))/(np.pi*a*self.rmax*self.rmax)
		#z1=rmax-rmin + 0.5*(rmax**2 - rmin**2)*np.cos(np.arctan2(yt_g,xt_g)-the)
		return z

	def L_spat_plumC(self,xt_g,yt_g,ah,C):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-mass radii
		'''
		R=sqrt(xt_g**2+yt_g**2)
		N1=C*R-(ah**4*R)/((ah**2+R**2)**2)
		D1=np.pi*(d*self.rmax**2+((ah**2*self.rmax**2)/(ah**2+self.rmax**2)))
		#n1 = 2 * sqrt(xt_g**2+yt_g**2) * (C + (ah**4/((ah**2+(xt_g**2+yt_g**2))**2)))
		#d1 = C * (self.rmax**2 - self.rmin**2) + ah**4 * (- (1/(ah**2 + self.rmax**2))+(1/(ah**2 + self.rmin**2)))
		mc=N1/D1
		return mc

	def L_sat_quad(self,xt_g,yt_g,the,b):
		r=np.sqrt(xt_g**2+yt_g**2)
		mc=r*r*3*(2+b+b*np.cos(2*(np.arctan2(yt_g,xt_g)-the)))/(self.rmax**2*np.pi*(2+b)*2)
		return mc

	def L_sat_quad_x(self,xt_g,yt_g,the):
		r=np.sqrt(xt_g**2+yt_g**2)
		mc=r*2*np.cos(np.arctan2(yt_g,xt_g)-the)*np.cos(np.arctan2(yt_g,xt_g)-the)/(self.rmax**2*np.pi)
		return mc


	def L_sat_spat_IS_2(self,xt_g,yt_g,ah,rmin,rmax):
		'''
		Likelihood for the spatial distribution from the cluster based 
		on a Plumber model and a constanct. The variables are:
		xt_g = tangental projection of R.A.
		yt_g = tangental projection of Dec.
		ah = Half-light radii
		c = constant for the background.
		rmin = minimum radius in degrees
		rmax = minimum radius in degrees
		'''
		r = sqrt(xt_g**2+yt_g**2)
		mc = r/(((-(ah**2)/(sqrt(1+(rmax/ah)**2))+(ah**2)/(sqrt(1+(rmin/ah)**2))))\
		*((1+(r/ah)**2)**(3/2.)))	
		return mc

	def L_sat_spat_IS(self, xt_g,yt_g,ah,rmin,rmax):
		r=sqrt(xt_g**2+yt_g**2)
		mc = 2*r/\
		((ah**2+r**2)*(log(ah**2+rmax**2)-log(ah**2+rmin**2)))
		return mc

	def L_pm_MW(self,x_g,y_g,sx_g,sy_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
		'''
		Max log-likelihood from the proper motions of the cluster. The variables are:
		x_g = cube for proper motion in x-profection
		y_g = cube for proper motion in y-profection
		sx_g = cube for dispersion in x-profection
		sy_g = cube for dispersion in y-profection
		x = proper motion in R.A. or x tangental projection
		y = proper motion in Dec. or y tangental projection
		cv_pmr = covariance matrix entry which is the uncertainty on the proper motion in R.A. or x
		cv_pmd = covariance matrix entry which is the uncertainty on the proper motion in Dec. or y
		cv_coeff = correlation between proper motion values.
		'''
		expn=cv_pmdecer**2*(x_g-x_pm)**2-2*cv_coeff*cv_pmraer*cv_pmdecer*(x_g-x_pm)*(y_g-y_pm)+(cv_pmraer**2+sx_g**2)*(y_g-y_pm)**2+(x_g-x_pm)**2*sy_g**2
		expd=2*cv_pmraer**2*(sy_g**2-(-1+cv_coeff**2)*cv_pmdecer**2)+2*sx_g**2*(cv_pmdecer**2+sy_g**2)
		mc=(1/(2*pi*sqrt(cv_pmdecer**2*cv_pmraer**2-cv_coeff**2*cv_pmdecer**2*cv_pmraer**2\
		+cv_pmdecer**2*sx_g**2+sy_g**2*cv_pmraer**2+sx_g**2*sy_g**2)))*exp(-expn/expd)
		return mc

	def L_pm_MW22(self,x_g,y_g,sx_g,sy_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
		'''
		Max log-likelihood from the proper motions of the cluster. The variables are:
		x_g = cube for proper motion in x-profection
		y_g = cube for proper motion in y-profection
		sx_g = cube for dispersion in x-profection
		sy_g = cube for dispersion in y-profection
		x = proper motion in R.A. or x tangental projection
		y = proper motion in Dec. or y tangental projection
		cv_pmr = covariance matrix entry which is the uncertainty on the proper motion in R.A. or x
		cv_pmd = covariance matrix entry which is the uncertainty on the proper motion in Dec. or y
		cv_coeff = correlation between proper motion values.
		''' 
		mc=exp(-1/2.*log(4*(pi**2)*((cv_pmraer**2+sx_g**2)*(cv_pmdecer**2+sy_g**2)-(cv_coeff*cv_pmraer*cv_pmdecer)**2))        -(0.5*(((x_g-x_pm)**2*(cv_pmdecer**2+sy_g**2)-2*cv_coeff*cv_pmraer*cv_pmdecer*(x_g-x_pm)*(y_g-y_pm)+           (y_g-y_pm)**2*(cv_pmraer**2+sx_g**2))/               ((cv_pmraer**2+sx_g**2)*(cv_pmdecer**2+sy_g**2)-                cv_coeff**2*cv_pmraer**2*cv_pmdecer**2))))
		return mc


	def L_spat_stripe(self,sig,rot,x_ps,y_ps):
		'''
		Likelihood function for gaussian stripe in field of view.
		sig: width of strip
		rot: rotation about the axis of the cluster
		x_ps: Position of x
		y_psL Position of y.
		'''
		z= 1/(sig * np.sqrt(2 * np.pi)) * np.exp( -(((np.sqrt(x_ps**2+y_ps**2)*np.cos(np.arctan2(y_ps,x_ps)-rot)))**2 / (2 * sig**2) ))
		return z

	def L_spat_MVG(self,sx_g,sy_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
		'''
		Max log-likelihood from the proper motions of the cluster. The variables are:
		x_g = cube for proper motion in x-profection
		y_g = cube for proper motion in y-profection
		sx_g = cube for dispersion in x-profection
		sy_g = cube for dispersion in y-profection
		x = proper motion in R.A. or x tangental projection
		y = proper motion in Dec. or y tangental projection
		cv_pmr = covariance matrix entry which is the uncertainty on the proper motion in R.A. or x
		cv_pmd = covariance matrix entry which is the uncertainty on the proper motion in Dec. or y
		cv_coeff = correlation between proper motion values.
		''' 
		mc=exp(-1/2.*log(4*(pi**2)*((cv_pmraer**2+sx_g**2)*(cv_pmdecer**2+sy_g**2)-(cv_coeff*cv_pmraer*cv_pmdecer)**2))\
		-(0.5*(((-x_pm)**2*(cv_pmdecer**2+sy_g**2)-2*cv_coeff*cv_pmraer*cv_pmdecer*(-x_pm)*(-y_pm)+ \
		(-y_pm)**2*(cv_pmraer**2+sx_g**2))/  \
		((cv_pmraer**2+sx_g**2)*(cv_pmdecer**2+sy_g**2)-    \
		cv_coeff**2*cv_pmraer**2*cv_pmdecer**2))))
		return mc

	def L_cmd_cl(self,sig_g,w_par):
		'''
		sig_g = estimating the spread of the cluster distribution 
		fromt the w-parameter.
		w_par
		'''
		likelihood = (1./(sqrt(2*pi*sig_g**2)))*exp(-(w_par**2/(2.*sig_g**2.)))
		return likelihood


	def L_cmd_mw(self,sig_g,cen_g,w_par):
		'''
		sig_g = estimating the spread of the cluster distribution 
		fromt the w-parameter.
		w_par
		'''
		likelihood = (1./(sqrt(2*pi*sig_g**2)))*exp(-(((w_par-cen_g)**2)/(2.*sig_g**2.)))
		return likelihood



	def loglike_ndisp(self,cube, ndim, nparams):
		x_cl,y_cl,sx_cl,sy_cl,x_g,y_g,sx_g,sy_g,fcl,the,c=\
		cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10]
		mc=(np.log(self.L_pm_MW(x_cl,y_cl,sx_cl,sy_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)*fcl*\
		self.King\
		+self.L_sat_grad(self.x_ps,self.y_ps,the,1,c)*\
		(1-fcl)*self.L_pm_MW(x_g,y_g,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)\
		)).sum()
		return mc



	def loglike_V2(self,cube, ndim, nparams):
		x_cl,y_cl,sx_cl,sy_cl,x_g,y_g,sx_g,sy_g,rc,fmw=\
		cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9]
		mc=(log(self.L_pm_MW(x_cl,y_cl,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)*(1-fmw)*\
		self.L_sat_spat_IS(self.x_ps,self.y_ps,rc,self.rmin,self.rmax)+\
			 fmw*self.L_pm_MW(x_g,y_g,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)\
			*self.L_sat_const(self.x_ps,self.y_ps))).sum()
		return mc

	def loglike_mem_V3(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample):
		'''
		Calculates the membership probability for an individual star
		'''
		gcsp=self.L_spat_stripe(sample[:,9],sample[:,8],x_ps,y_ps)
		mysp=self.L_sat_const(x_ps,y_ps)
		gcpm=self.L_pm_MW(sample[:,0],sample[:,1],sample[:,2],sample[:,3]\
		,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwpm=self.L_pm_MW(sample[:,4],sample[:,5],sample[:,6]\
		,sample[:,7],x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		fmw=sample[:,10]
		mc=((1-fmw)*gcsp*gcpm)/((1-fmw)*gcpm*gcsp+(fmw*mwpm*mysp))
		return np.mean(mc),np.std(mc)

	def loglike_mem_v2(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample):
		'''
		Calculates the membership probability for an individual star
		'''
		gcct=self.L_sat_quad(x_ps,y_ps,sample[:,12],sample[:,13])
		#gcsp=where(x_psself.L_sat_king(x_ps,y_ps,sample[:,14],sample[:,15])
		#gcsp=self.Kingi
		gcsp=where(sqrt(x_ps**2+y_ps**2)<=self.tr,self.L_sat_king(x_ps,y_ps,self.cr,self.tr),0)
		gcpm=self.L_pm_MW(sample[:,0],sample[:,1],sample[:,2],sample[:,3]\
		,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwpm=self.L_pm_MW(sample[:,4],sample[:,5],sample[:,6]\
		,sample[:,7],x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwsp=self.L_sat_grad(x_ps,y_ps,sample[:,10],1,sample[:,11])
		tspm=self.L_pm_GC(sample[:,0],sample[:,1],\
		x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		fcl=sample[:,8]
		fev=sample[:,9]
		mc_cl=(((fcl*fev)*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct))/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp))
		mc_co=(fcl*fev*gcsp*gcpm)/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp)
		mc_ts=(fcl*(1-fev)*tspm*gcct)/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp)
		return np.nanmean(mc_cl),np.nanstd(mc_cl),np.nanmean(mc_co),np.nanstd(mc_co),np.nanmean(mc_ts),np.nanstd(mc_ts)

	def loglike_mem(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample):
		'''
		Calculates the membership probability for an individual star
		'''
		gcct=self.L_sat_quad(x_ps,y_ps,sample[:,12],sample[:,13])
		#gcsp=where(x_psself.L_sat_king(x_ps,y_ps,sample[:,14],sample[:,15])
		gcsp=where(sqrt(x_ps**2+y_ps**2)<=self.tr,self.L_sat_king(x_ps,y_ps,self.cr,self.tr),0)
		gcpm=self.L_pm_MW(sample[:,0],sample[:,1],sample[:,2],sample[:,3]\
		,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwpm=self.L_pm_MW(sample[:,4],sample[:,5],sample[:,6]\
		,sample[:,7],x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwsp=self.L_sat_grad(x_ps,y_ps,sample[:,10],1,sample[:,11])
		tspm=self.L_pm_GC(sample[:,0],sample[:,1],\
		x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		fcl=sample[:,8]
		fev=sample[:,9]
		mc_cl=(((fcl*fev)*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct))/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp))
		mc_co=(fcl*fev*gcsp*gcpm)/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp)
		mc_ts=(fcl*(1-fev)*tspm*gcct)/\
		(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp)
		return np.nanmean(mc_cl),np.nanstd(mc_cl),np.nanmean(mc_co),np.nanstd(mc_co),np.nanmean(mc_ts),np.nanstd(mc_ts)


	def Membership_after_PyNM(self,sample_size,rad,gnom=True):
		'''
		Run this after PyMultiNest to calculate membership of all stars.
		'''
		try:
			f_in=fits.open("{0}_bays_ready_FULL.fits".format(self.cluster))
			f_data=Table(f_in[1].data)
			f_data=f_data[f_data['dist']<=self.rmax]
			x_ps=f_data['ra_g']
			y_ps=f_data['dec_g']
			if gnom==True:
				x_pm=f_data['pmra_g']
				y_pm=f_data['pmdec_g']
			else:
				x_pm=f_data['pmra']
				y_pm=f_data['pmdec']
			cv_pmraer=f_data['pmra_error']
			cv_pmdecer=f_data['pmdec_error']
			cv_coeff=f_data['pmra_pmdec_corr']
			w_par=f_data['w_iso']
			#self.King=where(f_data['dist']<=self.tr,self.L_sat_king(x_ps,y_ps,self.cr,self.tr),0)
			a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename= self.outbase_name)
			RWE=a.get_data()
			tot_sample=RWE[:,2:]
			zvf=zeros((len(f_data),6))
			print("Begin to calculate Membership probability.")
			for j in PB.progressbar(range(len(w_par))):
				zvf[j,0],zvf[j,1],zvf[j,2],zvf[j,3],zvf[j,4],zvf[j,5]=self.loglike_mem(x_ps[j],y_ps[j],x_pm[j],y_pm[j],\
				cv_pmraer[j],cv_pmdecer[j],cv_coeff[j],w_par[j],tot_sample)
			f_data['cl_mean']=zvf[:,0]
			f_data['cl_std']=zvf[:,1]
			f_data['co_mean']=zvf[:,2]
			f_data['co_std']=zvf[:,3]
			f_data['ts_mean']=zvf[:,4]
			f_data['ts_std']=zvf[:,5]
			#f_d3=f_data[f_data['mem_x']>=0.3]
			#f_d5=f_data[f_data['mem_x']>=0.5]
			#f_d7=f_data[f_data['mem_x']>=0.7]
			print("Writing to files.")
			f_data.write("{0}_mem_list_tot_{1}.fits".format(self.cluster,self.outbase_add),format="fits",overwrite=True)
			#f_d3.write("{0}_mem_list_0_3.fits".format(self.cluster),format="fits",overwrite=True)
			#f_d5.write("{0}_mem_list_0_5.fits".format(self.cluster),format="fits",overwrite=True)
			#f_d7.write("{0}_mem_list_0_7.fits".format(self.cluster),format="fits",overwrite=True)
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")


	def Membership_after_V2_PyNM(self,sample_size,rad):
		'''
		Run this after PyMultiNest to calculate membership of all stars.
		'''
		try:
			f_in=fits.open("{0}_bays_ready_FULL.fits".format(self.cluster))
			f_data=Table(f_in[1].data)
			f_data=f_data[f_data['dist']<=self.rmax]
			x_ps=f_data['ra_g']
			y_ps=f_data['dec_g']
			x_pm=f_data['pmra']
			y_pm=f_data['pmdec']
			cv_pmraer=f_data['pmra_error']
			cv_pmdecer=f_data['pmdec_error']
			cv_coeff=f_data['pmra_pmdec_corr']
			#=f_data['dist']
#self.King=where(f_data['dist']<=self.tr,self.L_sat_king(x_ps,y_ps,self.cr,self.tr),0)
			w_par=f_data['w_iso']
			a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename= self.outbase_name)
			RWE=a.get_data()
			ST=RWE.T[2:]
			print("Randomly sampling posterior distribution.")
			s1=np.random.choice(ST[0],sample_size)[np.newaxis].T
			s2=np.random.choice(ST[1],sample_size)[np.newaxis].T
			s3=np.random.choice(ST[2],sample_size)[np.newaxis].T
			s4=np.random.choice(ST[3],sample_size)[np.newaxis].T
			s5=np.random.choice(ST[4],sample_size)[np.newaxis].T
			s6=np.random.choice(ST[5],sample_size)[np.newaxis].T
			s7=np.random.choice(ST[6],sample_size)[np.newaxis].T
			s8=np.random.choice(ST[7],sample_size)[np.newaxis].T
			s9=np.random.choice(ST[8],sample_size)[np.newaxis].T
			s10=np.random.choice(ST[9],sample_size)[np.newaxis].T
			s11=np.random.choice(ST[10],sample_size)[np.newaxis].T
			s12=np.random.choice(ST[11],sample_size)[np.newaxis].T
			s13=np.random.choice(ST[12],sample_size)[np.newaxis].T
			s14=np.random.choice(ST[13],sample_size)[np.newaxis].T
			#s15=np.random.choice(ST[14],sample_size)[np.newaxis].T
			#s16=np.random.choice(ST[15],sample_size)[np.newaxis].T
			tot_sample=np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14),axis=1)#,s15,s16),axis=1)
			zvf=zeros((len(f_data),6))
			print("Begin to calculate Membership probability.")
			for j in range(len(w_par)):
				zvf[j,0],zvf[j,1],zvf[j,2],zvf[j,3],zvf[j,4],zvf[j,5]=self.loglike_mem(x_ps[j],y_ps[j],x_pm[j],y_pm[j],\
				cv_pmraer[j],cv_pmdecer[j],cv_coeff[j],w_par[j],tot_sample)
			f_data['cl_mean']=zvf[:,0]
			f_data['cl_std']=zvf[:,1]
			f_data['co_mean']=zvf[:,2]
			f_data['co_std']=zvf[:,3]
			f_data['ts_mean']=zvf[:,4]
			f_data['ts_std']=zvf[:,5]
			#f_d3=f_data[f_data['mem_x']>=0.3]
			#f_d5=f_data[f_data['mem_x']>=0.5]
			#f_d7=f_data[f_data['mem_x']>=0.7]
			print("Writing to files.")
			f_data.write("{0}_mem_list_tot_{1}.fits".format(self.cluster,self.outbase_add),format="fits",overwrite=True)
			#f_d3.write("{0}_mem_list_0_3.fits".format(self.cluster),format="fits",overwrite=True)
			#f_d5.write("{0}_mem_list_0_5.fits".format(self.cluster),format="fits",overwrite=True)
			#f_d7.write("{0}_mem_list_0_7.fits".format(self.cluster),format="fits",overwrite=True)
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")


'''
	def Nologlike_mem(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample):
	
		Calculates the membership probability for an individual star
	
		gcct=self.L_sat_const(x_ps,y_ps)
		gcsp=self.L_sat_spat_PL(x_ps,y_ps,sample[:,8],self.rmin,self.rmax)
		gcpm=self.L_pm_MW(sample[:,0],sample[:,1],sample[:,2],sample[:,3]\
		,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		mwpm=self.L_pm_MW(sample[:,4],sample[:,5],sample[:,6]\
		,sample[:,7],x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
		fcl=sample[:,9]
		fts=sample[:,10]
		mc_cl=((fcl*fts)*gcsp*gcpm+(fcl*(1-fts)*gcpm*gcct))/\
		(fcl*fts*gcsp*gcpm+(fcl*(1-fts)*gcpm*gcct)+(1-fcl*fts-fcl*(1-fts))*mwpm)
		mc_co=((fcl*fts)*gcsp*gcpm)/\
		(fcl*fts*gcsp*gcpm+(fcl*(1-fts)*gcpm*gcct))
		mc_ts=(fcl*(1-fts)*gcpm*gcct)/\
		(fcl*fts*gcsp*gcpm+(fcl*(1-fts)*gcpm*gcct))
		return np.mean(mc_cl),np.std(mc_cl),np.mean(mc_co),np.std(mc_co),np.mean(mc_ts),np.std(mc_ts)

	def NoMembership_after_PyNM(self,sample_size,rad):
		
		Run this after PyMultiNest to calculate membership of all stars.
		
		try:
			f_in=fits.open("{0}_bays_ready_FULL.fits".format(self.cluster))
			f_data=Table(f_in[1].data)
			x_ps=f_data['ra_g']
			y_ps=f_data['dec_g']
			x_pm=f_data['pmra']
			y_pm=f_data['pmdec']
			cv_pmraer=f_data['pmra_error']
			cv_pmdecer=f_data['pmdec_error']
			cv_coeff=f_data['pmra_pmdec_corr']
			w_par=f_data['w_iso']
			a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename= self.outbase_name)
			RWE=a.get_data()
			ST=RWE.T[2:]
			print("Randomly sampling posterior distribution.")
			s1=np.random.choice(ST[0],sample_size)[np.newaxis].T
			s2=np.random.choice(ST[1],sample_size)[np.newaxis].T
			s3=np.random.choice(ST[2],sample_size)[np.newaxis].T
			s4=np.random.choice(ST[3],sample_size)[np.newaxis].T
			s5=np.random.choice(ST[4],sample_size)[np.newaxis].T
			s6=np.random.choice(ST[5],sample_size)[np.newaxis].T
			s7=np.random.choice(ST[6],sample_size)[np.newaxis].T
			s8=np.random.choice(ST[7],sample_size)[np.newaxis].T
			s9=np.random.choice(ST[8],sample_size)[np.newaxis].T
			s10=np.random.choice(ST[9],sample_size)[np.newaxis].T
			s11=np.random.choice(ST[10],sample_size)[np.newaxis].T                       
			tot_sample=np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11),axis=1)
			zvf=zeros((len(f_data),6))
			print("Begin to calculate Membership probability.")
			for j in PB.progressbar(range(len(w_par))):
				zvf[j,0],zvf[j,1],zvf[j,2],zvf[j,3],zvf[j,4],zvf[j,5]=self.loglike_mem(x_ps[j],y_ps[j],x_pm[j],y_pm[j],\
					cv_pmraer[j],cv_pmdecer[j],cv_coeff[j],w_par[j],tot_sample)
			f_data['cl_mean']=zvf[:,0]
			f_data['cl_std']=zvf[:,1]
			f_data['co_mean']=zvf[:,2]
			f_data['co_std']=zvf[:,3]
			f_data['ts_mean']=zvf[:,4]
			f_data['ts_std']=zvf[:,5]
			#f_d3=f_data[f_data['mem_x']>=0.3]
			#f_d5=f_data[f_data['mem_x']>=0.5]
			#f_d7=f_data[f_data['mem_x']>=0.7]
			print("Writing to files.")
			f_data.write("{0}_mem_list_tot.fits".format(self.cluster),format="fits",overwrite=True)
		else:
			print("No")


	def NOMembership_after_PyNM_V2(self,sample_size,rad):
		
		Run this after PyMultiNest to calculate membership of all stars.
		
		try:
			f_in=fits.open("{0}_bays_ready_FULL.fits".format(self.cluster))
			f_data=Table(f_in[1].data)
			x_ps=f_data['ra_g']
			y_ps=f_data['dec_g']
			x_pm=f_data['pmra']
			y_pm=f_data['pmdec']
			cv_pmraer=f_data['pmra_error']
			cv_pmdecer=f_data['pmdec_error']
			cv_coeff=f_data['pmra_pmdec_corr']
			w_par=f_data['w_iso']
			a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename= self.outbase_name)
			RWE=a.get_data()
			ST=RWE.T[2:]
			print("Randomly sampling posterior distribution.")
			s1=np.random.choice(ST[0],sample_size)[np.newaxis].T
			s2=np.random.choice(ST[1],sample_size)[np.newaxis].T
			s3=np.random.choice(ST[2],sample_size)[np.newaxis].T
			s4=np.random.choice(ST[3],sample_size)[np.newaxis].T
			s5=np.random.choice(ST[4],sample_size)[np.newaxis].T
			s6=np.random.choice(ST[5],sample_size)[np.newaxis].T
			s7=np.random.choice(ST[6],sample_size)[np.newaxis].T
			s8=np.random.choice(ST[7],sample_size)[np.newaxis].T
			s9=np.random.choice(ST[8],sample_size)[np.newaxis].T
			s10=np.random.choice(ST[9],sample_size)[np.newaxis].T
			s11=np.random.choice(ST[10],sample_size)[np.newaxis].T
			tot_sample=np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11),axis=1)
			zvf=zeros((len(f_data),2))
			print("Begin to calculate Membership probability.")
			for j in PB.progressbar(range(len(w_par))):
				zvf[j,0],zvf[j,1]=self.loglike_mem_V2(x_ps[j],y_ps[j],x_pm[j],y_pm[j],\
					cv_pmraer[j],cv_pmdecer[j],cv_coeff[j],w_par[j],tot_sample)
			f_data['cl_mean']=zvf[:,0]
			f_data['cl_std']=zvf[:,1]
			f_d3=f_data[f_data['cl_mean']>=0.3]
			f_d5=f_data[f_data['cl_mean']>=0.5]
			f_d7=f_data[f_data['cl_mean']>=0.7]
			print("Writing to files.")
			f_data.write("{0}_mem_list_tot.fits".format(self.cluster),format="fits",overwrite=True)
			f_d3.write("{0}_mem_list_0_3.fits".format(self.cluster),format="fits",overwrite=True)
			f_d5.write("{0}_mem_list_0_5.fits".format(self.cluster),format="fits",overwrite=True)
			f_d7.write("{0}_mem_list_0_7.fits".format(self.cluster),format="fits",overwrite=True)
		except FileNotFoundError:
			print("Set-up not performed. Please run PyMultinest_setup.")		
	
'''
### other versions
'''
	def loglike_V1(self,cube, ndim, nparams):
		x_cl,y_cl,x_g,y_g,sx_g,sy_g,fmw,fev,fcl=\
		cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8]
		mc=(self.L_pm_GC(x_cl,y_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)*(1-fmw)*\
		where(sqrt(self.x_ps**2+self.y_ps**2)<=self.rad_sel/60.,self.L_sat_spatial(self.x_ps,self.y_ps,self.rad_sel/60.),1)+\
			 fmw*self.L_pm_MW(x_g,y_g,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)\
			 )).sum()
		return mc
'''

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
from multinest_base import PyNM

# ---------------------------------------------------
# Definitions
# ---------------------------------------------------
class PyMN_RUN(PyNM):
	def __init__(self,cluster,radius,prior,inner_radii,sample_size,cr,tr,select=True,pm_sel="norm",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None):
		PyNM.__init__(self,cluster,radius,prior,inner_radii,sample_size,cr,tr,select=True,pm_sel="norm",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None)
		self.King=where(self.dist<=tr,self.L_sat_king(self.x_ps,self.y_ps,self.cr,self.tr),1e-99)
		self.Parameters=["x_pm,cl","y_pm,cl","x_dsp,cl","y_dsp,cl","x_pm,MW","y_pm,MW","x_dsp,MW","y_dsp,MW","f_cl","f_ev","theta","k","theta2","k2"]
		self.N_params = len(self.Parameters)

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



	def L_pm_GCOLD(self,x_g,y_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
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



	def L_sat_king(self,xt_g,yt_g,ah,rt):
		r=sqrt(xt_g**2+yt_g**2)
		mc=r *( 1/(r*r+ah*ah)+1./(ah*ah+rt*rt)-2/(sqrt(ah*ah+r*r)*sqrt(ah*ah+rt*rt)))/\
		(pi*((self.rmax**2+4*(ah-sqrt(ah**2+self.rmax**2))*sqrt(ah**2+rt**2))/(ah**2+rt**2)\
		+log(1+self.rmax**2/ah**2)))       
		return mc




	def L_sat_grad(self,xt_g,yt_g,the,a,b):
		z=(np.sqrt(xt_g**2+yt_g**2)*a+b*(np.sqrt(xt_g**2+yt_g**2))*np.cos(np.arctan2(yt_g,xt_g)-the))/(np.pi*a*self.rmax*self.rmax)
		return z



	def L_sat_quad(self,xt_g,yt_g,the,b):
		r=np.sqrt(xt_g**2+yt_g**2)
		mc=r*(2+b+b*np.cos(2*(np.arctan2(yt_g,xt_g)-the)))/(self.rmax*self.rmax*np.pi*(2+b))
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

	def L_pm_MWOLD(self,x_g,y_g,sx_g,sy_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
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






	def loglike_ndisp(self,cube, ndim, nparams):
		x_cl,y_cl,sx_cl,sy_cl,x_g,y_g,sx_g,sy_g,fcl,fev,the,c,the2,k=\
		cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12],cube[13]
		mc=(np.log(self.L_pm_MW(x_cl,y_cl,sx_cl,sy_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)*fev*fcl*\
		self.King+(1-fev)*fcl*\
		self.L_sat_quad(self.x_ps,self.y_ps,the2,k)*self.L_pm_GC(x_cl,y_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)\
		+self.L_sat_grad(self.x_ps,self.y_ps,the,1,c)*\
		(1-fcl)*self.L_pm_MW(x_g,y_g,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)\
		)).sum()
		return mc




	def loglike_mem(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample):
		'''
		Calculates the membership probability for an individual star
		'''
		gcct=self.L_sat_quad(x_ps,y_ps,sample[:,12],sample[:,13])
		#gcsp=where(x_psself.L_sat_king(x_ps,y_ps,sample[:,14],sample[:,15])
		gcsp=where(self.dist<=tr,self.L_sat_king(self.x_ps,self.y_ps,self.cr,self.tr),1e-99)
		#gcsp=self.King
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



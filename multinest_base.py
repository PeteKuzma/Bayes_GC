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
	def __init__(self,cluster,radius,prior,inner_radii,sample_size,cr,tr,select=True,pm_sel="norm",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None):
		if outbase_add!=None:
			self.outbase_add=outbase_add
			self.outbase_name="{0}_{1}_pymn_out_".format(cluster,outbase_add)
		else:
			self.outbase_name="{0}_pymn_out_".format(cluster)
		self.rad_sel=radius
		self.cluster=cluster
		self.cluster_F=cluster
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
		self.King=where(M2['dist']<=tr,self.L_sat_king(self.x_ps,self.y_ps,cr,tr),0)
		#self.cv_raer=M2['ra_error']
		#self.cv_deer=M2['dec_error']
		#self.cv_radeccov=M2['ra_dec_corr']
		print("Changing to PyMultinest output directory:\n{0}".format{self.outbase_name})
		try:
		    os.chdir(self.outbase_name)
		except:
		    print("Output directory needs to be created. Creating now...")
		    os.mkdir(self.outbase_name)
		    os.chdir(self.outbase_name)

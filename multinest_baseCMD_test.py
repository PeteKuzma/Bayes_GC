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
    def __init__(self,cluster,prior,inner_radii,cr,tr,lh,survey,select=True,pm_sel="gnom",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None,pmsel=1,phot=1.6):
        if outbase_add!=None:
            self.outbase_add=outbase_add
            self.outbase_name="{0}_{1}_pymn_out_".format(cluster,outbase_add)
        else:
            self.outbase_name="{0}_pymn_out_".format(cluster)
        self.cluster=cluster
        self.cluster_F=cluster
        self.Prior=prior
        self.rmin=inner_radii/60.
        self.rmax=rmax
        self.Live_Points=live_points
        self.phot=phot
        self.psel=pmsel
        if Fadd!=None:
            self.cluster_F="{0}{1}".format(self.cluster,Fadd)
        print("Changing to cluster directory:\nGaia/{0}/\n".format(self.cluster_F))
        os.chdir(self.cluster_F)
        print("Load fits file: \n{0}_bays_ready.fits\n".format(self.cluster))
        hdu3=fits.open("{0}_bays_ready.fits".format(self.cluster))
        M2_d=Table(hdu3[1].data)
        M2=M2_d[M2_d['dist']>=(inner_radii/60.)]
        M2=M2[np.isnan(M2['w_iso'])==False]    
        print("Selecting full sample\n")
        M2=M2
        M2=M2[M2['dist']>=self.rmin]
        M2=M2[M2['dist']<=rmax]
        #if survey=="gaia":
        #    M2=M2[(M2['bp_0']-M2['rp_0'])<1.6]
        #if survey=="PS1":
        #    M2=M2[(M2['g_R0']-M2['i_R0'])<1.6]
        self.x_ps=M2['ra_g'] # Spatial position in x_projection.
        self.y_ps=M2['dec_g'] # Spatial position in y_projection.
        if pm_sel=="norm":
            print("Selecting Gaia PMs\n")
            self.x_pm=M2['pmra_g'] # Proper motion in ra_projection.
            self.y_pm=M2['pmdec_g'] # Proper motion in dec_projection.
        else:
            print("Selecting projected PMs\n")
            self.x_pm=M2['pmra_g_SRM'] # Proper motion in x_projection.
            self.y_pm=M2['pmdec_g_SRM'] # Proper motion in y_projection.
        self.cv_pmraer=M2['pmra_g_err']# Proper Motion Covariance Matrix elements - error in pmra
        self.cv_pmdecer=M2['pmdec_g_err'] # Proper motion err in pmdec
        self.cv_coeff=M2['pmra_pmdec_g_corr'] # Proper motion correlation factor between pmra and pmdec
        self.w_par=M2['w_iso']
        self.tr=tr
        self.cr=cr
        self.lh=lh
        self.dist=M2['dist']
        self.M2=M2
        hdu3.close()
        del M2
	#if survey=="gaia":
         #   self.gmag=M2['g_0']
            #self.w_par=self.w_par*sqrt(M2['bp_err']*M2['bp_err']+M2['rp_err']*M2['rp_err'])
            #self.colerr=sqrt(M2['bp_err']*M2['bp_err']+M2['rp_err']*M2['rp_err'])
          #  self.w_par=self.w_par
           # self.colerr=np.ones(np.shape(self.w_par))
        #elif survey=="PS1":
         #   self.gmag=M2["i_R0"]
          #  self.w_par=self.w_par
          #  self.colerr=np.ones(np.shape(self.w_par))            
        #else:
         #   print("NO SURVEY")
        #self.cv_raer=M2['ra_error']
        #self.cv_deer=M2['dec_error']
        #self.cv_radeccov=M2['ra_dec_corr']
        print("Changing to PyMultinest output directory:\n{0}".format(self.outbase_name))
        try:
            os.chdir(self.outbase_name)
        except FileNotFoundError:
            try:
                print("Output directory needs to be created. Creating now...")
                os.mkdir(self.outbase_name)
                os.chdir(self.outbase_name)
            except FileExistsError:
                print("created as part of the other thread.")
                os.chdir(self.outbase_name)

#---------------------------------------------
#
#       LIMEPY definitions and codes
#
#---------------------------------------------

#---------------------------------------------
# Import required modules
#---------------------------------------------
from limepy import limepy
from scipy.optimize import fmin
from pylab import sqrt, log10
import numpy
import cPickle as cPickle
import cPickle as pck
import glob as glob
import numpy as np
import os
import sys
import astropy.io.fits as fits
from astropy.table import Table






# ---------------------------------------------------
# Definitions
# ---------------------------------------------------
class LP_profiles:
	def __init__(self,cluster,weight,point_cut_range):
		self.cluster=cluster
		try:
			os.chdir("{0}".format(self.cluster))
			print "Moved into cluster directiory."
		except:
			print "Current directory not changed"
		self.weight_str=str(weight).replace(".","_")
		self.pickle_loads()
		self.grab_trager()
		self.cut_range=point_cut_range

	def grab_trager(self):
		file_in=fits.open("{0}_trager_s_{1}.fits".format(self.cluster,self.weight_str))
		self.tprof=Table(file_in[1].data)		
		
	def bootstrap_resample(self,X, n=None):
		""" Bootstrap resample an array_like
		Parameters
		----------
		X : array_like
		  data to resample
		n : int, optional
		  length of resampled array, equal to len(X) if n==None
		Results
		-------
		returns X_resamples
		"""
		if n == None:
			n = len(X)
		
		resample_i = numpy.floor(numpy.random.rand(n)*len(X)).astype(int)
		X_resample = X[resample_i]
		return X_resample


	def pickle_loads(self):
		'''
		Loads all required dictionaries and arrays to a pickled output.
		'''
		fopen=open("{0}_{1}_rp_data.pickle".format(self.cluster,self.weight_str),'rb')
		if sys.version_info[0]==3:
			self.ang=pck.load(fopen)
			self.dns=pck.load(fopen)
			self.dns_bck=pck.load(fopen)
			self.dns_norm_err=pck.load(fopen)
			self.dns_log_err=pck.load(fopen)
			self.dns_norm_ber=pck.load(fopen)
			self.dns_log_ber=pck.load(fopen)
			self.back_est=pck.load(fopen)
			self.back_err=pck.load(fopen)
		if sys.version_info[0]==2:
			self.ang=pck.load(fopen)
			self.dns=pck.load(fopen)
			self.dns_bck=pck.load(fopen)
			self.dns_norm_err=pck.load(fopen)
			self.dns_log_err=pck.load(fopen)
			self.dns_norm_ber=pck.load(fopen)
			self.dns_log_ber=pck.load(fopen)
			self.back_est=pck.load(fopen)
			self.back_err=pck.load(fopen)
		
	def leastsq(self,p, x, y, yerr):
		if p[0] <= 0 or p[0] > 20 or p[1] <= 0 or p[2] <=0 or p[3] <=0:
			return 9e99
		m = limepy(p[0], p[1], M=p[2], rh=p[3], project=True)
		# Interpolate surface density of model at x data
		ymod = numpy.interp(x, m.r, m.Sigma)
		return sum( (ymod - y)**2/yerr**2 )
	
	def leastsq_K(self,p, x, y, yerr):
		if p[0] <= 0 or p[0] > 20 or p[1] <=0 or p[2] <=0:
			return 9e99
		m = limepy(p[0], 1, M=p[1], rh=p[2], project=True)
		# Interpolate surface density of model at x data
		ymod = numpy.interp(x, m.r, m.Sigma)
		return sum( (ymod - y)**2/yerr**2 )
	
	def leastsq_W(self,p, x, y, yerr):
		if p[0] <= 0 or p[0] > 20 or p[1] <=0 or p[2] <=0:
			return 9e99
		m = limepy(p[0], 2, M=p[1], rh=p[2], project=True)
		# Interpolate surface density of model at x data
		ymod = numpy.interp(x, m.r, m.Sigma)
		return sum( (ymod - y)**2/yerr**2 )

	def leastsq_P(self,p, x, y, yerr):
		if p[0] <= 0 or p[1] <=0:
			return 9e99
		m = limepy(0.001, 3.499, M=p[0], rh=p[1], project=True)
		# Interpolate surface density of model at x data
		ymod = numpy.interp(x, m.r, m.Sigma)
		return sum( (ymod - y)**2/yerr**2 )


	def limepy_run(self,n_sam=1000):
		print "Appending and creating the radial profile array for LIMEPY."
		yProf=self.dns_bck[self.cut_range[0]:self.cut_range[1]]
		xProf=self.ang[self.cut_range[0]:self.cut_range[1]]
		yTrag=self.tprof['N_Scale'][0::5]
		xTrag=self.tprof['arcmin'][0::5]
		x=np.append(xProf,xTrag)
		y=np.append(yProf,yTrag)
		area = x**2
		Sigma = y
		self.king=dict()
		self.wils=dict()
		self.plum=dict()
		self.kings=dict()
		self.wilss=dict()
		kbest=[]
		wbest=[]
		pbest=[]
		self.plums=dict()		
		dy = np.sqrt(Sigma*area)/area
		print "MODEL FITTING"
		for i in range(15):
			k0=[i,1e5,3]
			w0=[i,1e5,3]
			p0=[1e5,3]
			try:
				self.king[i]=fmin(self.leastsq_K,k0,args=(x,y,dy),full_output=True)
				self.kings[i]=1
				kbest.append(self.king[i][1])
			except ValueError:
				print "Errorz"
				self.kings[i]=0
				kbest.append(9e99)
			try:
				self.wils[i]=fmin(self.leastsq_W,w0,args=(x,y,dy),full_output=True)
				self.wilss[i]=1
				wbest.append(self.wils[i][1])
			except ValueError:
				print "Errorz"
				self.wilss[i]=0	
				wbest.append(9e99)
			try:
				self.plum[i]=fmin(self.leastsq_P,p0,args=(x,y,dy),full_output=True)
				self.plums[i]=1
				pbest.append(self.plum[i][1])
			except ValueError:
				print "Errorz"
				self.plums[i]=0	
				pbest.append(9e99)
		F1=open("{0}_King_LIMEPY_{1}.pickle".format(self.cluster,self.weight_str),'wb')
		F2=open("{0}_Wilson_LIMEPY_{1}.pickle".format(self.cluster,self.weight_str),'wb')	
		F3=open("{0}_Plummer_LIMEPY_{1}.pickle".format(self.cluster,self.weight_str),'wb')
		cPickle.dump(self.king,F1)
		cPickle.dump(self.kings,F1)
		cPickle.dump(self.wils,F2)
		cPickle.dump(self.wilss,F2)
		cPickle.dump(self.plum,F3)
		cPickle.dump(self.plums,F3)
		self.kbest=kbest
		self.wbest=wbest
		self.pbest=pbest
		mnk,idk=min((kbest[i],i) for i in xrange(len(kbest)))
		mnw,idw=min((wbest[i],i) for i in xrange(len(wbest)))
		mnp,idp=min((pbest[i],i) for i in xrange(len(pbest)))
		k_lim=self.king[idk]
		w_lim=self.wils[idw]
		p_lim=self.plum[idp]
		cPickle.dump(k_lim,F1)
		cPickle.dump(w_lim,F2)
		cPickle.dump(p_lim,F3)
		F1.close()
		F2.close()
		F3.close()
		F1=open("{0}_King_BootLP_{1}.pickle".format(self.cluster,self.weight_str),'wb')
		F2=open("{0}_Wilson_BootLP_{1}.pickle".format(self.cluster,self.weight_str),'wb')
		F3=open("{0}_Plummer_BootLP_{1}.pickle".format(self.cluster,self.weight_str),'wb')
		self.boot_w=np.zeros((n_sam,5))
		self.boot_k=np.zeros((n_sam,5))
		self.boot_p=np.zeros((n_sam,5))
		print "Bootstrapping for uncertainties"
		xcomby=np.append(x[np.newaxis].T,y[np.newaxis].T,axis=1)
		xycomdy=np.append(xcomby,dy[np.newaxis].T,axis=1)
		print "Bootstrapping..."
		for i in range(n_sam):
			print "N = {0}".format(i)
			BXY=self.bootstrap_resample(xycomdy)
			x=BXY[:,0]
			y=BXY[:,1]
			dy=BXY[:,2]
			xk0=[k_lim[0][0],k_lim[0][1],k_lim[0][2]]
			xw0=[w_lim[0][0],w_lim[0][1],w_lim[0][2]]
			xp0=[p_lim[0][0],p_lim[0][1]]
			K_boot=fmin(self.leastsq_K,xk0,args=(x,y,dy))
			W_boot=fmin(self.leastsq_W,xw0,args=(x,y,dy))
			P_boot=fmin(self.leastsq_P,xp0,args=(x,y,dy))	
			BKm=limepy(K_boot[0], 1, M=K_boot[1], rh=K_boot[2], project=True)
			BWm=limepy(W_boot[0], 2, M=W_boot[1], rh=W_boot[2], project=True)
			BPm=limepy(0.001, 3.499, M=P_boot[0], rh=P_boot[1], project=True)
			self.boot_k[i,0]=K_boot[0]
			self.boot_k[i,1]=BKm.r0
			self.boot_k[i,2]=BKm.rt
			self.boot_k[i,3]=BKm.rv
			self.boot_k[i,4]=BKm.rh
			self.boot_w[i,0]=W_boot[0]
			self.boot_w[i,1]=BWm.r0
			self.boot_w[i,2]=BWm.rt
			self.boot_w[i,3]=BWm.rv
			self.boot_w[i,4]=BWm.rh
			self.boot_p[i,0]=0.001
			self.boot_p[i,1]=BPm.r0
			self.boot_p[i,2]=BPm.rt
			self.boot_p[i,3]=BPm.rv
			self.boot_p[i,4]=BPm.rh
		cPickle.dump(self.boot_k,F1)
		cPickle.dump(self.boot_w,F2)
		cPickle.dump(self.boot_p,F3)
		F1.close()
		F2.close()
		F3.close()

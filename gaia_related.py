# Prepare for Bayesian stats direct from Gaia and Panstarrs
#---------------------------------------------
#Import requires modules

import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table,vstack
import numpy as np
import os
from pygaia.errors.photometric import gMagnitudeError, bpMagnitudeError, rpMagnitudeError
from pygaia.errors.photometric import gMagnitudeErrorEoM, bpMagnitudeErrorEoM, rpMagnitudeErrorEoM
from pygaia.photometry.transformations import gminvFromVmini
import astropy.units as u
from astropy.coordinates import Angle
from astropy.coordinates.sky_coordinate import SkyCoord
import astropy.coordinates as coord
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy import coordinates
import _pickle as cPickle
from dustmaps.sfd import SFDQuery
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from astropy.modeling import models, fitting
from matplotlib import ticker
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree
import scipy.optimize as sop
from astropy.modeling import models, fitting
import pymultinest
import progressbar as PB
# Ignore warnings from TAP queries
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Get Gaia tables.
from astroquery.gaia import Gaia
tables = Gaia.load_tables(only_names=True)

# ---------------------------------------------------
# Definitions
# ---------------------------------------------------
class gaia:
	def __init__(self):
		Vizier.ROW_LIMIT=1000000000
		self.ruwe=Table(fits.open("ruwe_table.fits")[1].data)
		
	def get_gaia_data(self,cluster,rad,altname=False):
		'''
		Retrieving the data from Gaia.
		Input parameters:
		cluster - the cluster of interest
		rad - radius of interest from the cluster center
		'''
		print("Retreiving Gaia data.")
		orig_path=os.getcwd()
		if os.path.isdir("{0}".format(cluster)) == False:
			print("No folder for cluster. Creating now...")
			os.makedirs("{0}".format(cluster))
			os.chdir("{0}".format(cluster))
			print("Moving into cluster folder.\n")
		else:
			os.chdir("{0}".format(cluster))
			print("Moving into cluster folder.\n")	
		center = coordinates.SkyCoord.from_name(cluster)
		RA=center.ra.value
		DEC=center.dec.value
		job = Gaia.launch_job_async("SELECT *, distance( POINT('ICRS', {0},{1}), \
		POINT('ICRS', gaia.ra, gaia.dec)) as DIST \
		FROM gaiadr2.gaia_source as gaia \
		WHERE CONTAINS(POINT('ICRS',gaia.ra,gaia.dec),CIRCLE('ICRS',{0},{1},{2}))=1"\
		.format(RA,DEC,rad),dump_to_file=True,output_file="{0}_gaia.fits".format(cluster),\
		output_format="fits")
		os.chdir(orig_path)

	def get_PS1_data(self,cluster,rad):
		'''
		Retrieving the data from Panstarrs.
		Input parameters:
		cluster - the cluster of interest
		rad - radius of interest from the cluster center
		'''
		print("Retreiving Panstarrs data.")
		orig_path=os.getcwd()
		if os.path.isdir("{0}".format(cluster)) == False:
			print("No folder for cluster. Creating now...")
			os.makedirs("{0}".format(cluster))
			os.chdir("{0}".format(cluster))
			print("Moving into cluster folder.\n")
		else:
			os.chdir("{0}".format(cluster))
			print("Moving into cluster folder.\n")	
		center = coordinates.SkyCoord.from_name(cluster)
		RA=center.ra.value
		DEC=center.dec.value
		job = Vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,unit=(u.deg, u.deg),\
									frame='icrs'),radius=Angle(rad,"deg"),catalog=["II/349"])		
		WE=Table(job[0])
		WE.meta['description']=u'Panstarrs'
		WE.write('{0}_PS.fits'.format(cluster),format='fits')
		os.chdir(orig_path)		


	def stilts(self,cluster, tol_a = 2 ,join = "1and2"):
		''' 
		Cross matching data from the outputs from SExtractor.
	 python command of stilts. The output from this routine will give a file that
	 contains matches from both the lists supplied. 
	 tol_a is the tolerance between sky locations in arcsecs
	join indicates the type of output. See Stilts documentation for types.
	'''
		tmatch2 ="java -jar /home/pkuzma/own_codes/stilts.jar tmatch2 \
		 matcher=sky params={0} in1={1}/{1}_gaia.fits ifmt1='fits' values1='ra dec' in2={1}/{1}_PS.fits \
		 ifmt2='fits' values2='RAJ2000 DEJ2000' out={1}/{1}_G_PS_ready.fits ofmt='fits' join={2}".format(tol_a, cluster, join)
		os.system(tmatch2)
		print("Complete!")


	def gaia_u_clean(self,cluster,has_space="no",altname=False,dust_first=True):
		'''
		Cleans the gaia data as well as calculate the photometric uncertainties.
		Input parameters:
		cluster - the cluster we wish to analyse.
		'''
		if altname!=False:
			center=	coordinates.SkyCoord.from_name(altname)
		else:
			if has_space=="no":
				clust=cluster[:3]+' '+cluster[3:]
				center = coordinates.SkyCoord.from_name(cluster)
			elif has_space=="yes":
				center = coordinates.SkyCoord.from_name(cluster)
		RA=center.ra.value
		DEC=center.dec.value
		print(RA,DEC)
		if dust_first==True:
			try:
				f_in=fits.open("{0}/{0}_G_PS_ready.fits".format(cluster))
				print("Loaded in {0}_G_PS_ready.fits for cleaning.".format(cluster))			
			except FileNotFoundError:
				f_in=fits.open("{0}/{0}_gaia.fits".format(cluster))
				print("Loaded in {0}_gaia_cleaned.fits for cleaning.".format(cluster))		
		else:
			try:
				f_in=fits.open("{0}/{0}_G_PS_ready.fits".format(cluster))
				print("Loaded in {0}_G_PS_ready.fits for cleaning.".format(cluster))
			except FileNotFoundError:
				f_in=fits.open("{0}/{0}_gaia.fits".format(cluster))
				print("Loaded in {0}_gaia.fits for cleaning.".format(cluster))
		f_data=Table(f_in[1].data)
		f_data=f_data[np.isnan(f_data['bp_rp'])==False]
		f_data=f_data[np.isnan(f_data['pmra'])==False]
		u=(f_data['astrometric_chi2_al']/(f_data['astrometric_n_good_obs_al']-5))**0.5
		f_data['u']=u
		tw=np.asarray((self.ruwe['#g_mag'],self.ruwe['bp_rp'])).T
		tw1=np.asarray((self.ruwe['#g_mag'],self.ruwe['bp_rp'],self.ruwe['u0'])).T
		RUWE_Tree=KDTree(tw)
		colmag_ar=np.asarray((f_data['phot_g_mean_mag'],f_data['bp_rp'])).T
		AD=tw1[RUWE_Tree.query(colmag_ar)[1]][:,2]
		f_data['u0']=AD
		f_data['ruwe']=f_data['u']/f_data['u0']
		ones=np.ones((len(f_data)))
		f_data['V_mag']=f_data['phot_g_mean_mag']+0.01760+0.00686*f_data['bp_rp']+0.1732*f_data['bp_rp']**2
		f_data['I_mag']=f_data['phot_g_mean_mag']-0.02085-0.7419*f_data['bp_rp']+0.09631*f_data['bp_rp']**2
		f_data['Vmini']=f_data['V_mag']-f_data['I_mag']
		gr=np.zeros((len(f_data),7))
		for i in PB.progressbar(range(len(gr))):
			try:
				gr[i,0]=gMagnitudeError(f_data['phot_g_mean_mag'][i])
			except:
				gr[i,0]=0
			try:
				gr[i,1]=bpMagnitudeError(f_data['phot_g_mean_mag'][i],f_data['Vmini'][i])
			except:
				gr[i,1]=0
			try:
				gr[i,2]=rpMagnitudeError(f_data['phot_g_mean_mag'][i],f_data['Vmini'][i])
			except:
				gr[i,2]=0
			a=np.deg2rad(f_data['ra'][i])
			racl=np.deg2rad(f_data['ra'][i]-RA)
			de=np.deg2rad(f_data['dec'][i])
			decl=np.deg2rad(DEC)
			pmr=f_data['pmra'][i]
			gr[i,3]=np.rad2deg(np.cos(de)*np.sin(racl))
			gr[i,4]=np.rad2deg(np.sin(de)*np.cos(decl)-np.cos(de)*np.sin(decl)*np.cos(racl))
			gr[i,5]=pmr*np.cos(racl)-f_data['pmdec'][i]*np.sin(de)*np.sin(racl)
			gr[i,6]=pmr*np.sin(decl)*np.sin(racl)+f_data['pmdec'][i]*\
			(np.cos(de)*np.cos(decl)+np.sin(de)*np.sin(decl)*np.cos(racl))
		f_data['g_err']=gr[:,0]
		f_data['bp_err']=gr[:,1]
		f_data['rp_err']=gr[:,2]
		f_data['ra_g']=gr[:,3]
		f_data['dec_g']=gr[:,4]
		f_data['pmra_g']=gr[:,5]
		f_data['pmdec_g']=gr[:,6]
		f_data.write("{0}/{0}_Gaia_expand.fits".format(cluster),format="fits",overwrite=True)
		if dust_first==False:
			exponent=np.exp(-0.2*(f_data['phot_g_mean_mag']-19.5))
			Condition = 1.2*np.maximum(ones,exponent)
			new_dat= f_data[f_data['u'] <= Condition]
			new_dat=f_data[f_data['ruwe']<1.4]
			new_dat.write("{0}/{0}_Gaia_cleaned.fits".format(cluster),format="fits",overwrite=True)
		else:
			print("No trimming performed")		


	def GAIA_reddening(self,file,ps1=False,first=True):
		'''
		Create a de-reddened catalog. Values from Malhan et al. 2018 or from
		the e Padova model site: http://stev.oapd.inaf.it/cgi-bin/cmd 2.8. 
		'''
		if first==False:
			print("First run, dereddening entire Gaia catalogue")
			f_in=fits.open("{0}/{0}_Gaia_expand.fits".format(file))
		else:		
			f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(file))
		f_data=Table(f_in[1].data)
		X=f_data['ra']
		Y=f_data['dec']
		coords=SkyCoord(X,Y,unit='deg',frame='icrs')
		sfd = SFDQuery()
		DUST=sfd(coords)
		AG=0.85926
		ABP=1.06794
		ARP=0.65199
		RV=3.1
		dmag=RV*DUST
		g_0=f_data['phot_g_mean_mag']
		bp_0=f_data['phot_bp_mean_mag']
		rp_0=f_data['phot_rp_mean_mag']
		g_0=g_0-dmag*AG
		bp_0=bp_0-dmag*ABP
		rp_0=rp_0-dmag*ARP
		f_data['g_0']=g_0
		f_data['bp_0']=bp_0
		f_data['rp_0']=rp_0
		f_data['EB_V_G']=DUST
		if ps1==False:
			f_data.write("{0}/{0}_Gaia_dust.fits".format(file),format="fits",overwrite=True)
			f_data=f_data[f_data['ruwe']<1.4]
			#f_data=f_data[f_data['g_0']<=(20-max(f_data['EB_V_G'])*AG*RV)]
			#f_data=f_data[f_data['g_0']<=19.2]
			#print(20-min(f_data['EB_V_G'])*AG*RV,20-max(f_data['EB_V_G'])*AG*RV)
			f_par=f_data[f_data['parallax']>=(1/3000.+3*f_data['parallax_error'])]
			f_par.write("{0}/{0}_Parallax_removed.fits".format(file),format="fits",overwrite=True)
			f_data=f_data[f_data['parallax']<(1/3000.+3*f_data['parallax_error'])]
			f_data.write("{0}/{0}_Gaia_cleaned.fits".format(file),format="fits",overwrite=True)
		if ps1==True:
			f_par=f_data[f_data['parallax']>=(1/3000.+3*f_data['parallax_error'])]
			f_par.write("{0}/{0}_Parallax_removed.fits".format(file),format="fits",overwrite=True)
			f_data=f_data[f_data['parallax']<(1/3000.+3*f_data['parallax_error'])]
			f_data.write("{0}/{0}_Gaia_cleaned.fits".format(file),format="fits",overwrite=True)





	def PS1_reddening(self,file,ps1=False):
		'''
		Create a de-reddened catalog. 
		Values from Green et al. 2017
		'''
		if ps1==False:
			f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(file))
			f_data=Table(f_in[1].data)
			X=f_data['ra']
			Y=f_data['dec']
		else:
			f_in=fits.open("{0}/{0}_PS.fits".format(file))
			f_data=Table(f_in[1].data)
			X=f_data['RAJ2000']
			Y=f_data['DEJ2000']
		coords=SkyCoord(X,Y,unit='deg',frame='icrs')
		sfd = SFDQuery()
		DUST=sfd(coords)
		Rg=3.384
		Rr=2.483
		Ri=1.838
		Rz=1.414
		Ry=1.126
		dmag=DUST
		g_0=f_data['gmag']
		bp_0=f_data['rmag']
		rp_0=f_data['imag']
		g_0=g_0-dmag*Rg
		bp_0=bp_0-dmag*Rr
		rp_0=rp_0-dmag*Ri
		f_data['g_R0']=g_0
		f_data['r_R0']=bp_0
		f_data['i_R0']=rp_0
		f_data['EB_V_P']=DUST
		f_data=f_data[np.isnan(f_data['g_R0'])==False]
		f_data=f_data[np.isnan(f_data['i_R0'])==False]
		f_data=f_data[np.isnan(f_data['r_R0'])==False]
		f_data=f_data[np.isnan(f_data['e_gmag'])==False]
		f_data=f_data[np.isnan(f_data['e_imag'])==False]
		f_data=f_data[np.isnan(f_data['e_rmag'])==False]
		#f_data=f_data[f_data['g_R0']<(22-np.argmax(f_data['EB_V_P'])*Rg)]	
		if ps1==False:
			f_data.write("{0}/{0}_Gaia_cleaned.fits".format(file),format="fits",overwrite=True)
		else:
			f_data.write("{0}/{0}_PS_cleaned.fits".format(file),format="fits",overwrite=True)


	def func(self,x,a,b,c,d):
		return a+b*x+c/(x-d)

	def catalog_selection(self,file,survey,rin,rout,pmra_up,pmra_down,pmdec_up,pmdec_down):
		'''
		This will create the catalog that will be used to create the
		selection isochrone. The input parameters are:
		file - input file
		rin - inner radii in arcmin
		rout - outer radii in arcmin
		pmra_up - upper limit on proper motion selection
		pmra_down - lower limit on proper motion selection
		pmdec_up - upper limit on proper motion selection
		pmdec_down - lower limit on proper motion selection
		'''

		f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(file))

		f_data=Table(f_in[1].data)
		f_data=f_data[f_data['dist']>(rin/60.)]
		f_data=f_data[f_data['dist']<(rout/60.)]
		f_data=f_data[f_data['pmra']>pmra_down]
		f_data=f_data[f_data['pmra']<pmra_up]
		f_data=f_data[f_data['pmdec']>pmdec_down]
		f_data=f_data[f_data['pmdec']<pmdec_up]
		f_data.write("{0}/{0}_isochrone_select.fits".format(file),format="fits",overwrite=True)



	def iso_from_CMD(self,cluster,survey,magu,magl,magturn,mstep=0.01,bstep=0.01,bsize=0.1,brgb_size=0.3,plot="yes",coll=0.0,colh=1.1,colstep=0.002,cmd_split=0.2):
		'''
		Will create the isochrone from the data. Lots of inputer data
		Input parameters:
		cluster - input file
		survey - photometry being used. So far: "Gaia" and "PS"
		magu - bright cut-off
		magl - lower cut-off
		mstep - magnitude step for calculating the 
		bstep - magnitude bin width 
		plot - Want to plot things? Make this "yes"
		coll - lower colour limit
		colh - higher colour limit
		colstep - bin width in colour
	
		'''
		f_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		ht=Table(f_in[1].data)
		if survey=="PS1":
			ht=ht[np.isnan(ht['g_R0'])==False]
		if survey=="PS1":
			if magu>=max(ht['i_R0']):
				magu=max(ht['i_R0'])
		if survey=="Gaia":
			ht=ht[ht['phot_g_mean_mag']<20] 
			if magu>=max(ht['g_0']):
				magu=max(ht['g_0'])
				
		mag=np.arange(magl,(magu)+mstep,mstep)
	
		#print(mag)
		mag_iso=np.arange(magl,(magu)+mstep,0.0001)
		col=np.arange(coll,colh,colstep)
		binstep=mag
		#print(len(mag))
		iso=np.zeros((len(binstep),2))
		ISO=np.zeros((len(mag_iso),2))
		if survey=="Gaia":
		
			ht=ht[(ht['bp_0']-ht['rp_0'])>coll]
			for i in PB.progressbar(range(len(binstep))):
				mid_bin=binstep[i]
				if mid_bin<=magturn:
					htm=ht[ht['g_0']<=(mid_bin+brgb_size)]
					htm=htm[htm['g_0']>=(mid_bin-brgb_size)]
					#print(len(htm))
					#print(mid_bin,i)
					Hst,bins=np.histogram((htm['bp_0']-htm['rp_0']),bins=col)
					Bns=0.5*(bins[1:]+bins[:-1])
					#g_int=models.Gaussian1D(amplitude=1,mean=0.5,stddev=1)
					#fit_g=fitting.LevMarLSQFitter()
					#gw=fit_g(g_int,Bns,Hst)
					#Func=gw(Bns)
					iso[i,1]=binstep[i]
					#iso[i,0]=Bns[Func.argmax()]
					iso[i,0]=np.median(htm['bp_0']-htm['rp_0'])
					#iso[i,0]=np.mean(htm['g_R0']-htm['i_R0'])
				#iso=iso[iso[:,0]>coll]
				#sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
				if mid_bin >magturn and mid_bin <= (magturn+bsize):
					iso[i,0]=0
					iso[i,1]=0
				if mid_bin>(magturn+bsize):
					htm=ht[ht['g_0']<=(mid_bin+bsize)]
					htm=htm[htm['g_0']>=(mid_bin-bsize)]
					#print(len(htm))
					#print(mid_bin,i)
					Hst,bins=np.histogram((htm['bp_0']-htm['rp_0']),bins=col)
					Bns=0.5*(bins[1:]+bins[:-1])
					#g_int=models.Gaussian1D(amplitude=1,mean=0.5,stddev=1)
					#fit_g=fitting.LevMarLSQFitter()
					#gw=fit_g(g_int,Bns,Hst)
					#Func=gw(Bns)
					iso[i,1]=binstep[i]
					#iso[i,0]=Bns[Func.argmax()]
					iso[i,0]=np.median(htm['bp_0']-htm['rp_0'])
					#iso[i,0]=np.mean(htm['g_R0']-htm['i_R0'])
				#iso=iso[iso[:,0]>coll]
				#sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
			iso=iso[iso[:,1]!=0]
			f=interp1d(iso[:,1],iso[:,0],kind="cubic")
			sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
			if plot=="yes":
				plt.plot(ht['bp_0']-ht['rp_0'],ht['g_0'],',')
				plt.plot(f(mag_iso),mag_iso,'r')
				plt.plot(ht['g_0'],sp3(ht['g_0']),'g.')
				plt.plot(iso[:,0],iso[:,1],'r')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
			xs=np.linspace(magl,magu,10000)
		if survey=="PS1":
			ht=ht[(ht['g_R0']-ht['i_R0'])>coll]
			for i in PB.progressbar(range(len(binstep))):
				mid_bin=binstep[i]
				if mid_bin<=magturn:
					htm=ht[ht['i_R0']<=(mid_bin+brgb_size)]
					htm=htm[htm['i_R0']>=(mid_bin-brgb_size)]
					#print(len(htm))
					#print(mid_bin,i)
					Hst,bins=np.histogram((htm['g_R0']-htm['i_R0']),bins=col)
					Bns=0.5*(bins[1:]+bins[:-1])
					#g_int=models.Gaussian1D(amplitude=1,mean=0.5,stddev=1)
					#fit_g=fitting.LevMarLSQFitter()
					#gw=fit_g(g_int,Bns,Hst)
					#Func=gw(Bns)
					iso[i,1]=binstep[i]
					#iso[i,0]=Bns[Func.argmax()]
					iso[i,0]=np.median(htm['g_R0']-htm['i_R0'])
					#iso[i,0]=np.mean(htm['g_R0']-htm['i_R0'])
				#iso=iso[iso[:,0]>coll]
				#sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
				if mid_bin>magturn:
					htm=ht[ht['i_R0']<=(mid_bin+bsize)]
					htm=htm[htm['i_R0']>=(mid_bin-bsize)]
					#print(len(htm))
					#print(mid_bin,i)
					Hst,bins=np.histogram((htm['g_R0']-htm['i_R0']),bins=col)
					Bns=0.5*(bins[1:]+bins[:-1])
					#g_int=models.Gaussian1D(amplitude=1,mean=0.5,stddev=1)
					#fit_g=fitting.LevMarLSQFitter()
					#gw=fit_g(g_int,Bns,Hst)
					#Func=gw(Bns)
					iso[i,1]=binstep[i]
					#iso[i,0]=Bns[Func.argmax()]
					iso[i,0]=np.median(htm['g_R0']-htm['i_R0'])
					#iso[i,0]=np.mean(htm['g_R0']-htm['i_R0'])
				#iso=iso[iso[:,0]>coll]
				#sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
			f=interp1d(iso[:,1],iso[:,0],kind="cubic")
			if plot=="yes":
				plt.plot(ht['g_R0']-ht['i_R0'],ht['i_R0'],',')
				plt.plot(f(mag_iso),mag_iso,'r')
			#	plt.plot(ht['i_R0'],sp3(ht['i_R0']),'g.')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
			xs=np.linspace(magl,magu,10000)
		a1=mag_iso[mag_iso>=(magturn+bsize)]
		print(a1)
		a2=mag_iso[mag_iso<(magturn-brgb_size)][:-1]
		print(a2)
		P4,p4=sop.curve_fit(self.func,a2,f(a2),bounds=([-np.inf,-np.inf,-np.inf,-100],[np.inf,np.inf,np.inf,100]))
		P3,p3=sop.curve_fit(self.func,a1,f(a1),bounds=([-np.inf,-np.inf,-np.inf,-100],[np.inf,np.inf,np.inf,100]))
		F1=self.func(a1,*P3)
		F2=self.func(a2,*P4)
		VCS=interp1d(np.append(a1,a2),np.append(F1,F2),kind="cubic")
		if survey=="Gaia":
			if plot=="yes":
				plt.plot(ht['bp_0']-ht['rp_0'],ht['g_0'],',')
				plt.plot(VCS(mag_iso),mag_iso,'r')
				#plt.plot(ht['g_0'],sp3(ht['g_0']),'g.')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
		if survey=="PS1":
			if plot=="yes":
				plt.plot(ht['g_R0']-ht['i_R0'],ht['i_R0'],',')
				plt.plot(VCS(mag_iso),mag_iso,'r')
			#	plt.plot(ht['i_R0'],sp3(ht['i_R0']),'g.')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
		fopen=open("{0}/{0}_isochrone.pickle".format(cluster),'wb')
		cPickle.dump(VCS,fopen)
		fopen.close()
		
	def exp_func(self,x,a,b,c):
		'''exponential function'''
		return a*np.exp(b*x)+c
	

	
	def weight_calculation(self,cluster,survey,magu,magl,colu,coll,t_rad,turnoff_tip=0,turnoff_base=0,pert=0.4,sig_sel=3):
		f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
		f_data=Table(f_in[1].data)
		fopen=open("{0}/{0}_isochrone.pickle".format(cluster),'rb')
		turnoff=np.arange(turnoff_tip,turnoff_base+0.001,0.001)
		spl=cPickle.load(fopen)
		x_MAG=np.array((magl,turnoff_tip,turnoff_base,magu))
		F_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		F_data=Table(F_in[1].data)
		fopen.close()
		if survey=="Gaia":
			f_data=f_data[np.isnan(f_data['g_0'])==False]
			f_data=f_data[f_data['g_0']>= magl]
			f_data=f_data[f_data['g_0']<= magu]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])>= coll]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])<= colu]
			F_data=F_data[F_data['g_0']>= magl]
			F_data=F_data[F_data['g_0']<= magu]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])>= coll]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])<= colu]
			col_err=(np.sqrt(f_data['bp_err']**2+f_data['rp_err']**2))
			act_mag=f_data['g_0']
			col_mag=(f_data['bp_0']-f_data['rp_0'])
			col_errF=(np.sqrt(F_data['bp_err']**2+F_data['rp_err']**2))
			act_magF=F_data['g_0']
			col_magF=(F_data['bp_0']-F_data['rp_0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<i+0.1) & (YE[:,1]>i - 0.1)][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			f_ms=f_data1[f_data1['g_0']>=turnoff_base]
			f_trnof
			f_rgb=f_data1[f_data1['g_0']<turnoff_tip]
			hst,bns=np.histogram(f_ms['w_iso'],bins=np.arange(-30,30,0.001))
			mid_bns=bns[:-1]+0.0005			
			g = models.Gaussian1D()
			fit_t = fitting.LevMarLSQFitter()
			t1 = fit_t(g, mid_bns, hst)
			plt.hist(f_ms['w_iso'],bins=np.arange(-30,30,0.001))
			plt.plot(mid_bns,t1(mid_bns))
			plt.xlim((-10,10))
			plt.show()
			hst,bns=np.histogram(f_rgb['w_iso'],bins=np.arange(-30,30,0.001))
			mid_bns=bns[:-1]+0.0005			
			g = models.Gaussian1D()
			fit_t = fitting.LevMarLSQFitter()
			t2 = fit_t(g, mid_bns, hst)
			y_COL=np.array((3*t2.stddev.value,3*t2.stddev.value,3*t1.stddev.value,3*t1.stddev.value))
			Func=interp1d(x_MAG,y_COL)
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			f_OUT=f_data[~((abs(f_data['w_iso']))<=Func(f_data['g_0']))]
			f_data=f_data[((abs(f_data['w_iso']))<=Func(f_data['g_0']))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['i_R0'])==False]
			f_data=f_data[f_data['i_R0']>= magl]
			f_data=f_data[f_data['i_R0']<= magu]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])>= coll]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])<= colu]
			F_data=F_data[F_data['i_R0']>= magl]
			F_data=F_data[F_data['i_R0']<= magu]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])>= coll]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])<= colu]
			col_err=(np.sqrt(f_data['e_gmag']**2+f_data['e_imag']**2))
			act_mag=f_data['i_R0']
			col_mag=(f_data['g_R0']-f_data['i_R0'])
			col_errF=(np.sqrt(F_data['e_gmag']**2+F_data['e_imag']**2))
			act_magF=F_data['i_R0']
			col_magF=(F_data['g_R0']-F_data['i_R0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<i+0.1) & (YE[:,1]>i - 0.1)][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			hst,bns=np.histogram(f_data1['w_iso'],bins=np.arange(-300,300,1))
			mid_bns=bns[:-1]+0.5
			g = models.Gaussian1D(amplitude=100, mean=0, stddev=10)
			fit_t = fitting.LevMarLSQFitter()
			t = fit_t(g, mid_bns, hst)
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			f_OUT=f_data[~((f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value)))]
			f_data=f_data[(f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)

	def iso_from_CMD_RGBonly(self,cluster,survey,magu,magl,magturn,mstep=0.01,bstep=0.01,bsize=0.1,brgb_size=0.3,plot="yes",coll=0.0,colh=1.1,colstep=0.002,cmd_split=0.2):
		'''
		Will create the isochrone from the data. Lots of inputer data
		Input parameters:
		cluster - input file
		survey - photometry being used. So far: "Gaia" and "PS"
		magu - bright cut-off
		magl - lower cut-off
		mstep - magnitude step for calculating the 
		bstep - magnitude bin width 
		plot - Want to plot things? Make this "yes"
		coll - lower colour limit
		colh - higher colour limit
		colstep - bin width in colour
	
		'''
		f_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		ht=Table(f_in[1].data)
		if survey=="PS1":
			ht=ht[np.isnan(ht['g_R0'])==False]
		mag=np.arange(magl,(magu)+mstep,mstep)
		#print(mag)
		mag_iso=np.arange(magl,(magu),0.0001)
		col=np.arange(coll,colh,colstep)
		binstep=mag
		#print(len(mag))
		iso=np.zeros((len(binstep),2))
		ISO=np.zeros((len(mag_iso),2))
		if survey=="Gaia":
			for i in PB.progressbar(range(len(binstep))):
				mid_bin=binstep[i]
				htm=ht[ht['g_0']<=(mid_bin+brgb_size)]
				htm=htm[htm['g_0']>=(mid_bin-brgb_size)]
				#print(len(htm))
				#print(mid_bin,i)
				Hst,bins=np.histogram((htm['bp_0']-htm['rp_0']),bins=col)
				Bns=0.5*(bins[1:]+bins[:-1])
				#g_int=models.Gaussian1D(amplitude=1,mean=0.5,stddev=1)
				#fit_g=fitting.LevMarLSQFitter()
				#gw=fit_g(g_int,Bns,Hst)
				#Func=gw(Bns)
				iso[i,1]=binstep[i]
				#iso[i,0]=Bns[Func.argmax()]
				iso[i,0]=np.median(htm['bp_0']-htm['rp_0'])
				#iso[i,0]=np.mean(htm['g_R0']-htm['i_R0'])
				#iso=iso[iso[:,0]>coll]
				#sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
			f=interp1d(iso[:,1],iso[:,0],kind="cubic")
			sp3=UnivariateSpline(iso[:,1],iso[:,0],k=3)
			if plot=="yes":
				plt.plot(ht['bp_0']-ht['rp_0'],ht['g_0'],',')
				plt.plot(f(mag_iso),mag_iso,'r')
				plt.plot(ht['g_0'],sp3(ht['g_0']),'g.')
				plt.plot(iso[:,0],iso[:,1],'r')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
			xs=np.linspace(magl,magu,10000)
		if survey=="PS1":
			P3,p3=sop.curve_fit(self.func,ht['i_R0'],(ht['g_R0']-ht['i_R0']),bounds=([-np.inf,-np.inf,-np.inf,-100],[np.inf,np.inf,np.inf,100]))
		a1=mag_iso
		print(a1)
		F1=self.func(a1,*P3)
		if survey=="Gaia":
			if plot=="yes":
				plt.plot(ht['bp_0']-ht['rp_0'],ht['g_0'],',')
				plt.plot(F1(mag_iso),mag_iso,'r')
				#plt.plot(ht['g_0'],sp3(ht['g_0']),'g.')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
		if survey=="PS1":
			if plot=="yes":
				plt.plot(ht['g_R0']-ht['i_R0'],ht['i_R0'],',')
				plt.plot(F1,a1,'r')
			#	plt.plot(ht['i_R0'],sp3(ht['i_R0']),'g.')
				plt.ylim((magl,magu))
				plt.xlim((coll,colh))
				plt.gca().invert_yaxis()
				plt.show()
		fopen=open("{0}/{0}_isochrone_param.pickle".format(cluster),'wb')
		cPickle.dump(P3,fopen)
		fopen.close()

	def weight_calculationP(self,cluster,survey,magu,magl,colu,coll,t_rad,pert=0.4,sig_sel=3):
		f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
		f_data=Table(f_in[1].data)
		fopen=open("{0}/{0}_isochrone_param.pickle".format(cluster),'rb')
		spl=cPickle.load(fopen)
		F_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		F_data=Table(F_in[1].data)
		fopen.close()
		if survey=="Gaia":
			f_data=f_data[np.isnan(f_data['g_0'])==False]
			f_data=f_data[f_data['g_0']>= magl]
			f_data=f_data[f_data['g_0']<= magu]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])>= coll]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])<= colu]
			F_data=F_data[F_data['g_0']>= magl]
			F_data=F_data[F_data['g_0']<= magu]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])>= coll]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])<= colu]
			col_err=(np.sqrt(f_data['bp_err']**2+f_data['rp_err']**2))
			act_mag=f_data['g_0']
			col_mag=(f_data['bp_0']-f_data['rp_0'])
			col_errF=(np.sqrt(F_data['bp_err']**2+F_data['rp_err']**2))
			act_magF=F_data['g_0']
			col_magF=(F_data['bp_0']-F_data['rp_0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<i+0.1) & (YE[:,1]>i - 0.1)][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-self.func(act_mag,*spl))/col_err
			F_data['w_iso']=(col_magF-self.func(act_mag,*spl))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data			
			hst,bns=np.histogram(f_data1['w_iso'],bins=np.arange(-300,300,1))
			mid_bns=bns[:-1]+0.5
			g = models.Gaussian1D(amplitude=100, mean=0, stddev=10)
			fit_t = fitting.LevMarLSQFitter()
			t = fit_t(g, mid_bns, hst)
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			f_OUT=f_data[~((f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value)))]
			f_data=f_data[(f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['i_R0'])==False]
			f_data=f_data[f_data['i_R0']>= magl]
			f_data=f_data[f_data['i_R0']<= magu]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])>= coll]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])<= colu]
			F_data=F_data[F_data['i_R0']>= magl]
			F_data=F_data[F_data['i_R0']<= magu]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])>= coll]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])<= colu]
			col_err=(np.sqrt(f_data['e_gmag']**2+f_data['e_imag']**2))
			act_mag=f_data['i_R0']
			col_mag=(f_data['g_R0']-f_data['i_R0'])
			col_errF=(np.sqrt(F_data['e_gmag']**2+F_data['e_imag']**2))
			act_magF=F_data['i_R0']
			col_magF=(F_data['g_R0']-F_data['i_R0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<i+0.1) & (YE[:,1]>i - 0.1)][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-self.func(act_mag,*spl))/col_err
			F_data['w_iso']=(col_magF-self.func(act_magF,*spl))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			hst,bns=np.histogram(f_data1['w_iso'],bins=np.arange(-300,300,1))
			mid_bns=bns[:-1]+0.5
			g = models.Gaussian1D(amplitude=100, mean=0, stddev=10)
			fit_t = fitting.LevMarLSQFitter()
			t = fit_t(g, mid_bns, hst)
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			f_OUT=f_data[~((f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value)))]
			f_data=f_data[(f_data['w_iso']>=(-3*t.stddev.value)) & (f_data['w_iso']<=(3*t.stddev.value))]
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)


	def weight_calculation_vari(self,cluster,survey,magu,magl,colu,coll,turnoff_tip=0,turnoff_base=0):
		'''
		Inputs:
		cluster: Cluster name
		survey: survey name
		magu: upper magnitude limit
		magl: lower magnitude limit
		colu: upper colour limit
		coll: uppper colour limit
		t_rad: tidal radius
		'''
		f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
		f_data=Table(f_in[1].data)
		fopen=open("{0}/{0}_isochrone.pickle".format(cluster),'rb')
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['g_R0'])==False]
		if survey=="PS1":
			if magu>=max(f_data['i_R0']):
				magu=max(f_data['i_R0'])
		if survey=="Gaia":
			if magu>=max(f_data['g_0']):
				magu=max(f_data['g_0'])
		turnoff=np.arange(turnoff_tip,turnoff_base+0.001,0.001)
		spl=cPickle.load(fopen)
		x_MAG=np.array((magl,turnoff_tip,turnoff_base,magu))
		F_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		F_data=Table(F_in[1].data)
		self.mag_range=np.arange(magl,magu,0.1)
		fopen.close()
		self.weight=np.zeros(len(self.mag_range))
		if survey=="Gaia":
			f_data=f_data[np.isnan(f_data['g_0'])==False]
			f_data=f_data[f_data['g_0']>= magl]
			f_data=f_data[f_data['g_0']<= magu]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])>= coll]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])<= colu]
			F_data=F_data[F_data['g_0']>= magl]
			F_data=F_data[F_data['g_0']<= magu]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])>= coll]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])<= colu]
			col_err=(np.sqrt(f_data['bp_err']**2+f_data['rp_err']**2))
			act_mag=f_data['g_0']
			col_mag=(f_data['bp_0']-f_data['rp_0'])
			col_errF=(np.sqrt(F_data['bp_err']**2+F_data['rp_err']**2))
			act_magF=F_data['g_0']
			col_magF=(F_data['bp_0']-F_data['rp_0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			print(len(magg))
			print(len(gte))
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):
				sel=f_data1[(f_data1['g_0']<=(self.mag_range[i]+0.1))&(f_data1['g_0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005	
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=3*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array[self.weight_array[:,1]>10**-4]
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],bounds=([-np.inf,-np.inf,-np.inf],[np.inf,0,np.inf]))
			self.PC=PC
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mem_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['i_R0'])==False]
			f_data=f_data[f_data['i_R0']>= magl]
			f_data=f_data[f_data['i_R0']<= magu]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])>= coll]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])<= colu]
			F_data=F_data[F_data['i_R0']>= magl]
			F_data=F_data[F_data['i_R0']<= magu]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])>= coll]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])<= colu]
			col_err=(np.sqrt(f_data['e_gmag']**2+f_data['e_imag']**2))
			act_mag=f_data['i_R0']
			col_mag=(f_data['g_R0']-f_data['i_R0'])
			col_errF=(np.sqrt(F_data['e_gmag']**2+F_data['e_imag']**2))
			act_magF=F_data['i_R0']
			col_magF=(F_data['g_R0']-F_data['i_R0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):	
				sel=f_data1[(f_data1['i_R0']<=(self.mag_range[i]+0.1))&(f_data1['g_0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=3*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array[self.weight_array[:,1]>10**-4]
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],bounds=([-np.inf,-np.inf,-np.inf],[np.inf,0,np.inf]))
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)	
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mean_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)	
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)	
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)	

	def weight_calculation_vari_FULL(self,cluster,survey,magu,magl,colu,coll,turnoff_tip=0,turnoff_base=0):
		'''
		Inputs:
		cluster: Cluster name
		survey: survey name
		magu: upper magnitude limit
		magl: lower magnitude limit
		colu: upper colour limit
		coll: uppper colour limit
		t_rad: tidal radius
		'''
		f_in=fits.open("{0}/{0}_Gaia_dust.fits".format(cluster))
		f_data=Table(f_in[1].data)
		fopen=open("{0}/{0}_isochrone.pickle".format(cluster),'rb')
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['g_R0'])==False]
		if survey=="PS1":
			if magu>=max(f_data['i_R0']):
				magu=max(f_data['i_R0'])
		if survey=="Gaia":
			if magu>=max(f_data['g_0']):
				magu=max(f_data['g_0'])
		turnoff=np.arange(turnoff_tip,turnoff_base+0.001,0.001)
		spl=cPickle.load(fopen)
		x_MAG=np.array((magl,turnoff_tip,turnoff_base,magu))
		F_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		F_data=Table(F_in[1].data)
		self.mag_range=np.arange(magl,magu,0.1)
		fopen.close()
		self.weight=np.zeros(len(self.mag_range))
		if survey=="Gaia":
			f_data=f_data[np.isnan(f_data['g_0'])==False]
			f_data=f_data[f_data['g_0']>= magl]
			f_data=f_data[f_data['g_0']<= magu]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])>= coll]
			f_data=f_data[(f_data['bp_0']-f_data['rp_0'])<= colu]
			F_data=F_data[F_data['g_0']>= magl]
			F_data=F_data[F_data['g_0']<= magu]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])>= coll]
			F_data=F_data[(F_data['bp_0']-F_data['rp_0'])<= colu]
			col_err=(np.sqrt(f_data['bp_err']**2+f_data['rp_err']**2))
			act_mag=f_data['g_0']
			col_mag=(f_data['bp_0']-f_data['rp_0'])
			col_errF=(np.sqrt(F_data['bp_err']**2+F_data['rp_err']**2))
			act_magF=F_data['g_0']
			col_magF=(F_data['bp_0']-F_data['rp_0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			print(len(magg))
			print(len(gte))
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):
				sel=f_data1[(f_data1['g_0']<=(self.mag_range[i]+0.1))&(f_data1['g_0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005	
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=2*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array[self.weight_array[:,1]>10**-4]
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],bounds=([-np.inf,-np.inf,-np.inf],[np.inf,0,np.inf]))
			f_data.write("{0}/{0}_F_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
			EXP=np.zeros((len(f_data)))
			MEMX=np.zeros((len(f_data)))
			MEMSTD=np.zeros((len(f_data)))
			for i in PB.progressbar(range(len(f_data))):
				if (~((abs(f_data['w_iso'][i]))<=self.exp_func(f_data['g_0'][i],*PC)))==True:
					EXP[i]=True
					MEMX[i]=0
					MEMSTD[i]=0
				else:
					EXP[i]=False
					MEMX[i]=True
					MEMSTD[i]=True
			f_data['col_sel']=EXP
			f_data['mem_x']=MEMX
			f_data['mean_std']=MEMSTD
			f_data=f_data[f_data['col_sel']==True]
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mem_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT.write("{0}/{0}_F_everything_else.fits".format(cluster),format="fits",overwrite=True)
			f_data.write("{0}/{0}_F_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']
#			f_data.write("{0}/{0}_F_bays_ready.fits".format(cluster),format="fits",overwrite=True)
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['i_R0'])==False]
			f_data=f_data[f_data['i_R0']>= magl]
			f_data=f_data[f_data['i_R0']<= magu]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])>= coll]
			f_data=f_data[(f_data['g_R0']-f_data['i_R0'])<= colu]
			F_data=F_data[F_data['i_R0']>= magl]
			F_data=F_data[F_data['i_R0']<= magu]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])>= coll]
			F_data=F_data[(F_data['g_R0']-F_data['i_R0'])<= colu]
			col_err=(np.sqrt(f_data['e_gmag']**2+f_data['e_imag']**2))
			act_mag=f_data['i_R0']
			col_mag=(f_data['g_R0']-f_data['i_R0'])
			col_errF=(np.sqrt(F_data['e_gmag']**2+F_data['e_imag']**2))
			act_magF=F_data['i_R0']
			col_magF=(F_data['g_R0']-F_data['i_R0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):	
				sel=f_data1[(f_data1['i_R0']<=(self.mag_range[i]+0.1))&(f_data1['g_0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=3*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array[self.weight_array[:,1]>10**-4]
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],bounds=([-np.inf,-np.inf,-np.inf],[np.inf,0,np.inf]))
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)	
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mean_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)	
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)	
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec']	
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)	

	def weight_calculation_ISO(self,cluster,survey,magu,magl,colu,coll,magmov,colmov,dist,turnoff_tip=0,turnoff_base=0,stw=3,bounds=([0,-np.inf,0],[np.inf,0,np.inf]),cut=True,fit_cut=15):
		'''
		Inputs:
		cluster: Cluster name
		survey: survey name
		magu: upper magnitude limit
		magl: lower magnitude limit
		colu: upper colour limit
		coll: uppper colour limit
		t_rad: tidal radius
		'''
		f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
		f_data=Table(f_in[1].data)
		iso_file=ascii.read("{0}/{0}_isochrone.iso".format(cluster))
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['g_R0'])==False]
		if survey=="PS1":
			COLY=iso_file['col7']-iso_file['col9'] + colmov
			MAGX=iso_file['col9'] + magmov + 5*np.log10(dist)-5
			spl=interp1d(MAGX,COLY)
		if survey=="Gaia":
			COLY=iso_file['col7']-iso_file['col8'] + colmov
			MAGX=iso_file['col6'] + magmov +5*np.log10(dist)-5
			spl=interp1d(MAGX,COLY)
		F_in=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
		F_data=Table(F_in[1].data)
		self.mag_range=np.arange(turnoff_tip,magu,0.1)
		self.weight=np.zeros(len(self.mag_range))
		if survey=="Gaia":
			f_data=f_data[np.isnan(f_data['g_0'])==False]
			f_data=f_data[np.isnan(f_data['bp_0'])==False]
			f_data=f_data[f_data['g_0']>= magl]
			f_data=f_data[f_data['g_0']<= magu]
			F_data=F_data[F_data['g_0']>= turnoff_tip]
			F_data=F_data[F_data['g_0']<= magu]
			col_err=(np.sqrt(f_data['bp_err']**2+f_data['rp_err']**2))
			act_mag=f_data['g_0']
			col_mag=(f_data['bp_0']-f_data['rp_0'])
			col_errF=(np.sqrt(F_data['bp_err']**2+F_data['rp_err']**2))
			act_magF=F_data['g_0']
			col_magF=(F_data['bp_0']-F_data['rp_0'])
			gte=np.zeros(len(np.arange(turnoff_tip+0.2,magu-0.2,0.2)))
			magg=np.arange(turnoff_tip+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(turnoff_tip+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):	
				sel=f_data1[(f_data1['g_0']<=(self.mag_range[i]+0.1))&(f_data1['g_0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=stw*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array[self.weight_array[:,1]!=0]
			W_ARY=self.weight_array[self.weight_array[:,1]>10**-1]
			if cut==True:
				for i in range(len(W_ARY[:,0])):
					if W_ARY[i,0]<=turnoff_base:
						if W_ARY[i,1]<=2:
							W_ARY[i,1]=0
					else:
						j=4
			W_ARY=W_ARY[W_ARY[:,1]>0]
			W_ARY=W_ARY[W_ARY[:,0]>=fit_cut]
			self.W_ARY=W_ARY
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],sigma=np.sqrt(W_ARY[:,1]),bounds=bounds)
			self.fexp1,self.fexp=PC,PC1
			#PC,PC1=sop.curve_fit(self.exp_func,F_data['g_0'],abs(std*F_data['w_iso']),bounds=([-np.inf,-np.inf,-np.inf],[np.inf,0,np.inf]))
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)	
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mean_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['g_0'],*PC))]
			#f_data=f_data[f_data['g_0']>=turnoff_tip]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)	
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)	
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)	
		if survey=="PS1":
			f_data=f_data[np.isnan(f_data['i_R0'])==False]
			f_data=f_data[f_data['i_R0']>= magl]
			f_data=f_data[f_data['i_R0']<= magu]
			F_data=F_data[F_data['i_R0']>= magl]
			F_data=F_data[F_data['i_R0']<= magu]
			col_err=(np.sqrt(f_data['e_gmag']**2+f_data['e_imag']**2))
			act_mag=f_data['i_R0']
			col_mag=(f_data['g_R0']-f_data['i_R0'])
			col_errF=(np.sqrt(F_data['e_gmag']**2+F_data['e_imag']**2))
			act_magF=F_data['i_R0']
			col_magF=(F_data['g_R0']-F_data['i_R0'])
			gte=np.zeros(len(np.arange(magl+0.2,magu-0.2,0.2)))
			magg=np.arange(magl+0.2,magu-0.2,0.2)
			YE=np.concatenate((col_err.data[np.newaxis].T,act_mag.data[np.newaxis].T),axis=1)
			n=0
			for i in PB.progressbar(np.arange(magl+0.2,magu-0.2,0.2)):
				gte[n]=np.median(YE[(YE[:,1]<(i+0.1)) & (YE[:,1]>(i - 0.1))][:,0])
				n+=1
			EQ,EQ1=sop.curve_fit(self.exp_func,magg,gte)
			col_err=self.exp_func(act_mag,*EQ)
			col_errF=self.exp_func(act_magF,*EQ)
			f_data['w_iso']=(col_mag-spl(act_mag))/col_err
			F_data['w_iso']=(col_magF-spl(act_magF))/col_errF
			f_data=f_data[np.isnan(f_data['w_iso'])==False]
			F_data=F_data[np.isnan(F_data['w_iso'])==False]
			f_data1=F_data
			for i in PB.progressbar(range(len(self.mag_range))):	
				sel=f_data1[(f_data1['i_R0']<=(self.mag_range[i]+0.1))&(f_data1['g_R0']>=(self.mag_range[i]-0.1))]
				hst,bns=np.histogram(sel['w_iso'],bins=np.arange(-30,30,0.001))
				mid_bns=bns[:-1]+0.0005
				g = models.Gaussian1D()
				fit_t = fitting.LevMarLSQFitter()
				t1 = fit_t(g, mid_bns, hst)
				self.weight[i]=stw*t1.stddev.value
			self.weight_array=np.array((self.mag_range,self.weight)).T
			W_ARY=self.weight_array
			W_ARY=W_ARY[W_ARY[:,1]<=150]
			for i in range(len(W_ARY)):
				if W_ARY[i,0]<=17:
					if W_ARY[i,1]<25:
						W_ARY[i,1]=0
			W_ARY=W_ARY[W_ARY[:,1]>stw]				
			self.W_ARY=W_ARY
			PC,PC1=sop.curve_fit(self.exp_func,W_ARY[:,0],W_ARY[:,1],bounds=bounds)
			#PC,PC1=sop.curve_fit(self.exp_func,F_data['i_R0'],abs(stw*F_data['w_iso']),bounds=bounds)
			f_data.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)	
			f_OUT=f_data[~((abs(f_data['w_iso']))<=self.exp_func(f_data['i_R0'],*PC))]
			f_OUT['mem_x']=-1
			f_OUT['mean_std']=-1
			f_data=f_data[((abs(f_data['w_iso']))<=self.exp_func(f_data['i_R0'],*PC))]
			f_data=f_data[f_data['i_R0']>=turnoff_tip]
			f_OUT.write("{0}/{0}_everything_else.fits".format(cluster),format="fits",overwrite=True)	
			f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)	
			f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
			'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr']
			f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)	

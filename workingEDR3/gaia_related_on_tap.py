#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:39:20 2020

@author: pete
"""

# Prepare for Bayesian stats direct from Gaia and Panstarrs
#---------------------------------------------
#Import requires modules

import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table
import numpy as np
import os
import math
from pygaia.errors.photometric import gMagnitudeError, bpMagnitudeError, rpMagnitudeError
#from pygaia.errors.photometric import gMagnitudeErrorEoM, bpMagnitudeErrorEoM, rpMagnitudeErrorEoM
#from pygaia.photometry.transformations import gminvFromVmini
import astropy.units as u
from astropy.coordinates import Angle
from astropy.coordinates.sky_coordinate import SkyCoord
import astropy.coordinates as coord
#from astropy.units import Quantity
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy import coordinates
import _pickle as cPickle
from dustmaps.sfd import SFDQuery
#from scipy.stats import norm
import matplotlib.pyplot as plt
#from scipy.ndimage import filters
from astropy.modeling import models, fitting
#from matplotlib import ticker
#import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree
import scipy.optimize as sop
#from astropy.modeling import models, fitting
#import pymultinest
import progressbar as PB
# Ignore warnings from TAP queries
#from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
import ezdart
import ezmist
from ezpadova import parsec
from uncertainties import ufloat,unumpy
import scipy.stats as stats
# Get Gaia tables.
from astroquery.gaia import Gaia
#tables = Gaia.load_tables(only_names=True)
#import proxyTap
import time

#PANSTARRS
import mastcasjobs
from astropy.io import ascii
from astropy.table import Table
import glob as glob
import sys
import os
import re
import numpy as np
import pylab
import json

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib

# get the WSID and password if not already defined
import getpass
os.environ['CASJOBS_WSID'] = '1506678277'
os.environ['CASJOBS_PW'] = 'eqJg4651'
#Gaia-on-Tap
#import gaia as GOT


# ---------------------------------------------------
# Definitions

class gaia:
    def __init__(self):
        #Vizier.ROW_LIMIT=1000000000
        self.ruwe=Table(fits.open("ruwe_table.fits")[1].data)
       # GOT.config.read("/Users/pete/Downloads/owncodes/edr3code_working/xredentials.yaml")

    def mastQuery(self,request):
        """Perform a MAST query.
    
        Parameters
        ----------
        request (dictionary): The MAST request json object
    
        Returns head,content where head is the response HTTP headers, and content is the returned data"""
        
        server='mast.stsci.edu'
    
        # Grab Python Version 
        version = ".".join(map(str, sys.version_info[:3]))
    
        # Create Http Header Variables
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain",
                   "User-agent":"python-requests/"+version}
    
        # Encoding the request as a json string
        requestString = json.dumps(request)
        requestString = urlencode(requestString)
        
        # opening the https connection
        conn = httplib.HTTPSConnection(server)
    
        # Making the query
        conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)
    
        # Getting the response
        resp = conn.getresponse()
        head = resp.getheaders()
        content = resp.read().decode('utf-8')
    
        # Close the https connection
        conn.close()
    
        return head,content


    def resolve(self,name):
        """Get the RA and Dec for an object using the MAST name resolver
        
        Parameters
        ----------
        name (str): Name of object
    
        Returns RA, Dec tuple with position"""
    
        resolverRequest = {'service':'Mast.Name.Lookup',
                           'params':{'input':name,
                                     'format':'json'
                                    },
                          }
        headers,resolvedObjectString = self.mastQuery(resolverRequest)
        resolvedObject = json.loads(resolvedObjectString)
        # The resolver returns a variety of information about the resolved object, 
        # however for our purposes all we need are the RA and Dec
        try:
            objRa = resolvedObject['resolvedCoordinate'][0]['ra']
            objDec = resolvedObject['resolvedCoordinate'][0]['decl']
        except IndexError as e:
            raise ValueError("Unknown object '{}'".format(name))
        return (objRa, objDec)


    
    def get_gaia_data(self,cluster,rad=4):
        '''
        Retrieving the data from Gaia.
        Input parameters:
        cluster - the cluster of interest
        rad - radius of interest from the cluster center
        '''
        print("Retreiving Gaia data.")
        self.cluster=cluster
        #gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")
        Gaia.login(user='pkuzma', password='$eqJg4651')
        orig_path=os.getcwd()
        if os.path.isdir("{0}".format(cluster)) == False:
            print("No folder for cluster. Creating now...")
            os.makedirs("{0}".format(cluster))
            os.chdir("{0}".format(cluster))
            print("Moving into cluster folder.\n")
        else:
            os.chdir("{0}".format(cluster))
            print("Moving into cluster folder.\n")
        center=self.resolve(cluster)
        RA=center[0]
        DEC=center[1]
        job = Gaia.launch_job_async("SELECT gaia.source_id,gaia.ra,gaia.ra_error,gaia.dec, \
                gaia.dec_error,gaia.parallax,gaia.parallax_error,gaia.phot_g_mean_mag, \
                gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag, \
                gaia.bp_rp,gaia.radial_velocity,gaia.radial_velocity_error, \
                gaia.phot_variable_flag,gaia.teff_val,gaia.a_g_val, \
                gaia.pmra, gaia.pmdec, gaia.pmra_error, gaia.pmdec_error, \
                gaia.pmra_pmdec_corr, gaia.l, gaia.b, gaia.astrometric_chi2_al, \
                gaia.astrometric_n_good_obs_al, gaia.phot_bp_rp_excess_factor, \
                distance( POINT('ICRS', {0},{1}), \
                POINT('ICRS', gaia.ra, gaia.dec)) as DIST \
                FROM gaiadr2.gaia_source as gaia \
                WHERE CONTAINS(POINT('ICRS',gaia.ra,gaia.dec),CIRCLE('ICRS',{0},{1},{2}))=1 \
                AND gaia.pmra IS NOT NULL AND gaia.pmdec IS NOT NULL \
                AND gaia.phot_g_mean_mag IS NOT NULL \
                AND gaia.phot_g_mean_mag < 20.3"\
                .format(RA,DEC,rad),dump_to_file=True,output_file="{0}_gaia.fits".format(cluster),\
                output_format="fits")
        Gaia.remove_jobs(job.jobid)
        #ascii.write(job,"{0}_raw.txt".format(cluster),format="commented_header")
        f1=fits.open("{0}_gaia.fits".format(cluster))
        job=Table(f1[1].data)
        job['V_mag']=job['phot_g_mean_mag']+0.01760+0.00686*job['bp_rp']+0.1732*job['bp_rp']**2
        job['I_mag']=job['phot_g_mean_mag']-0.02085-0.7419*job['bp_rp']+0.09631*job['bp_rp']**2
        job['Vmini']=job['V_mag']-job['I_mag']
        ra= job['ra'].astype(float)
        dec= job['dec'].astype(float)
        pmra  = job['pmra'].astype(float)
        pmdec = job['pmdec'].astype(float)
        pmrae = job['pmra_error'].astype(float)
        pmdece= job['pmdec_error'].astype(float)
        pmcorr= job['pmra_pmdec_corr'].astype(float)
        sin= np.sin
        cos   = np.cos
        ra0=RA
        dec0=DEC
        d2r   = np.pi/180  # degrees to radians
        x     = (cos(dec * d2r) * sin((ra-ra0) * d2r)) / d2r   # x,y are in degrees
        y     = (sin(dec * d2r) * cos(dec0 * d2r) - cos(dec * d2r) * sin(dec0 * d2r) * cos((ra-ra0) * d2r)) / d2r
        # transformation of PM and its uncertainty covariance matrix
        Jxa   = cos((ra-ra0) * d2r)
        Jxd   = -sin(dec * d2r) * sin((ra-ra0) * d2r)
        Jya   = sin(dec0 * d2r) * sin((ra-ra0) * d2r)
        Jyd   = cos(dec  * d2r) * cos(dec0 * d2r) + sin(dec * d2r) * sin(dec0 * d2r) * cos((ra-ra0) * d2r)
        job['pmra_g']    = pmra * Jxa + pmdec * Jxd
        job['pmdec_g']    = pmra * Jya + pmdec * Jyd
        Cxx   = (Jxa * pmrae)**2 + (Jxd * pmdece)**2 + 2 * Jxa * Jxd * pmcorr * pmrae * pmdece
        Cyy   = (Jya * pmrae)**2 + (Jyd * pmdece)**2 + 2 * Jya * Jyd * pmcorr * pmrae * pmdece
        Cxy   = Jxa * Jya * pmrae**2 + Jxd * Jyd * pmdece**2 + (Jya * Jxd + Jxa * Jyd) * pmcorr * pmrae * pmdece
        job['pmra_g_err']   = Cxx**0.5
        job['pmdec_g_err']   = Cyy**0.5
        job['pmra_pmdec_g_corr'] = Cxy / (job['pmra_g_err'] * job['pmdec_g_err'])
        job['dist_r'] = (x**2 + y**2)**0.5 * 60.  # distance from cluster center in arcmin
        job['ra_g']=x
        job['dec_g']=y
        gr=np.zeros((len(job),3))
        for i in PB.progressbar(range(len(gr))):
            try:
                gr[i,0]=gMagnitudeError(job['phot_g_mean_mag'][i])
            except:
                gr[i,0]=0
            try:
                gr[i,1]=bpMagnitudeError(job['phot_g_mean_mag'][i],job['Vmini'][i])
            except:
                gr[i,1]=0
            try:
                gr[i,2]=rpMagnitudeError(job['phot_g_mean_mag'][i],job['Vmini'][i])
            except:
                gr[i,2]=0
        job['g_err']=gr[:,0]
        job['bp_err']=gr[:,1]
        job['rp_err']=gr[:,2]
        job['bp_rp_phot_excess_corrected']=self.correct_flux_excess_factor(job['bp_rp'],job['phot_bp_rp_excess_factor'])
        job.write("{0}_gaia.fits".format(self.cluster),format='fits',overwrite=True)
        os.chdir(orig_path)

    def correct_flux_excess_factor(self,bp_rp, phot_bp_rp_excess_factor):
        """
        Calculate the corrected flux excess factor for the input Gaia EDR3 data.
        
        Parameters
        ----------
        
        bp_rp: float, numpy.ndarray
            The (BP-RP) colour listed in the Gaia EDR3 archive.
        phot_bp_rp_flux_excess_factor: float, numpy.ndarray
            The flux excess factor listed in the Gaia EDR3 archive.
            
        Returns
        -------
        
        The corrected value for the flux excess factor, which is zero for "normal" stars.
        
        Example
        -------
        
        phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
        """
        
        if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
            bp_rp = np.float64(bp_rp)
            phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
        
        if bp_rp.shape != phot_bp_rp_excess_factor.shape:
            raise ValueError('Function parameters must be of the same shape!')
            
        do_not_correct = np.isnan(bp_rp)
        bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
        greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
        redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
        
        correction = np.zeros_like(bp_rp)
        correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange],2)
        correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange],2) \
            - 0.005879*np.power(bp_rp[greenrange],3)
        correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
        
        return phot_bp_rp_excess_factor - correction
    
    
   
    def get_PS1_data(self,cluster,rad=4,delay=10):
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
        #center=self.resolve(cluster)
        #RA=center[0]
        #DEC=center[1]
        query = """select o.objID, o.raMean, o.decMean, o.nDetections, \
        o.ng, o.nr, o.ni, o.nz, o.ny, m.gMeanPSFMag, m.rMeanPSFMag, m.iMeanPSFMag, \
        m.zMeanPSFMag, m.yMeanPSFMag, \
        m.gMeanPSFMagErr, m.rMeanPSFMagErr, m.iMeanPSFMagErr, \
        m.yMeanPSFMagErr, m.zMeanPSFMagErr \
        into mydb.{3}_python \
        from fGetNearbyObjEq({0},{1},{2}) nb inner join ObjectThin as o \
        on o.objid=nb.objid and o.nDetections>1 \
        inner join MeanObject as m on o.objid=m.objid and o.uniquePspsOBid=m.uniquePspsOBid"""\
        .format(RA,DEC,rad*60,cluster)
        jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
        results = jobs.submit(query, task_name="{0}_python".format(cluster),estimate=30)
        time.sleep(delay*60)
        jobs.request_and_get_output("mydb.{0}_python".format(cluster),"FITS",\
                                    "{0}_PS.fits".format(cluster))
        os.chdir(orig_path)
        jobs.drop_table(cluster+"_python")


    def stilts(self,cluster, tol_a = 2 ,join = "1and2"):
        ''' 
        Cross matching data from the outputs from SExtractor.
     python command of stilts. The output from this routine will give a file that
     contains matches from both the lists supplied. 
     tol_a is the tolerance between sky locations in arcsecs
    join indicates the type of output. See Stilts documentation for types.
    '''
        tmatch2 ="java -jar /Users/pete/Documents/Software/stilts.jar tmatch2 \
         matcher=sky params={0} in1={1}/{1}_gaia_EXT.fits ifmt1='fits' values1='ra dec' in2={1}/{1}_PS_EXT.fits \
         ifmt2='fits' values2='raMEAN decMEAN' out={1}/{1}_G_PS_ready.fits ofmt='fits' join={2}".format(tol_a, cluster, join)
        os.system(tmatch2)
        print("Complete!")


    def gaia_u_clean(self,cluster):
        '''
        Cleans the gaia data as well as calculate the photometric uncertainties.
        Input parameters:
        cluster - the cluster we wish to analyse.
        '''
        center=self.resolve(cluster)
        RA=center[0]
        DEC=center[1]   
        print(RA,DEC)
        try:
            f_in=fits.open("{0}/{0}_G_PS_ready.fits".format(cluster))
            print("Loaded in {0}_G_PS_ready.fits for cleaning.".format(cluster)) 
            ps1=True           
        except FileNotFoundError:
            f_in=fits.open("{0}/{0}_gaia_EXT.fits".format(cluster))
            print("Loaded in {0}_gaia_EXT.fits for cleaning.".format(cluster))
            ps1=False
        f_data=Table(f_in[1].data)
        f_data=f_data[np.isnan(f_data['bp_rp'])==False]
        f_data=f_data[np.isnan(f_data['pmra'])==False]
        u_me=(f_data['astrometric_chi2_al']/(f_data['astrometric_n_good_obs_al']-5))**0.5
        f_data['u']=u_me
        tw=np.asarray((self.ruwe['#g_mag'],self.ruwe['bp_rp'])).T
        tw1=np.asarray((self.ruwe['#g_mag'],self.ruwe['bp_rp'],self.ruwe['u0'])).T
        RUWE_Tree=KDTree(tw)
        ones=np.ones((len(f_data)))
        colmag_ar=np.asarray((f_data['phot_g_mean_mag'],f_data['bp_rp'])).T
        AD=tw1[RUWE_Tree.query(colmag_ar)[1]][:,2]
        f_data['u0']=AD
        f_data['ruwe']=f_data['u']/f_data['u0']
        f_data.write("{0}/{0}_Gaia_ruwe.fits".format(cluster),format="fits",overwrite=True)
        exponent=np.exp(-0.2*(f_data['phot_g_mean_mag']-19.5))
        f_data=f_data[(f_data['parallax']-3*f_data['parallax_error'])<=0]
        #Condition = 1.2*np.maximum(ones,exponent)
        #new_dat= f_data[f_data['u'] <= Condition]
        f_data=f_data[f_data['ruwe']<1.4]
        #new_dat.write("{0}/{0}_Gaia_cleaned.fits".format(cluster),format="fits",overwrite=True)
        if ps1==False:
            print("Removing phot_excess")
            f_data=f_data[(f_data['phot_bp_rp_excess_factor']<(1.3+0.06*f_data['bp_rp']*f_data['bp_rp']))\
            &(f_data['phot_bp_rp_excess_factor']>(1.0+0.015*f_data['bp_rp']*f_data['bp_rp']))]
            f_data.write("{0}/{0}_Gaia_cleaned.fits".format(cluster),format="fits",overwrite=True)
        else:
            print("Using PS1 photometry")
            f_data=f_data[f_data['gMeanPSFMag']>=0]
            f_data=f_data[f_data['iMeanPSFMag']>=0]
            f_data.write("{0}/{0}_Gaia_cleaned.fits".format(cluster),format="fits",overwrite=True)
        '''
              
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
        '''

    def gnom_pm_correct(self,cluster,has_space="no",altname=False,specfile=""):
        center=self.resolve(cluster)
        RA=center[0]
        DEC=center[1]
        print(RA,DEC)
        f_in=fits.open("{0}/{0}_bays_ready_FULL.fits".format(cluster))
        f_data=Table(f_in[1].data)
        for i in PB.progressbar(range(len(f_data))):
            a=np.deg2rad(f_data['ra'][i])
            racl=np.deg2rad(f_data['ra'][i]-RA)
            de=np.deg2rad(f_data['dec'][i])
            decl=np.deg2rad(DEC)
            pmr=f_data['pmra'][i]
            f_data['pmra_g'][i]=pmr*np.cos(racl)-f_data['pmdec'][i]*np.sin(de)*np.sin(racl)
            f_data['pmdec_g'][i]=pmr*np.sin(decl)*np.sin(racl)+f_data['pmdec'][i]*\
            (np.cos(de)*np.cos(decl)+np.sin(de)*np.sin(decl)*np.cos(racl))
        f_data.write("{0}/{0}_bays_ready_FULL.fits".format(cluster),format="fits",overwrite=True)    
        f_data=f_data['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
            'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr','p_cmdM','p_cmdC']
        f_data.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)    
    


    def GAIA_reddening(self,file,ps1=False,first=False):
        '''
        Create a de-reddened catalog. Values from Malhan et al. 2018 or from
        the e Padova model site: http://stev.oapd.inaf.it/cgi-bin/cmd 2.8. 
        '''
        f_in=fits.open("{0}/{0}_gaia.fits".format(file))
        f_data=Table(f_in[1].data)
        X=f_data['ra']
        Y=f_data['dec']
        coords=SkyCoord(X,Y,unit='deg',frame='icrs')
        sfd = SFDQuery()
        DUST=sfd(coords)
        print("GOT DUST MAP")
        AG=0.85926
        ABP=1.06794
        ARP=0.65199
        RV=3.1
        dmag=RV*DUST
        cbp1,cbp2,cbp3,cbp4,cbp5,cbp6,cbp7=1.1517,-0.0871,-0.0333,0.0173,-0.0230,0.0006,0.0043
        crp1,crp2,crp3,crp4,crp5,crp6,crp7=0.6104,-0.0170,-0.0026,-0.0017,-0.0078,0.00005,0.0006
        cg1,cg2,cg3,cg4,cg5,cg6,cg7=0.9761,-0.1704,0.0086,0.0011,-0.0438,0.0013,0.0099
        print("correcting dust")
        g_0=f_data['phot_g_mean_mag']
        bp_0=f_data['phot_bp_mean_mag']
        rp_0=f_data['phot_rp_mean_mag']
        bprp=f_data['bp_rp']
        kbp=1.1517 + -0.0871*bprp + -0.0333*bprp*bprp + 0.0173*bprp*bprp*bprp + -0.0230*dmag + 0.0006*dmag*dmag + 0.0043*dmag*bprp
        krp=0.6104 + -0.0170*(bprp) + -0.0026*bprp*bprp + -0.0017*bprp*bprp*bprp + -0.0078*dmag + 0.00005*dmag*dmag + 0.0006*dmag*bprp
        kg=0.9761 + -0.1704*(bprp) + 0.0086*bprp*bprp + 0.0011*bprp*bprp*bprp + -0.0438*dmag + 0.0013*dmag*dmag + 0.0099*dmag*bprp
        g_0m=g_0-dmag*AG
        bp_0m=bp_0-dmag*ABP
        rp_0m=rp_0-dmag*ARP
        f_data['g_0_malin']=g_0m
        f_data['bp_0_malin']=bp_0m
        f_data['rp_0_malin']=rp_0m
        f_data['EB_V_G']=DUST
        f_data['g_0']=g_0-dmag*kg
        f_data['bp_0']=bp_0-dmag*kbp
        f_data['rp_0']=rp_0-dmag*krp
        f_data['kbp']=kbp
        f_data['krp']=krp
        f_data['kg']=kg
        f_data['ag']=dmag
        print("writing out")
        f_data.write("{0}/{0}_Gaia_EXT.fits".format(file),format="fits",overwrite=True)
  

    def exp_func(self,x,a,b,c):
        '''exponential function'''
        return a*np.exp(b*x)+c
    
    def lin_func(self,x,a,b):
        return a*x+b


    def PS1_reddening(self,file,ps1=False):
        '''
        Create a de-reddened catalog. 
        Values from Green et al. 2017
        '''
        f_in=fits.open("{0}/{0}_PS.fits".format(file))
        f_data=Table(f_in[1].data)
        X=f_data['raMean']
        Y=f_data['decMean']
        coords=SkyCoord(X,Y,unit='deg',frame='icrs')
        sfd = SFDQuery()
        DUST=sfd(coords)
        Rg=3.384
        Rr=2.483
        Ri=1.838
        Rz=1.414
        Ry=1.126
        dmag=DUST
        g_0=f_data['gMeanPSFMag']
        bp_0=f_data['rMeanPSFMag']
        rp_0=f_data['iMeanPSFMag']
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
        f_data=f_data[np.isnan(f_data['gMeanPSFMagErr'])==False]
        f_data=f_data[np.isnan(f_data['rMeanPSFMagErr'])==False]
        f_data=f_data[np.isnan(f_data['iMeanPSFMagErr'])==False]
        #f_data=f_data[f_data['g_R0']<(22-np.argmax(f_data['EB_V_P'])*Rg)]    
        f_data.write("{0}/{0}_PS_EXT.fits".format(file),format="fits",overwrite=True)


    def func(self,x,a,b,c,d):
        return a+b*x+c/(x-d)


    def catalog_selection_v2(self,file,survey,rin,rout,cl_pmra,cl_pmdec,shift=1.5):
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


    def catalog_selection(self,cluster,rin=0,pmsel=1):
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
        data=fits.open(glob.glob("/Users/pete/Documents/Software/own_codes/edr3code_working/cluster_pipeline.fits")[0])
        lf=Table(data[1].data)
        for i in range(len(lf)):
            lf['Name'][i]=lf['Name'][i].split(' ')[0]
        dat1=lf[lf['Name']==cluster]        
        pmra=dat1['pmra'][0]
        pmdec=dat1['pmde'][0]
        rout=dat1['rta'][0]/60
        dist=dat1['R_Sun'][0]*1000
        #vvhb=dat1['V_HB']
        #rout=np.rad2deg(np.arctan(rout/dist))
        rin=0.2*rout
        f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
        f_data=Table(f_in[1].data)
        f_data=f_data[f_data['dist']>(rin)]
        f_data=f_data[f_data['dist']<(rout)]
        f_data=f_data[((f_data['pmra_g']-pmra)**2+(f_data['pmdec_g']-float(pmdec))**2)<=pmsel**2]
        f_data.write("{0}/{0}_isochrone_select.fits".format(cluster),format="fits",overwrite=True)
    

    
    def weightCMD_AUTOISO(self,cluster,age13=False,phot_limit=19,dback=2,rin=0.5,pm_sel=0.5,source="Parsec",phot=1.6,mag_lit=16,colerr_cor=15,field_maglim=16,loadin=False,grabiso=True):
        data=fits.open(glob.glob("/Users/pete/Documents/Software/own_codes/edr3code_working/cluster_pipeline.fits")[0])
        lf=Table(data[1].data)
        for i in range(len(lf)):
            lf['Name'][i]=lf['Name'][i].split(' ')[0]
        dat1=lf[lf['Name']==cluster]        
        feh=dat1['[Fe/H]'][0]
        if age13==False:
            age=dat1['Age'][0]
        else:
            age=13e9
        dist=dat1['R_Sun'][0]*1000   
        print(feh,age,dist)
        if os.path.isfile(cluster+"/"+cluster+"_PS.fits")==True:
            survey="PS1"
        else:
            survey="gaia"
        f_in=fits.open("{0}/{0}_Gaia_cleaned.fits".format(cluster))
        fdata=Table(f_in[1].data)
       #self.fdata=fdata
        if survey=="PS1":
             print("Photometry: PS1")
             mag_string='i_R0'
             mag_err_string='iMeanPSFMagErr'
             colb_string='g_R0'
             colb_err_string='gMeanPSFMagErr'
             colr_string='i_R0'
             colr_err_string='iMeanPSFMagErr'
             fdata=fdata[fdata[colr_err_string]>=0]
             fdata=fdata[fdata[colb_err_string]>=0]
             #fdata=fdata[(fdata['g_R0']-fdata['i_R0'])>0]
             print("Yee:")
             fp=fdata[(fdata[mag_string]>=mag_lit)]
             #self.fp=fp
             print("nee")
             G=unumpy.uarray(fp['i_R0'],fp['iMeanPSFMagErr'])
             C=unumpy.uarray(fp['g_R0'],fp['gMeanPSFMagErr'])-unumpy.uarray(fp['i_R0'],fp['iMeanPSFMagErr'])
             mag=unumpy.nominal_values(G)
             magerr=unumpy.std_devs(G)
             col=unumpy.nominal_values(C)
             colerr=unumpy.std_devs(C)
        elif survey=="gaia":
             print("Photometry: Gaia")
             mag_string='g_0'
             mag_err_string='g_err'
             colb_string='bp_0'
             colb_err_string='bp_err'
             colr_string='rp_0'
             colr_err_string='rp_err'
             #fp=fdata[(fdata['bp_0']-fdata['rp_0'])>0]
             #fp=fp[(fp[mag_string]>=mag_lit)]
             G=unumpy.uarray(fp['g_0'],fp['g_err'])
             C=unumpy.uarray(fp['bp_0'],fp['bp_err'])-unumpy.uarray(fp['rp_0'],fp['rp_err'])
             mag=unumpy.nominal_values(G)
             magerr=unumpy.std_devs(G)
             col=unumpy.nominal_values(C)
             colerr=unumpy.std_devs(C)
        if survey=="PS1":
            fdata=fdata[np.isnan(fdata['g_R0'])==False]
        f_in.close()
        #rin=rin*rdist
        iso_file_pick=open("isochrones/{0}_isochrone_{1}_*.iso".format(cluster,age),'rb')
        iso_file=cPickle.load(iso_file_pick)
        if survey=="PS1":
            coliso=iso_file['gP1mag']-iso_file['iP1mag']
            magiso=iso_file['iP1mag'] + 5*np.log10(dist)-5
            iso=interp1d(magiso,coliso,axis=0, fill_value="extrapolate")
        if survey=="gaia":
            coliso=iso_file['Gaia_BP']-iso_file['Gaia_RP'] 
            magiso=iso_file['Gaia_G'] + 5*np.log10(dist)-5
            iso=interp1d(magiso,coliso,axis=0, fill_value="extrapolate")
        def func(x,a,b):
            r=iso(x+a)+b
            return r
        print("Gathered isochrone")
        print("Now, shifting isochrone to fit the data.\nNow, select your cluster sample.")
        print(mag_string,mag_err_string)
        colerr_array=np.zeros((len(magiso)-1,2))
        #fp.write("{0}/{0}_WISO_select_catalog.fits".format(cluster),format="fits",overwrite=True)
        for i in range(len(magiso)-1):
            up=magiso[i]
            down=magiso[i+1]
            ad=fdata[fdata[colr_err_string]>=0]
            ad=ad[ad[colb_err_string]>=0]
            ad=ad[ad[mag_string]<=up]
            ad=ad[ad[mag_string]>=down]
            clG=unumpy.uarray(ad[mag_string],ad[mag_err_string])
            clC=unumpy.uarray(ad[colb_string],ad[colb_err_string])-unumpy.uarray(ad[colr_string],ad[colr_err_string])
            cl_mag=unumpy.nominal_values(clG)
            cl_magerr=unumpy.std_devs(clG)
            cl_col=unumpy.nominal_values(clC)
            cl_colerr=unumpy.std_devs(clC)
            try:
                colerr_array[i,0]=(up+down)/2.
                colerr_array[i,1]=np.median(cl_colerr)
            except ValueError:
                colerr_array[i,0]=(up+down)/2.
                colerr_array[i,1]=0
        colerr_array=colerr_array[colerr_array[:,1]!=0]
        colerr_array=colerr_array[~np.isnan(colerr_array[:,1])]
        colerr_array=colerr_array[colerr_array[:,0]<=20]
        colerr_array=colerr_array[colerr_array[:,0]>=14]
        self.de=colerr_array
        Ef1,Ef2=sop.curve_fit(self.exp_func,colerr_array[:,0],colerr_array[:,1])
        clG=unumpy.uarray(fdata[mag_string],fdata[mag_err_string])
        clC=unumpy.uarray(fdata[colb_string],fdata[colb_err_string])-unumpy.uarray(fdata[colr_string],fdata[colr_err_string])
        cl_mag=unumpy.nominal_values(clG)
        cl_magerr=unumpy.std_devs(clG)
        cl_col=unumpy.nominal_values(clC)
        cl_colerr=unumpy.std_devs(clC)
        new_colerr=self.exp_func(cl_mag,Ef1[0],Ef1[1],Ef1[2])
        try:
            newerr=self.exp_func(mag,Ef1[0],Ef1[1],Ef1[2])
            fp_Colerr=self.exp_func(fp[mag_string],Ef1[0],Ef1[1],Ef1[2])
        except:
            print("loaded iso - no fp catalogue yet")
        for i in range(len(new_colerr)):
            if fdata[mag_string][i]<=colerr_cor:
                new_colerr[i]=self.exp_func(colerr_cor,Ef1[0],Ef1[1],Ef1[2])
            else:
                new_colerr[i]=new_colerr[i]
        fdata['new_err']=new_colerr
        try:
            fp['new_err']=fp_Colerr
            fp1=fdata[(fdata['bp_0']-fdata['rp_0'])>0]
            #fp1=fp1[(fp1[mag_string]>=mag_lit)]
            fp1=fp1[(fp1[colb_string]-fp1[colr_string])<=1.0]
            fp1COL=fp1[colb_string]-fp1[colr_string]
            fp1MAG=fp1[mag_string]
        except:
            print("Loading in Isochrone and respective catalogues")
        if loadin==True:
            fe=fits.open("{0}/{0}_isochrone_select.fits".format(cluster))
            fp=Table(fe[1].data)
            fp1=Table(fe[1].data)
            fp=fp[fp[colr_err_string]>=0]
            fp=fp[fp[colb_err_string]>=0]
            f1p=fp1[fp1[colr_err_string]>=0]
            fp1=fp1[fp1[colb_err_string]>=0]
            fe.close()
        else:
            fp1.write("{0}/{0}_iso_select_catalog.fits".format(cluster),format="fits",overwrite=True)
        fp=fp[fp[mag_string]>(np.ceil(min(magiso)))]
        fp1=fp1[fp1[mag_string]>(np.ceil(min(magiso)))]
        fp1=fp1[(fp1[colb_string]-fp1[colr_string])>0]
        #fp1=fp1[(fp1[colb_string]-fp1[colr_string])>0]
       # fp=fp[fp[mag_string]<20.5]
        #fp1=fp1[fp1[mag_string]<20.5]
        fdata=fdata[fdata[mag_string]>(np.ceil(min(magiso)))]
        if survey=="gaia":
            fp=fp[fp[mag_string]<=20]
            fp1=fp1[fp1[mag_string]<=20]
            #fdata=fdata[fdata[mag_string]>(np.ceil(min(magiso)))]           
        fp1COL=fp1[colb_string]-fp1[colr_string]
        fp1MAG=fp1[mag_string]
        fp_Colerr=self.exp_func(fp[mag_string],Ef1[0],Ef1[1],Ef1[2])    
        fp1_Colerr=self.exp_func(fp1[mag_string],Ef1[0],Ef1[1],Ef1[2])    
        #for i in range(len(fp_Colerr)):
         #   if fp[mag_string][i]<=colerr_cor:
         #       fp_Colerr[i]=self.exp_func(colerr_cor,Ef1[0],Ef1[1],Ef1[2])
         #   else:
         #       fp_Colerr[i]=fp_Colerr[i]
        fp['new_err']=fp_Colerr
        
        self.func=func
        self.magiso,self.coliso=magiso,coliso
        #self.x,self.y=mag,col
        self.cl_mag,self.cl_col=cl_mag,cl_col
        self.xx,self.yy,self.colerr=fp1MAG,fp1COL,fp1_Colerr
        #plt.plot(self.xx,self.yy,'.')
        #plt.plot(self.xx,func())
        f1,f2=sop.curve_fit(func,fp1MAG,fp1COL,sigma=(1/fp1_Colerr),bounds=([-0.5,-0.05],[0.5,0.05]))
        self.f1=f1
        print(f1)
        #f1=[0,-0.1]
        #fdata=fdata[fdata[mag_string]>12]
        #self.da=fdata
        #f1=[0,0]
        fdata=fdata[fdata[mag_string]>((min(magiso)-4))]
        fdata['w_iso']=((fdata[colb_string]-fdata[colr_string])-func(fdata[mag_string],f1[0],f1[1]))/(fdata['new_err'])
        fp['w_iso']=((fp[colb_string]-fp[colr_string])-func(fp[mag_string],f1[0],f1[1]))/(fp['new_err'])
        #self.wiso=(col-func(mag,f1[0],f1[1]))/newerr
        fdata.write("{0}/{0}_FULL_catalog.fits".format(cluster),format="fits",overwrite=True)
        colerr_array=np.zeros((len(magiso)-1,2))
        G=models.Linear1D()
        fit_t = fitting.LevMarLSQFitter()
        for k in range(1):
            colerr_array=np.zeros((len(magiso)-1,2))
            for i in range(len(magiso)-60):
                up=magiso[(i+30)-30]
                down=magiso[(i+30)+30]
                ad=fp[fp[mag_string]<=up]
                ad=ad[ad[mag_string]>=down]
                #ad['new_err']=self.exp_func(ad[mag_string],Ef1[0],Ef1[1],Ef1[2])
                clG=unumpy.uarray(ad[mag_string],ad[mag_err_string])
                clC=unumpy.uarray(ad[colb_string],ad[colb_err_string])-unumpy.uarray(ad[colr_string],ad[colr_err_string])
                cl_mag=unumpy.nominal_values(clG)
                cl_magerr=unumpy.std_devs(clG)
                cl_col=unumpy.nominal_values(clC)
                cl_colerr=ad['new_err']
                awiso=(cl_col-func(cl_mag,f1[0],f1[1]))/cl_colerr
                #print("step:{0}".format(k))
               # print(len(magiso),up,down)
                try:
                    hst,bns=np.histogram(awiso,bins=np.arange(-30,30,0.001))
                    mid_bns=bns[:-1]+0.0005
                    g = models.Gaussian1D()
                    fit_t = fitting.LevMarLSQFitter()
                    t1 = fit_t(g, mid_bns, hst)
                    #print(t1.stddev.value)
                    colerr_array[i,0]=(up+down)/2.
                    colerr_array[i,1]=t1.stddev.value
                except:
                    colerr_array[i,0]=(up+down)/2.
                    colerr_array[i,1]=0
            colerr_array=colerr_array[colerr_array[:,1]!=0]
            colerr_array=colerr_array[~np.isnan(colerr_array[:,1])]
            colerr_array=colerr_array[colerr_array[:,0]<=20]
            colerr_array=colerr_array[colerr_array[:,0]>=14]
            colerr_array=colerr_array[colerr_array[:,1]>10**(-30)]
            self.wisoa=colerr_array
           # F1,F2=sop.curve_fit(self.exp_func,colerr_array[:,0],colerr_array[:,1],sigma=(np.sqrt(colerr_array[:,1])),bounds=([0,-np.inf,0],[np.inf,0,np.inf]))
            XL=np.log10(colerr_array[:,0])
            YL=np.log10(colerr_array[:,1])
            #F1,F2=sop.curve_fit(self.exp_func,colerr_array[:,0],colerr_array[:,1],bounds=([0,-np.inf,0],[np.inf,0,np.inf]))
            tt=fit_t(G,XL,YL)
            self.tt=tt
           #if k==0:
                #fp=fp[((abs(fp['w_iso']))<=(10**tt.intercept.value*abs(fp['w_iso'])**tt.slope.value))]
        #fp.write("{0}/{0}_WISO_select_catalog.fits".format(cluster),format="fits",overwrite=True)
        fdata.write("{0}/{0}_full_cat.fits".format(cluster),format="fits",overwrite=True)  
        fdata=fdata[(fdata[colb_string]-fdata[colr_string])<phot]
        fdate=fdata[fdata['dist']>dback]
        if survey=="gaia":
            cd1=np.arange(17.3,19.9,0.2)
            cd2=np.arange(13,15.5,0.5)
            cd=np.concatenate((cd2,cd1))
            #cd=cd1
        elif survey=="PS1":
            cd1=np.arange(16,25,0.2)
            cd2=np.arange(13,16,0.5)
            cd=np.concatenate((cd2,cd1))
        else:
            print("blurg")
        colerr_array=np.zeros((len(cd),3))
        for i in range(len(cd)-1):
            #gr=f1[f1['g_0']>cd[i]]
            #gr=gr[gr['g_0']<=cd[i+1]]
            gr=fdate[fdate[mag_string]>cd[i]]
            gr=gr[gr[mag_string]<=cd[i+1]]
            try:
                hst,bns=np.histogram(gr['w_iso'],bins=np.arange(-500,500,0.1))
                mid_bns=bns[:-1]+0.05
                g = models.Gaussian1D()
                fit_t = fitting.LevMarLSQFitter()
                t1 = fit_t(g, mid_bns, hst)
                #print(t1.stddev.value)
                colerr_array[i,0]=(cd[i+1]+cd[i])/2.
                colerr_array[i,1]=t1.stddev.value
                colerr_array[i,2]=t1.mean.value
            except:
                colerr_array[i,0]=(cd[i+1]+cd[i])/2.
                colerr_array[i,1]=0
                colerr_array[i,2]=0
        self.ce1=colerr_array
        colerr_array=colerr_array[colerr_array[:,1]!=0]
        colerr_array=colerr_array[colerr_array[:,1]>0]
        colerr_array=colerr_array[colerr_array[:,1]!=1]
        colerr_array=colerr_array[colerr_array[:,2]!=0]
        colerr_array=colerr_array[colerr_array[:,2]!=1]
        self.ce2=colerr_array
        Test_ARy=colerr_array[:,2]
        if len(np.log10(Test_ARy[~np.isnan(Test_ARy)]))<=2:
            print("reversing for fitting")
            colerr_array[:,2]=-colerr_array[:,2]
        G=models.Linear1D()
        fit_t = fitting.LevMarLSQFitter()
        print("Did try work")
        std_array=colerr_array[:,0:2]
        men_array=colerr_array[:,(0,2)]
        men2_array=colerr_array[:,(0,2)]
        #men_array[:,1]=-men_array[:,1]
        std_array=np.log10(std_array)
        men_array=np.log10(men_array)
        std_array=std_array[std_array[:,1]>=0]
        men_array=men_array[men_array[:,1]>=0]
        std_array=std_array[~np.isnan(std_array[:,1])]
        men_array=men_array[~np.isnan(men_array[:,1])]
        
        print(men_array)
        print(std_array)
        tM=fit_t(G,men_array[:,0],men_array[:,1])
        tS=fit_t(G,std_array[:,0],std_array[:,1])    
        ZZ=np.polyfit(men2_array[:,0],men2_array[:,1],3)
        RT=np.poly1d(ZZ)
        self.RT=RT
        fdata['p_cmdM_cubic']=stats.norm.pdf(fdata['w_iso'],loc=(RT(fdata[mag_string])),scale=(10**(tS.intercept.value)*fdata[mag_string]**(tS.slope.value)))
        fdata['p_cmdM']=stats.norm.pdf(fdata['w_iso'],loc=(10**(tM.intercept.value)*fdata[mag_string]**(tM.slope.value)),scale=(10**(tS.intercept.value)*fdata[mag_string]**(tS.slope.value)))
        fdata['p_cmdC']=stats.norm.pdf(fdata['w_iso'],loc=0,scale=(10**(tt.intercept.value)*fdata[mag_string]**(tt.slope.value)))
        self.dat=fdata
        if survey=="gaia":
            fdata=fdata[fdata[mag_string]<=phot_limit]
        fdata.write("{0}/{0}_bays_ready_FULLmid.fits".format(cluster),format="fits",overwrite=True)  
        fdat=fdata[((abs(fdata['w_iso']))<=(3*10**tt.intercept.value*fdata[mag_string]**tt.slope.value))]
        fdat.write("{0}/{0}_testing_EXP_FULLmid.fits".format(cluster),format="fits",overwrite=True) 
        #fdaa=fdata['ra_g','dec_g','pmra_g','pmdec_g','pmra_error','pmdec_error','dist',\
        #'pmra_pmdec_corr','w_iso','pmra','pmdec','ra_error','dec_error','ra_dec_corr','p_cmdM','p_cmdC']
        #fdaa.write("{0}/{0}_bays_ready_mid.fits".format(cluster),format="fits",overwrite=True)
        #CD=np.array([[tt.slope.value,tt.intercept.value],[tM.slope.value,tM.intercept.value],[tS.slope.value,tS.intercept.value]])
        #ßascii.write(CD,"{0}_exp_weight_mean_stf_fit_mid.txt".format(cluster))
        print("except error")
        std_array=colerr_array[:,0:2]
        men_array=colerr_array[:,(0,2)]
        #men_array[:,1]=-men_array[:,1],
       # std_array=np.log10(std_array)
       # men_array=np.log10(men_array)
        #std_array=std_array[std_array[:,1]>=0]
        #men_array=men_array[men_array[:,1]>=0]       
        std_array=std_array[~np.isnan(std_array[:,1])]
        men_array=men_array[~np.isnan(men_array[:,1])]
        tM=fit_t(G,men_array[:,0],men_array[:,1])
        tS=fit_t(G,std_array[:,0],std_array[:,1])
        fdata['p_cmdM']=stats.norm.pdf(fdata['w_iso'],loc=(10**(tM.intercept.value)*fdata[mag_string]**(tM.slope.value)),scale=(10**(tS.intercept.value)*fdata[mag_string]**(tS.slope.value)))
        fdata['p_cmdC']=stats.norm.pdf(fdata['w_iso'],loc=0,scale=(10**(tt.intercept.value)*fdata[mag_string]**(tt.slope.value)))
        fdata.write("{0}/{0}_bays_ready_FULL_out.fits".format(cluster),format="fits",overwrite=True)  
        #fdat=fdata[((abs(fdata['w_iso']))<=(3*10**tt.intercept.value*fdata[mag_string]**tt.slope.value))]
        #fdat.write("{0}/{0}_testing_EXP_FULLout.fits".format(cluster),format="fits",overwrite=True) 
        fdaa=fdata['ra_g','dec_g','pmra_g','pmdec_g','pmra_g_err','pmdec_g_err','dist',\
        'pmra_pmdec_g_corr','w_iso','pmra','pmdec','ra_error','dec_error','p_cmdM','p_cmdC']
        fdaa.write("{0}/{0}_bays_ready.fits".format(cluster),format="fits",overwrite=True)
        CD=np.array([[tt.slope.value,tt.intercept.value],[tM.slope.value,tM.intercept.value],[tS.slope.value,tS.intercept.value]])
        ascii.write(CD,"{0}_exp_weight_mean_stf_fit_out.txt".format(cluster))
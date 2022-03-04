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
from scipy.stats import skewnorm
from astropy.units import Quantity
#from astroquery.gaia import Gaia
#from astroquery.vizier import Vizier
from astropy import coordinates
import _pickle as cPickle
#from dustmaps.sfd import SFDQuery
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
from multinest_baseCMD_test_rmin import PyNM
from mpi4py import MPI
import corner

# ---------------------------------------------------
# Definitions
# ---------------------------------------------------
class PyMN_RUN(PyNM):
    def __init__(self,cluster,prior,inner_radii,cr,tr,cre,tre,lh,dist,survey,pmcsel,select=True,pm_sel="gnom",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None,pmsel=1,phot=1.6,SET=0.5,SSE=0.8,CEF=False):
        PyNM.__init__(self,cluster,prior,inner_radii,cr,tr,lh,survey,pmcsel,select=select,pm_sel=pm_sel,live_points=live_points,existing=existing,rmax=rmax,Fadd=Fadd,preking=preking,outbase_add=outbase_add,pmsel=1,phot=phot)
#PyNM.__init__(self,cluster,radius,prior,inner_radii,sample_size,cr,tr,select=True,pm_sel="norm",live_points=400,existing=False,rmax=4.,Fadd=None,preking=False,outbase_add=None)
        self.Parameters=["x_pm,cl","y_pm,cl","x_dsp,cl","y_dsp,cl","x_pm,MW","y_pm,MW","x_dsp,MW","y_dsp,MW","f_cl+ex","f_cl","theta","k","theta2","k2","gamma","rc","rt"]
        self.N_params = len(self.Parameters)
        self.survey=survey
        self.PCMD_CL=self.M2['p_cmdC']
        self.PCMD_MW=self.M2[pmcsel]
        self.phot=phot
        self.survey=survey
        self.pmcsel=pmcsel
        self.SSE=SSE
        self.SET=SET
        self.CEF=CEF
        self.rmin=self.lh
        self.TR=self.M2['kg_tr']
        self.CR=self.M2['kg_cr']
        #self.King=where(self.dist<=self.TR,self.L_sat_king(self.x_ps,self.y_ps,self.CR,self.TR),1e-99)
        #self.Plum=where(self.dist<=self.TR,self.L_sat_spat_PL(self.x_ps,self.y_ps,self.CR,0,self.rmax),1e-99)
        self.rmax=rmax
        print("RMAX={0}".format(self.rmax))
        print("RMIN={0}".format(self.rmin))

    def PyMultinest_run(self,resume=False):
        print("Run PyMultiNest")
        try:
            tstart=time.time()
            pymultinest.run(self.loglike_ndisp, self.Prior, self.N_params,outputfiles_basename=self.outbase_name, \
            resume = resume, verbose = True,n_live_points=self.Live_Points,const_efficiency_mode=self.CEF,\
            evidence_tolerance = self.SET, sampling_efficiency = self.SSE)
            json.dump(self.Parameters, open("{0}_params.json".format(self.outbase_name), 'w')) # save parameter names
            tend=time.time()
            print("time taken: {0}".format(tend-tstart))
        except FileNotFoundError:
            print("Set-up not performed. Please run PyMultinest_setup.")
    
    def PyMultinest_results(self,setup="complete"):
        try:
            result = pymultinest.solve(LogLikelihood=self.loglike_ndisp, Prior=self.Prior, 
            n_dims=self.N_params, outputfiles_basename=self.outbase_name, verbose=True,n_live_points=self.Live_Points)
            f=open("{0}_parameter_summary.txt".format(self.outbase_name),'w')
            print('parameter values:')
            for name, col in zip(self.Parameters, result['samples'].transpose()):
                print('%15s : %.5f +- %.5f' % (name, col.mean(), col.std()))
                f.write('%15s : %.5f +- %.5f \n' % (name, col.mean(), col.std()))
        except FileNotFoundError:
            print("Set-up not performed. Please run PyMultinest_setup.")
              
        
        
    def PyMultinest_plots(self,setup="complete",save_fig=True):
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            nproc = MPI.COMM_WORLD.Get_size()

        except ImportError:
            rank = 0
            nproc = 1
        if rank==0:
            try:
                a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename=self.outbase_name)
                s = a.get_stats()
                #plt.clf()
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
            done=os.path.exists("{0}_{1}_post_dist.pdf".format(self.cluster,self.outbase_name))
            for proc in range(1,nproc):
                MPI.COMM_WORLD.send(done,dest=proc)
        else:
            print("Running membership on rank one.")
            done = MPI.COMM_WORLD.recv(source=0)


    def PyMultinest_plots_corner(self,setup="complete",save_fig=True):
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            nproc = MPI.COMM_WORLD.Get_size()

        except ImportError:
            rank = 0
            nproc = 1
        if rank==0:
            try:
                a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename=self.outbase_name)
                self.nsx=a
                s = a.get_stats()
                #plt.clf()
                data = a.get_data()[:,2:]
                for i in range(len(data)):
                    data[i,10]=-57.296*data[i,10]+90
                    data[i,12]=-57.296*data[i,12]+90
                weights = a.get_data()[:,0]
                mask = weights > 1e-4
                modes = s['modes']
                parameters=["$\mu^{*}_{\\xi,cl}$","$\mu_{\eta,cl}$","$\sigma_{\mu^{*}_{\\xi},cl}$",\
                "$\sigma_{\mu_{\eta},cl}$","$\mu^{*}_{\\xi,MW}$","$\mu_{\eta,MW}$","$\sigma_{\mu^{*}_{\\xi},MW}$",\
                "$\sigma_{\mu_{\eta},MW}$","$f_{cl+ex}$","$f_{cl}$","$\\theta_{MW}$",\
                "$k_{MW}$","$\\theta_{ex}$","$k_{ex}$","$\gamma$","$r_{c}$","$r_{t}$"]
                figure=corner.corner(data[mask,:], weights=weights[mask],labels=parameters, show_titles=True,title_fmt='.3f')
                axes = np.array(figure.axes).reshape((self.N_params, self.N_params))
                for i in range(self.N_params):
                    m = s['marginals'][i]
                    ax = axes[i, i]
                    ax.set_title("{0}".format(parameters[i]))
                    ylim = ax.get_ylim()
                    y = min(ylim) +max(ylim)/10
                    if i == 10 or i==12:
                        center = -57.296*m['median']+90
                        low1, high1 = m['1sigma']
                        low1 =-57.296*low1+90
                        high1=-57.296*high1+90
                        print(i,center,center - low1,high1 - center)
                    else:
                        center=m['median']
                        low1, high1 = m['1sigma']
                    ax.errorbar(x=center, y=y,xerr=np.transpose([[center - low1, high1 - center]]),color='red', linewidth=2, marker='s')
                if save_fig==True:
                    plt.savefig("{0}_{1}_post_corner.pdf".format(self.cluster,self.outbase_name),format='pdf')
                else:
                    plt.show()
            except FileNotFoundError:
                print("Set-up not performed. Please run PyMultinest_setup.")
            done=os.path.exists("{0}_{1}_post_corner.pdf".format(self.cluster,self.outbase_name))
            for proc in range(1,nproc):
                MPI.COMM_WORLD.send(done,dest=proc)
        else:
            print("Running membership on rank one.")
            done = MPI.COMM_WORLD.recv(source=0)


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

    def L_pm_GC_moving(self,x_g,y_g,xc_g,yc_g,xt_g,yt_g,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff):
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
        ((((x_g+xc_g*xt_g)-x_pm)/(cv_pmraer))**2+(((y_g+yc_g*yt_g)-y_pm)/(cv_pmdecer))**2-\
        ((2*cv_coeff*((x_g+xc_g*xt_g)-x_pm)*((y_g+yc_g*yt_g)-y_pm))/(cv_pmraer*cv_pmdecer))))
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
        mc = (r *ah*ah* (ah*ah+rmax*rmax))/\
        (np.pi*rmax*rmax*((ah*ah+r*r)**2))
        return mc


    def L_sat_kingi(self,xt_g,yt_g,ah,rt):
        r=sqrt(xt_g**2+yt_g**2)
        mask=np.array((r<=rt))
        mc=r *( 1/(r*r+ah*ah)+1./(ah*ah+rt*rt)-2/(sqrt(ah*ah+r*r)*sqrt(ah*ah+rt*rt)))/\
        (pi*((rt**2+4*(ah-sqrt(ah**2+rt**2))*sqrt(ah**2+rt**2))/(ah**2+rt**2)\
        +log(1+rt**2/ah**2)))
        mc=mc*mask
        return mc


    def L_sat_king(self,xt_g,yt_g,ah,rt,rmin):
        r=sqrt(xt_g**2+yt_g**2)
        mask=np.array((r<=rt))
        nom=r*r*r+r*(rt*rt + ah*ah *(2-2*np.sqrt(1+r*r/(ah*ah))*np.sqrt(1+rt*rt/(ah*ah))))
        dem=np.pi*(r*r + ah*ah) * (rmin*rmin+3*rt*rt+ah*ah*(4-4*np.sqrt(1+rmin*rmin/(ah*ah))*np.sqrt(1+rt*rt/(ah*ah)))+(ah*ah + rt*rt)*(np.log(ah*ah+rmin*rmin)-np.log(ah*ah+rt*rt)))
        #mc=r *( 1/(r*r+ah*ah)+1./(ah*ah+rt*rt)-2/(sqrt(ah*ah+r*r)*sqrt(ah*ah+rt*rt)))/\
        #(pi*((rt**2+4*(ah-sqrt(ah**2+rt**2))*sqrt(ah**2+rt**2))/(ah**2+rt**2)\
        #+log(1+rt**2/ah**2)))
        mc=-nom/dem
        mc=mc*mask
        return mc


    def L_sat_grad(self,xt_g,yt_g,the,a,b):
        r=np.sqrt(xt_g*xt_g+yt_g*yt_g)
        #mask=np.array((r>=self.rmin))
        z=r*(a+b*r*np.cos(np.arctan2(yt_g,xt_g)-the))/(np.pi*a*(self.rmax*self.rmax-self.rmin*self.rmin))
        return z

    def exp_func(self,x,a,b,c):
        '''exponential function'''
        return a*np.exp(b*x)+c
    
    
    def L_cmd_cl(self,w_par,g_mag,colerr,cl_spread,pmra,pmdec,R,pra,pde,rin,rout,pmsel):
       '''
       sig_g = estimating the spread of the cluster distribution
       from the w-parameter.
       w_par = w_iso limit
       a,b and c =
       g_mag = dereddened g-magnitude
       '''
       likelihood=where((pmra<=(pra+pmsel))&(pmra>=(pra-pmsel))&(pmdec<=(pde+pmsel))\
                         &(pmdec>=(pde-pmsel))&(R>=rin)&(R<=rout),norm.pdf(w_par,0,cl_spread),0)
       #likelihood = (1./(sqrt(2*pi*(exp(g_mag*b)**2)))*exp(-(w_par**2/(2.*(exp(g_mag*b))**2.))))
       return likelihood

    def L_cmd_TS(self,w_par,g_mag,colerr,cl_spread):
       '''
       sig_g = estimating the spread of the cluster distribution
       from the w-parameter.
       w_par = w_iso limit
       a,b and c =
       g_mag = dereddened g-magnitude
       '''
       likelihood=norm.pdf(w_par,0,cl_spread)
       #likelihood = (1./(sqrt(2*pi*(exp(g_mag*b)**2)))*exp(-(w_par**2/(2.*(exp(g_mag*b))**2.))))
       return likelihood

    def L_cmd_mb(self,w_par,g_mag,ol_mean,ol_spread,colerr):
       '''
       sig_g = estimating the spread of the cluster distribution
       from the w-parameter.
       w_par = w_iso limit
       a,b and c =
       g_mag = dereddened g-magnitude
       '''
       likelihood=norm.pdf(w_par,ol_mean,sqrt(colerr*colerr+ol_spread))
       #likelihood = (1./(sqrt(2*pi*(exp(g_mag*b)**2)))*exp(-(w_par**2/(2.*(exp(g_mag*b))**2.))))
       return likelihood



    def L_sat_quad_r_j(self,xt_g,yt_g,the,gam,b):
        r=sqrt(xt_g*xt_g+yt_g*yt_g)
        theta=np.arctan2(yt_g,xt_g)
        nom=-(r**(1-gam))*(self.rmax**(-2+gam))*(-2+gam)*(2+b+b*np.cos(2*(the-theta)))
        dem=2*(2+b)*np.pi
        mc=nom/dem
        return mc
    
    def L_sat_quad_r_rmin(self,xt_g,yt_g,the,gam,b,tr):
        r=sqrt(xt_g*xt_g+yt_g*yt_g)
        rmin=tr*self.factor
        theta=np.arctan2(yt_g,xt_g)
        #nom=(-2+gam)*(rmin**(gam))*r*((rmin*r)**(-gam))*((rmin*self.rmax)**gam)*(2+b+b*np.cos(2*(the-theta)))
        #dem=2*(2+b)*np.pi*(-rmin**gam * self.rmax*self.rmax + rmin*rmin * self.rmax**gam) 
        nom=r**(1-gam)*(-2+gam)*(b+2*r**gam + b*np.cos(2*(the-theta)))
        dem=2*np.pi*self.rmax*self.rmax*(-2-b*self.rmax**(-gam)+gam)
        mc=nom/dem
        return mc

    def L_sat_quad_r_mni0(self,xt_g,yt_g,the,gam,b,tr):
        r=sqrt(xt_g*xt_g+yt_g*yt_g)
        theta=np.arctan2(yt_g,xt_g)
        rmin=self.rmin
        #mask=((r>=rmin))
        nom=-((-2+gam) * (-1+gam) * r * (1+rmin)**(gam) *\
                ((1+r) * (1+rmin))**(-gam) * ((1+self.rmax) * (1+rmin))**gam * (2+b+b*np.cos(2*(the-theta))))
        dem=2 * (2+b) * np.pi * ((1+self.rmax) * (1+(-1+gam)*self.rmax) * (1+rmin)**gam\
                - (1+self.rmax)**gam * (1+rmin) * (1+(-1+gam)*rmin))
        #(1+rmin)*(1+(-1+gam)*rmin)*(1+self.rmax)**gam -(1+rmin)**gam * (1+self.rmax) * (1+(-1+gam)*self.rmax))
        #nom=(-2+gam)*r*rmin**gam * (r*rmin)**-gam * (self.rmax *rmin)**gam*(2+b+b*np.cos(2*(the-theta)))
        #dem=2*(2+b)*np.pi*(-rmin**gam * self.rmax**2 + self.rmin**2 * self.rmax**gam)
        mc=where(r<rmin,1e-99,nom/dem)
        #mc=nom/dem
        #mc=mc*mask
        return mc

    def L_sat_quad_r_mni01(self,xt_g,yt_g,the,gam,b):
        r=sqrt(xt_g*xt_g+yt_g*yt_g)
        theta=np.arctan2(yt_g,xt_g)
        rmin=0.0
        mask=np.array((r>=rmin))
        nom=r*(1+r)**(-gam)*((1+self.rmax)*(1+rmin))**(gam)*(-2+gam)*(-1+gam)*(b+2*(1+r)**gam+b*np.cos(2*(the-theta)))
        dem=(2*np.pi*(-b*(1+self.rmax)*(1+rmin)**(gam) * (1+self.rmax*(-1+gam))+b*(1+self.rmax)**gam * (1+rmin)* (1+rmin*(-1+gam))+(self.rmax-rmin)*((1+self.rmax)*(1+rmin))**gam * (self.rmax+rmin)*(-2+gam)*(-1+gam)))
        mc=nom/dem
        mc=mc*mask
        #imc=where(r<rmin,1e-99,nom/dem)
        return mc



    def L_sat_quad_r_mn(self,xt_g,yt_g,the,gam,b):
        r=sqrt(xt_g*xt_g+yt_g*yt_g)
        theta=np.arctan2(yt_g,xt_g)
        nom=(-2+gam)*(-1+gam)*r*(1+r)**(-gam)*((1+self.rmax))**gam * (2+b+b*np.cos(2*(the-theta)))
        dem=2*(2+b)*np.pi*(-1+self.rmax*self.rmax -gam*self.rmax*(1+self.rmax)+(1+self.rmax)**gam)
        #nom=(-2+gam)*r*rmin**gam * (r*rmin)**-gam * (self.rmax *rmin)**gam*(2+b+b*np.cos(2*(the-theta)))
        #dem=2*(2+b)*np.pi*(-rmin**gam * self.rmax**2 + self.rmin**2 * self.rmax**gam)
        mc=nom/dem
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
        x_cl,y_cl,sx_cl,sy_cl,x_g,y_g,sx_g,sy_g,fcl,fev,the,c,the2,k,gam,rc,rt=\
        cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12],cube[13],cube[14],cube[15],cube[16]
        mc=(self.L_pm_MW(x_cl,y_cl,sx_cl,sy_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)*fev*fcl*\
        self.L_sat_king(self.x_ps,self.y_ps,rc,rt,self.rmin)+(1-fev)*fcl*\
        #self.King+(1-fev)*fcl*\
        #self.Plum+(1-fev)*fcl*\
        #where(self.dist>=(self.factor*self.TR),\
        self.L_sat_quad_r_mni0(self.x_ps,self.y_ps,the2,gam,k,rt)*\
        self.L_pm_GC(x_cl,y_cl,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff))\
        +self.L_sat_grad(self.x_ps,self.y_ps,the,1,c)*\
        (1-fcl)*self.L_pm_MW(x_g,y_g,sx_g,sy_g,self.x_pm,self.y_pm,self.cv_pmraer,self.cv_pmdecer,self.cv_coeff)
        mc=np.where(mc>0,mc,1e-99)
        mc=np.log(mc).sum()
        return mc



    def loglike_mem(self,x_ps,y_ps,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff,w_par,sample,dist,prcl,prmw,cr,tr):
        '''
        Calculates the membership probability for an individual star
        '''
        #gcct=self.L_sat_quad_r_mn(x_ps,y_ps,sample[:,12],sample[:,14],sample[:,13])
        gcct=self.L_sat_quad_r_mni0(x_ps,y_ps,sample[:,12],sample[:,14],sample[:,13],tr)
        #gcct=self.L_sat_quad_r(x_ps,y_ps,sample[:,12],sample[:,14],sample[:,13])
        #gcsp=where(x_psself.L_sat_king(x_ps,y_ps,sample[:,14],sample[:,15])
        gcsp=self.L_sat_king(x_ps,y_ps,sample[:,15],sample[:,16],self.rmin)
        #gcsp=where(dist<self.tr,self.L_sat_spat_PL(x_ps,y_ps,self.cr,0,self.rmax),0)
        gcpm=self.L_pm_MW(sample[:,0],sample[:,1],sample[:,2],sample[:,3]\
        ,x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
        mwpm=self.L_pm_MW(sample[:,4],sample[:,5],sample[:,6]\
        ,sample[:,7],x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
        mwsp=self.L_sat_grad(x_ps,y_ps,sample[:,10],1,sample[:,11])
        #tspm=self.L_pm_GC(sample[:,0],sample[:,1],\
        #x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
        tspm=self.L_pm_GC(sample[:,0],sample[:,1],\
        x_pm,y_pm,cv_pmraer,cv_pmdecer,cv_coeff)
        gccmd=prcl
        mwcmd=prmw
        #tsccmd=self.L_cmd_cl(w_par,mag,colerr,sample[:,19])
        fcl=sample[:,8]
        fev=sample[:,9]
        #mc_cl=(gccmd*((fcl*fev)*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct))/\
        #(((gccmd*(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp*mwcmd)))
        #mc_co=(gccmd*fcl*fev*gcsp*gcpm)/\
        #(((gccmd*(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp*mwcmd))
        #mc_ts=(gccmd*fcl*(1-fev)*tspm*gcct)/\
        #(((gccmd*(fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp*mwcmd))
        mc_cl=(((fcl*fev)*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct))/\
        ((((fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp)))
        mc_co=(fcl*fev*gcsp*gcpm)/\
        ((((fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp))
        mc_ts=(gccmd*fcl*(1-fev)*tspm*gcct)/\
        ((((fcl*fev*gcsp*gcpm+(fcl*(1-fev)*tspm*gcct)))+(1-fcl*fev-fcl*(1-fev))*mwpm*mwsp))
        return np.nanmean(mc_cl),np.nanstd(mc_cl),np.nanmean(mc_co),np.nanstd(mc_co),np.nanmean(mc_ts),np.nanstd(mc_ts)


    def Membership_after_PyNM(self,sample_size,gnom=True):
        '''
        Run this after PyMultiNest to calculate membership of all stars.
        '''
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            nproc = MPI.COMM_WORLD.Get_size()

        except ImportError:
            rank = 0
            nproc = 1
        if rank==0:
            try:
                f_in=fits.open("../{0}_bays_ready.fits".format(self.cluster))
                f_data=Table(f_in[1].data)
                f_data=f_data[f_data['p_cmdC']>f_data[self.pmcsel]]
                f_data=f_data[f_data['dist']<=self.rmax]
                f_data=f_data[f_data['dist']>=self.rmin]
                x_ps=f_data['ra_g']
                y_ps=f_data['dec_g']
                if gnom==True:
                    x_pm=f_data['pmra_g_SRM']
                    y_pm=f_data['pmdec_g_SRM']
                else:
                    x_pm=f_data['pmra_g']
                    y_pm=f_data['pmdec_g']
                cv_pmraer=f_data['pmra_g_err']
                cv_pmdecer=f_data['pmdec_g_err']
                cv_coeff=f_data['pmra_pmdec_g_corr']
                w_par=f_data['w_iso']
                ktr=f_data['kg_tr']
                ctr=f_data['kg_cr']
                dist=f_data['dist']
                #if self.survey=="PS1":
                #    mag=f_data["i_R0"]
                #    colerr=sqrt(f_data['e_gmag']*f_data['e_gmag']+f_data['e_imag']*f_data['e_imag'])
                #elif self.survey=="gaia":
                #    mag=f_data["g_0"]
                #    colerr=sqrt(f_data['bp_err']*f_data['bp_err']+f_data['rp_err']*f_data['rp_err'])
                #else:
                #    print("BAD")
                #self.King=where(f_data['dist']<=self.tr,self.L_sat_king(x_ps,y_ps,self.cr,self.tr),0)
                a = pymultinest.Analyzer(n_params = self.N_params, outputfiles_basename= self.outbase_name)
                RWE=a.get_data()
                tot_sample=RWE[:,2:]
                zvf=zeros((len(f_data),6))
                print("Begin to calculate Membership probability.")
                for j in PB.progressbar(range(len(w_par))):
                    zvf[j,0],zvf[j,1],zvf[j,2],zvf[j,3],zvf[j,4],zvf[j,5]=self.loglike_mem(x_ps[j],y_ps[j],x_pm[j],y_pm[j],\
                    cv_pmraer[j],cv_pmdecer[j],cv_coeff[j],w_par[j],tot_sample,dist[j],self.PCMD_CL[j],self.PCMD_MW[j],ctr[j],ktr[j])
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
            done=os.path.exists("{0}_mem_list_tot_{1}.fits".format(self.cluster,self.outbase_add))
            for proc in range(1,nproc):
                MPI.COMM_WORLD.send(done,dest=proc)
        else:
            print("Running membership on rank one.")
            done = MPI.COMM_WORLD.recv(source=0)
        print("Complete. Moving on.")
    
  

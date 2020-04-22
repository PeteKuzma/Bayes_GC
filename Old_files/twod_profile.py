#-------------------
# 2D profile preparation
#-------------------
# Version 1

''' 
This take any file with positions and angular distances along with the histograms and 
segments to select the stars needed for the radial profile.
'''


#--------
# Importing classes
#--------

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import scipy.ndimage as snd
import time as tm
import glob as glob
import astropy
from astropy.io import ascii
import math as mt
import matplotlib.gridspec as grd
import matplotlib.colorbar as cm
from astropy.modeling import models, fitting
from matplotlib import ticker
import warnings
import numpy.ma as ma
try:
	import cPickle as pck
except:
	import _pickle as pck
import random as rand
from astroML.stats import fit_bivariate_normal
import scipy.spatial as spatial
import astropy.io.fits as fits
from astropy.table import Table,vstack
import numpy.random as rand
from astropy import coordinates
from astroquery.vizier import Vizier
import progressbar as PB
import sys

#-----

class twod_dist_circle:
	def __init__(self,cluster,weight,ang,stat_ang,plot_ang,pmra,pmdec,bins=300,oang=180,wos=False,smooth=(3,4,5,6)):
		'''
		intiates the code with the equation for the background, cluster RA and dec and
		main sequence box.
		'''
		if os.getcwd()=="/exports/eddie/scratch/pkuzma/Gaia/":
			print("Loading 2D profile routines. In the correct directory.")
		else:
			print("Loading 2D profile routines. Not in correct directory.")
			print("Changing to the correct directory.")
			os.chdir("/exports/eddie/scratch/pkuzma/Gaia/")
		center = coordinates.SkyCoord.from_name(cluster)
		Vizier.ROW_LIMIT=1000000000
		self.smoothing=smooth
		self.ra=center.ra.value
		self.dec=center.dec.value
		self.GC=coordinates.SkyCoord('17h45m40.04s -29d00m28.1s')
		#self.gpmx=pmra*np.cos(racl)-f_data['pmdec'][i]*np.sin(de)*np.cos(racl)
		#self.gpmy=pmdec*np.sin(decl)*np.sin(racl)+f_data['pmdec'][i]*\
		#(np.cos(de)*np.cos(decl)+np.sin(de)*np.sin(decl)*np.cos(racl))
		self.GC_ra=np.rad2deg(np.cos(np.deg2rad(self.GC.ra.value))*np.sin(np.deg2rad(self.ra)))
		self.GC_dec=np.rad2deg(np.sin(np.deg2rad(self.GC.dec.value))*np.cos(np.deg2rad(self.dec))-\
		np.cos(np.deg2rad(self.GC.dec.value))*np.sin(np.deg2rad(self.dec))*np.cos(np.deg2rad(self.ra)))
		self.cluster=cluster
		self.bns=bins
		self.weight=weight
		self.ang=ang
		self.plot_ang=plot_ang
		self.wos=wos
		self.stat_ang=stat_ang
		self.oang=oang
		self.weight_str=str(weight).replace(".","_")
		#mc_in=fits.open(glob.glob("{0}/gnom*".format(self.cluster))[0])
		#self.m_in=Table(mc_in[1].data)
		if wos==False:
			self.file_in=fits.open("{0}/{0}_mem_list_tot.fits".format(self.cluster,self.weight_str))
			full_in=fits.open("{0}/{0}_F_bays_ready_FULL.fits".format(self.cluster))
			self.f_FULL=Table(self.file_in[1].data)
			self.f_FULL['col_sel']=0
			self.full_in=Table(full_in[1].data)
			self.f_In=self.f_FULL[(self.f_FULL['mem_x'])>=self.weight]
			self.b_iN=self.f_FULL[(self.f_FULL['mem_x'])<self.weight]
			self.b_In=vstack((self.f_FULL,self.full_in))
		elif wos==True:
			print("Not loading in stellar catalogs.")
		else:
			raise ValueError('Which file? "wos" must be either True or False.')	


	def creation_2dmasks(self,x_lim=3,y_lim=3,ra1=0,dec1=0,redo=False):
		if os.path.isfile("{0}/{0}_{1}_2d_mask.pickle".format(self.cluster,self.oang))==False or redo==True:
			print("Performing MC simulation...")
			print(x_lim,y_lim,-x_lim)
			x=np.deg2rad(rand.uniform(-x_lim,x_lim,2000000))
			y=np.deg2rad(rand.uniform(-y_lim,y_lim,2000000))
			ra=np.deg2rad(ra1)
			dec=np.deg2rad(dec1)
			ang_dist=np.cos(np.pi/2-y)*np.cos(np.pi/2)+np.sin(np.pi/2-y)*np.sin(np.pi/2)*np.cos(x)
			ang_dist=np.rad2deg(np.arccos(ang_dist))*60.
			print("Simulation complete.")
			m_in=Table([np.rad2deg(x),np.rad2deg(y),ang_dist],names=("ra","dec","ang"))
			m_in=m_in[m_in['ang']<=self.oang]
			print ("Creating plotting mask.\n")
			m_in1=m_in[m_in['ang']>self.plot_ang]
			mask,self.X,self.Y=np.histogram2d(m_in1['ra'],m_in1['dec'],bins=((self.bns,self.bns)))
			b100=ma.masked_less(mask,1)
			self.mask=np.clip(b100,0,1)			
			print("Creating statistical mask.\n")
			m_in1=m_in[m_in['ang']>self.stat_ang]
			mask,self.X,self.Y=np.histogram2d(m_in1['ra'],m_in1['dec'],bins=((self.bns,self.bns)))
			b100=ma.masked_less(mask,1)
			self.st_mask=np.clip(b100,0,1)	
			print("Creating filling mask.\n")
			m_in1=m_in[m_in['ang']>(self.ang-2)]
			mask,self.X,self.Y=np.histogram2d(m_in1['ra'],m_in1['dec'],bins=((self.bns,self.bns)))
			b100=ma.masked_less(mask,1)
			self.fill_mask=np.clip(b100,0,1)
			fopen=open("{0}/{0}_{1}_2d_mask.pickle".format(self.cluster,self.oang),'wb')	
			if sys.version_info[0]==3:
				pck.dump(self.mask,fopen,2)
				pck.dump(self.st_mask,fopen,2)
				pck.dump(self.fill_mask,fopen,2)
				pck.dump(self.X,fopen,2)
				pck.dump(self.Y,fopen,2)
			if sys.version_info[0]==2:
				pck.dump(self.mask,fopen)
				pck.dump(self.st_mask,fopen)
				pck.dump(self.fill_mask,fopen)
				pck.dump(self.X,fopen)
				pck.dump(self.Y,fopen)
			fopen.close()
		if os.path.isfile("{0}/{0}_{1}_2d_mask.pickle".format(self.cluster,self.oang))==True:
			print("Mask exists. Loading.")
			fopen=open("{0}/{0}_{1}_2d_mask.pickle".format(self.cluster,self.oang),'rb')	
			if sys.version_info[0]==3:
				self.mask=pck.load(fopen)
				self.st_mask=pck.load(fopen)
				self.fill_mask=pck.load(fopen)
				self.X=pck.load(fopen)
				self.Y=pck.load(fopen)
			if sys.version_info[0]==2:
				self.mask=pck.load(fopen,encoding='bytes')
				self.st_mask=pck.load(fopen,encoding='bytes')
				self.fill_mask=pck.load(fopen,encoding='bytes')
				self.X=pck.load(fopen,encoding='bytes')
				self.Y=pck.load(fopen,encoding='bytes')
			fopen.close()

	def histogram2d_circle(self,create_plots=False):		
		if os.path.isfile("{0}/{0}_{1}_{2}_2d_dist_plots.pickle".format(self.cluster,self.oang,self.weight))==False or create_plots==True:
			print("Creating and pickling plots.")
			f_in=self.f_In[(self.f_In['dist']*60)>=self.ang]
			b_in=self.b_In[(self.b_In['dist']*60)>=self.ang]
			clhist,self.clX,self.clY=np.histogram2d(f_in['ra_g'],f_in['dec_g'],bins=((self.X,self.Y)))
			clhist_fm=clhist*self.fill_mask
			clhist_sm=clhist*self.st_mask
			clhist_filled=clhist_fm.filled(clhist_sm.mean())
			Sx_mid=np.zeros(len(self.clX)-1)
			Sy_mid=np.zeros(len(self.clX)-1)
			p_init = models.Polynomial2D(degree=1)
			fit_p = fitting.LevMarLSQFitter()
			for g in range(len(self.clX)-1):
				Sx_mid[g]=(self.clX[g]+self.clX[g+1])/2.
				Sy_mid[g]=(self.clY[g]+self.clY[g+1])/2.
			xv,yv=np.meshgrid(Sx_mid,Sy_mid)
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				p=fit_p(p_init, xv, yv, clhist_sm)
			self.poly=p(xv,yv)		
			self.dens_bank=clhist_fm-self.poly
			blhist,self.blX,self.blY=np.histogram2d(b_in['ra_g'],b_in['dec_g'],bins=((self.X,self.Y)))
			blhist_fm=blhist*self.fill_mask
			blhist_sm=blhist*self.st_mask
			blhist_filled=blhist_fm.filled(blhist_sm.mean())
			self.smooth_nb=dict()
			self.smooth_nbst=dict()
			self.smooth_dmap=dict()
			self.smooth_smap=dict()
			self.smooth_map=dict()
			self.smooth_map_nb=dict()
			smooth=self.smoothing
			for i in range(len(smooth)):	
				self.smooth_nb[i]=snd.filters.gaussian_filter(clhist_filled,smooth[i])*self.fill_mask			
				self.smooth_dmap[i]=snd.filters.gaussian_filter(self.dens_bank,smooth[i])*self.fill_mask	
				self.smooth_smap[i]=self.smooth_dmap[i]*self.st_mask
				self.smooth_map[i]=(self.smooth_dmap[i]-ma.mean(self.smooth_smap[i]))/(ma.std(self.smooth_smap[i]))*self.mask
				self.smooth_nbst[i]=self.smooth_nb[i]*self.st_mask
				self.smooth_map_nb[i]=(self.smooth_nb[i]-ma.mean(self.smooth_nbst[i]))/(ma.std(self.smooth_nbst[i]))*self.mask
			fopen=open("{0}/{0}_{1}_{2}_2d_dist_plots.pickle".format(self.cluster,self.oang,self.weight),'wb')	
			if sys.version_info[0]==3:
				pck.dump(self.smooth_map,fopen,2)
				pck.dump(self.smooth_map_nb,fopen,2)
				pck.dump(self.poly,fopen,2)
				pck.dump(self.X,fopen,2)
				pck.dump(self.Y,fopen,2)
			if sys.version_info[0]==2:
				pck.dump(self.smooth_map,fopen)
				pck.dump(self.smooth_map_nb,fopen)
				pck.dump(self.poly,fopen)
				pck.dump(self.smooth,fopen)
				pck.dump(self.X,fopen)
				pck.dump(self.Y,fopen)
			fopen.close()
		if os.path.isfile("{0}/{0}_{1}_{2}_2d_dist_plots.pickle".format(self.cluster,self.oang,self.weight))==True and create_plots==False:
			print("Loading in pickled plots.")
			fopen=open("{0}/{0}_{1}_{2}_2d_dist_plots.pickle".format(self.cluster,self.oang,self.weight),'rb')	
			smooth=self.smoothing
			if sys.version_info[0]==3:
				self.smooth_map=pck.load(fopen)
				self.smooth_map_nb=pck.load(fopen)
				self.poly=pck.load(fopen)
				self.X=pck.load(fopen)
				self.Y=pck.load(fopen)
			if sys.version_info[0]==2:
				self.smooth_map=pck.load(fopen,encoding='bytes')
				self.smooth_map_nb=pck.load(fopen,encoding='bytes')
				self.poly=pck.load(fopen,encoding='bytes')
				self.X=pck.load(fopen,encoding='bytes')
				self.Y=pck.load(fopen,encoding='bytes')
			fopen.close()
		

	def segment_define_gnom(self,sLvls=(2,3),seg_type="+",create_seg=True,create_zeta=True,sample_number=1000):
		opt=self.b_In[(self.b_In['dist']*60)>=self.ang]
		Dx=np.digitize(opt['ra_g'],self.X)
		Dy=np.digitize(opt['dec_g'],self.Y)
		opt['dig_x']=Dx-1
		opt['dig_y']=Dy-1
		if create_seg==True:
			self.segmentation=dict()
			self.segment=dict()
			if seg_type=="+":
				structuring_element= [[0,1,0],[1,1,1],[0,1,0]]
			if seg_type=="o":
				structuring_element = [[1,1,1],[1,1,1],[1,1,1]]
			for j in PB.progressbar(range(len(self.smoothing))):
				self.segmentation[j]=dict()
				self.segment[j]=np.zeros(2)
				for k in range(len(sLvls)):
					SMT=ma.masked_equal(self.smooth_map[j]>sLvls[k],0)
					self.segmentation[j][k],self.segment[j][k] = snd.label(SMT, structuring_element)
			fopen=open("{0}/{0}_{1}_{2}_2d_dist_plots_seg.pickle".format(self.cluster,self.oang,self.weight),'wb')	
			
			if sys.version_info[0]==3:
				pck.dump(self.segmentation,fopen,2)
				pck.dump(self.X,fopen,2)
				pck.dump(self.Y,fopen,2)	
				pck.dump(self.segment,fopen,2)		
			if sys.version_info[0]==2:
				pck.dump(self.segmentation,fopen)
				pck.dump(self.X,fopen)
				pck.dump(self.Y,fopen)
				pck.dump(self.cluster,fopen)	
		if create_seg==False:			
			fopen=open("{0}/{0}_{1}_{2}_2d_dist_plots_seg.pickle".format(self.cluster,self.oang,self.weight),'wb')	
			if sys.version_info[0]==3:
				self.segmentation=pck.load(fopen)
				self.X=pck.load(fopen)
				self.Y=pck.load(fopen)
				self.segment=pck.load(fopen)
			if sys.version_info[0]==2:
				self.segmentation=pck.load(fopen,encoding='bytes')
				self.X=pck.load(fopen,encoding='bytes')
				self.Y=pck.load(fopen,encoding='bytes')
				self.segment=pck.load(fopen,encoding='bytes')
		self.stat_dic=dict()
		self.zeta_value=dict()
		self.coord=dict()
		self.total_count=dict()	
		self.mean_count=dict()
		self.std_count=dict()
		for j in range(len(self.smoothing)):
			self.stat_dic[j]=dict()
			self.coord[j]=dict()
			self.total_count[j]=dict()
			for k in range(len(sLvls)):			
				self.coord[j][k]= snd.center_of_mass(SMT,self.segmentation[j][k],range(1,int(self.segment[j][k])+1))
				self.zeta_value[j]=np.zeros((len(sLvls),int(self.segment[j][k])))
				self.mean_count[j]=np.zeros((len(sLvls),int(self.segment[j][k])))
				self.std_count[j]=np.zeros((len(sLvls),int(self.segment[j][k])))
				self.stat_dic[j][k]=dict()
				self.total_count[j][k]=dict()
				cz=np.array(self.coord[j][k])
				bins_num=range(0,self.bns+1,1)
				pol_dec=np.poly1d(np.polyfit(bins_num,self.X,1))
				pol_ra=np.poly1d(np.polyfit(bins_num,self.Y,1))
				com=np.zeros((int(self.segment[j][k]),2))
				for i in range(int(self.segment[j][k])):
					com[i,1]=pol_dec(cz[i,0])
					com[i,0]=pol_ra(cz[i,1])
				sb=self.segmentation[j][k]*self.fill_mask
				SLIST=sb[opt['dig_x'],opt['dig_y']]
				opt['{0}_{1}'.format(sLvls[k],self.smoothing[j])]=(SLIST).data
				self.opt=opt
				self.control=opt[(opt['dist']*60)>=self.stat_ang]
		self.opt.write("{0}/{0}_with_sample.fits".format(self.cluster),format="fits",overwrite=True)


	def segment_stat_gnom(self,stdLvls=(1,2,3),smlvls=(5,6),sample_number=1000):
		ff=fits.open("{0}/{0}_with_sample.fits".format(self.cluster))
		self.opt=Table(ff[1].data)
		self.stat_dic=dict()
		self.zeta_value=dict()
		self.coord=dict()
		self.total_count=dict()	
		self.mean_count=dict()
		self.std_count=dict()
		for i in stdLvls:
			for j in smlvls:
				control=self.opt[self.opt["{0}_{1}".format(i,j)]==0]
				for k in range(int(max(self.opt["{0}_{1}".format(i,j)]))):
					l=k+1
					stars_in_seg=self.opt[self.opt["{0}_{1}".format(i,j)]==l]
					stars=stars_in_seg[stars_in_seg["mem_x"]>=self.weight]
					count_in=len(stars)
					print("Start sample: \n{0} - smoothing, {1} - sigma level, {2} - segment."\
					.format(j,i,l))
					total_count=np.zeros(sample_number)
					for l in PB.progressbar(range(sample_number)):
						sample=rand.uniform(0,len(control),size=len(stars_in_seg))
						g=sample.astype(int)
						SAMPLE=control[g]
						total_count[l]=len(SAMPLE[SAMPLE['mem_x']>=self.weight])
					mean_count=np.mean(total_count)
					std_count=np.std(total_count)
					print(mean_count,std_count,count_in)
					zeta=(count_in-mean_count)/(std_count)
					print("Zeta: {0}".format(zeta))
	
	



'''
		print("Perform sampling.")
		for i in range(int(self.segment[j][k])):
			self.stat_dic[j][k][i]=dict()
			self.total_count[j][k][i]=np.zeros(sample_number)
			w=i
			seg_length=len(opt[opt['{0}_{1}'.format(sLvls[k],self.smoothing[j])]==w])
			num_stars=len(opt[(opt['{0}_{1}'.format(sLvls[k],self.smoothing[j])]==w) \
			& (opt['mem_x']>=self.weight)])
			if seg_length==0 or i==0:
				print("Segment has no stars or is the background")
				self.zeta_value[j][k,i]=0
			else:
				print("Start sample: \n{0} - smoothing, {1} - sigma level, {2} - segment."\
				.format(self.smoothing[j],sLvls[k],i))
				for l in PB.progressbar(range(sample_number)):
					control=self.control[self.control['{0}_{1}'.format(sLvls[k],self.smoothing[j])]!=w]
					sample=rand.uniform(0,len(control)\
					,size=seg_length)
					g=sample.astype(int)
					SAMPLE=control[g]
#					self.stat_dic[j][k][i][l]=SAMPLE
					self.total_count[j][k][i][l]=len(SAMPLE[SAMPLE['mem_x']>=self.weight])
				self.mean_count[j][k,i]=np.mean(self.total_count[j][k][i])
				self.std_count[j][k,i]=np.std(self.total_count[j][k][i])
				self.zeta_value[j][k,i]=(self.mean_count[j][k,i] - num_stars)/(self.std_count[j][k,i])
	

# '''
# 	def stat_check_gnom(self, filename,mPickle,cluster_path,no_trails,outfile,mag,weight_cut):
# 		f_in=self.f_In[(self.f_In['dist']*60)>=self.ang]
# 		b_in=self.b_In[(self.b_In['dist']*60)>=self.ang]
# 		for j in PB.progressbar(range(len(self.smooth))):
# 		
# 		
# 		sLvls=np.array((1,1.5,2,3,5,10,15,20))
# 		self.STAT=dict()
# 		for i in range(len(sLvls)):
# 			self.STAT[i]=ma.masked_equal(self.smooth[j]>sLvls[i],0)
# 
# 
# 		stat_dic=dict()
# 		real_dic=dict()
# 		sam_dic=dict()
# 		CT=np.arange(1,len(seg),1)
# 		print "SEG ARRAY: \n{0}".format(CT)
# 		print "LENGTH OF SEG FILE:"
# 		print len(seg)
# 		rangez=range(1,len(seg))
# 		final_stat=np.zeros((3,len(rangez)))
# 		print "Creating samples..."
# 		for n in rangez:
# 			test=seg[n]
# 			ctrl=seg[0]
# 			CT_red=np.delete(CT,n-1)
# 			for i in range(len(CT_red)-1):
# 				print CT_red[i]
# 				if i==0:
# 					print "control defined"
# 					control=np.append(ctrl,seg[CT_red[i]],axis=0)
# 				else:
# 					print "control defined"
# 					control=np.append(control,seg[CT_red[i]],axis=0)
# 			print "Beginning new simulation. \n Testing segment: {0}".format(n)
# 			print "Total stars in segment: {0}".format(len(test))
# 			if len(test)==0:
# #				print "Empty set"
# 				stat_dic[n]=0
# 				real_dic[n]=0
# 			else:
# 				count_in_box=np.zeros(no_trails)
# 				for m in range(no_trails):
# 					control_sample=[ control[i] for i in sorted(random.sample(xrange(len(control)), len(test))) ]
# 					control_sample=np.array(control_sample)
# 					#print m, len(control_sample), len(test)
# 					z_array=np.zeros(len(test))
# 					z1_array=np.zeros(len(test))
# 					tESt=np.zeros(len(test))
# 					cm=np.zeros(len(control_sample))
# 					csam=control_sample[control_sample[:,14]>=weight_cut]
# 					count_in_box[m]=len(csam)
# 				stat_dic[n]=count_in_box
# 				
# 				#print len(test)
# 				cmt=np.zeros(len(test))
# 				for j in range(len(test)):
# 					tstt=test[test[:,14]>=weight_cut]
# 	
# 				real_count=len(tstt)
# 				real_dic[n]=real_count
# 		print "Pickle dumping..."
# 		fout=open("{0}/Final_catalogs/{1}_stat_cal.pickle".format(cluster_path,outfile),"wb")
# 		cPickle.dump(stat_dic,fout)
#		cPickle.dump(real_dic,fout)

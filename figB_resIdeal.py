#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#  2D residual  for  various qcrank simu for QuEra 


import os, sys

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from pprint import pprint
import numpy as np

from matplotlib.ticker import MaxNLocator

from matplotlib.gridspec import GridSpec
from toolbox.PlotterBackbone import PlotterBackbone
class Stump:
    a=1

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def residual2D(self,bigD,md,ax,isRaw=False):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']
        pom=md['postproc']
        
        normFac=pom['ampl_fact']
        #print('nFac=',normFac,lab)

               

        #....... plot data .....
        rdata=bigD['rec_udata'].flatten()
        if isRaw:  rdata/=normFac
        tdata=bigD['true_output'].flatten()
        ax.scatter(tdata,rdata,alpha=0.6,s=4)
        x12=[-1,1]
        ax.plot(x12,x12,ls='--',c='k',lw=0.7)

        
        ax.set_aspect(1.)
        ax.set_ylabel('reco  value')
        ax.set_xlabel(r'true $x_i \cdot y_i$ ')

        
        if isRaw:
            ax.axhline(0, color='k', linestyle='--',lw=1.0)
            ampl=1/normFac
            ax.axhline(ampl, color='m', linestyle='--',lw=1.0)
            txt='Ampl\n%.2f'%ampl
            ax.text(-0.3, 0.75, txt, color='m') 
        
#...!...!..................
    def residual1D(self,bigD,md,ax):
        #....... plot data .....
        rdata=bigD['rec_udata'].flatten()
        tdata=bigD['true_output'].flatten()
        res_data = rdata - tdata
        
        ax.hist(res_data, bins=25, color='salmon', alpha=0.7)
        ax.set_xlim(-0.21,0.21)
        ax.set(xlabel=r'true-reco  $x_i \cdot y_i$',ylabel='samples')
        
        #pprint(md)
        txt='RMSE\n%0.3f'%md['postproc']['res_std']
        ax.annotate(txt, xy=(0.1, 0.5),c='r', xycoords='axes fraction')
   
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    outPath='./'  
    inpPath1="data/post/"
    
    #.... geometry
    expD1,expMD1=read4_data_hdf5(os.path.join(inpPath1,'qc5adr_aer_ideal.h5'))
     
    args=Stump()
    args.prjName='figB_resIdeal'
    args.noXterm=True
    args.verb=1
    args.formatVenue='paper'
    args.outPath=outPath
    args.useHalfShots =None
 
    # ----  just plotting
    plot=Plotter(args)

    ax0,ax1=plot.blank_separate2D(nrow=1,ncol=2, figsize=(8,4),figId=1)
    
    plot.residual2D(expD1,expMD1,ax0)
    plot.residual1D(expD1,expMD1,ax1)

    ax0.text(0.05,0.90,'a)',color='k', transform=ax0.transAxes)
    ax1.text(0.05,0.90,'b)',color='k', transform=ax1.transAxes)
    plot.display_all(png=0)
    print('M:done')

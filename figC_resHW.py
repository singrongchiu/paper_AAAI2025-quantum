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
from figB_resIdeal import Plotter
class Stump:
    a=1

    

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    outPath='./'
  
    inpPath1="data/post/"

    
    #.... geometry
    expD1,expMD1=read4_data_hdf5(os.path.join(inpPath1,'qc5adr_ibm_kingston.h5'))
      
     
    args=Stump()
    args.prjName='figC_resHW'
    args.noXterm=True
    args.verb=1
    args.formatVenue='paper'
    args.outPath=outPath
    args.useHalfShots =None
 
    # ----  just plotting
    plot=Plotter(args)

    ax0,ax1=plot.blank_separate2D(nrow=1,ncol=2, figsize=(8,4),figId=1)
    
    plot.residual2D(expD1,expMD1,ax0)
    plot.residual2D(expD1,expMD1,ax0,isRaw=True)
    plot.residual1D(expD1,expMD1,ax1)

    ax0.text(0.05,0.90,'a)',color='k', transform=ax0.transAxes)
    ax1.text(0.05,0.90,'b)',color='k', transform=ax1.transAxes)

    plot.display_all(png=0)
    print('M:done')

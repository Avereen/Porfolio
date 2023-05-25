# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:51:03 2021

@author: AVEREEN
"""


#%% imports

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndi
from scipy import fftpack, signal # have to add 
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'

#%% function library

def loadScanData(sample,scan_path = "Scans\CSV\PN",state_var = ['p','d']):
    return [np.genfromtxt(scan_path+str(sample)+state+'.csv',dtype=float,delimiter=',') for state in state_var]

def interpolateSplines(array):

    # get array shape
    Dshape=np.shape(array)
    Xl=np.linspace(0,1,Dshape[0])
    Yl=np.linspace(0,1,Dshape[1])

    splines = RectBivariateSpline(Xl,Yl,array)
    return splines

def interpolateSplines2(array):
    # get Image shape
    conversion_mm2in=0.0393701
    parray = polyfitsk(array)*conversion_mm2in #
    Dshape=np.shape(parray)
    Xl=np.linspace(0,1,Dshape[0])
    Yl=np.linspace(0,1,Dshape[1])

    splines = RectBivariateSpline(Xl,Yl,parray)
    return splines


def polyfitsk(array):

    # get array shape
    Dshape=np.shape(array)
    x1=np.linspace(-1,1,Dshape[0])
    x2=np.linspace(-1,1,Dshape[1])
    xm = np.meshgrid(x1,x2)
    xx = np.asarray([xm[0].ravel().T,xm[1].ravel().T]).T
    yy = np.asarray(array).ravel()

    model = sk.pipeline.make_pipeline(sk.preprocessing.PolynomialFeatures(8),sk.linear_model.Ridge(alpha=0,solver='cholesky'))
    model.fit(xx,yy)
    ym = model.predict(xx)
    #plot_surf(np.reshape(ym,Dshape))
    return np.reshape(ym,Dshape)

def getInterpolates(image_one,image_two):
    newImagePrior=interpolateSplines(image_one)
    newImagePost=interpolateSplines(image_two)
    return newImagePrior, newImagePost


def interpolateArray(array,nrows=500,ncols=500):
    # get array shape
    Dshape=np.shape(array)
    Xl=np.linspace(0,1,Dshape[0])
    Yl=np.linspace(0,1,Dshape[1])
    
    # get array of x and y
    splines = RectBivariateSpline(Xl,Yl,array)
    
    # refine x and y
    Xl = np.linspace(0,1,nrows)
    Yl = np.linspace(0,1,ncols)
    fineZ = splines(Xl,Yl) 
    return fineZ



def reinterpolate(image_one,image_two):
    dim=max([max(np.shape(image_one)),max(np.shape(image_two))])
    newImagePrior=interpolateArray(image_one,nrows=dim,ncols=dim)
    newImagePost=interpolateArray(image_two,nrows=dim,ncols=dim)
    return newImagePrior, newImagePost

def trimNans(array):
    flags=[]
    Dshape=np.shape(array)
    for col in range(0,Dshape[1],1):
        if np.isnan(np.sum(array[:,col])):
            flags=np.append(flags,col)
        else:
            pass
        
    del_col=[]
    for flag in flags:
        flag=int(flag)
        sus_col=array[:,flag]
        if np.count_nonzero(np.isnan(sus_col)) > 10:
            del_col=np.append(del_col,flag)
        else:
            inds = np.where(np.isnan(sus_col))
            col_mean = np.nanmean(sus_col)
            array[inds,flag]=col_mean
    
    DCarray=array
    DCarray=np.delete(DCarray,np.asarray(del_col,dtype=int),axis=1) 
    return DCarray

def deltaStep(array, dims):
    shape=np.shape(array)
    delta_row=dims[0]/shape[0]
    delta_col=dims[1]/shape[1]
    return delta_row , delta_col

def surfaceArea(array, dims=[1,1]):
    # parmeters
    shape=np.shape(array)
    delta_row=dims[0]/shape[0]
    delta_col=dims[1]/shape[1]
    rs = shape[0]; cs = shape[1];
    
    # Difference 
    diffzr = np.asarray([np.diff(array[row]) for row in range(0,rs)]) #difference by row
    diffzc = np.asarray([np.diff(array[:,col]) for col in range(0,cs)]).T #difference by column
    
    # Set 1  -> x v
    diffzr1 = np.delete(diffzr, -1, 0)
    diffzc1 = np.delete(diffzc, -1, 1)
    hx1=[[delta_row,0,diffzr1[ix,iy]] for ix,iy in np.ndindex(diffzr1.shape)]
    hy1=[[0,delta_col,diffzc1[ix,iy]] for ix,iy in np.ndindex(diffzc1.shape)]
    cross_set1 = np.cross(hx1,hy1)
    area_set1 = np.sum([np.sqrt(a[0]**2+a[1]**2+a[2]**2) for a in cross_set1])/2
    
    # Set 2 A x <-
    diffzr2 = np.delete(diffzr, 0, 0)
    diffzc2 = np.delete(diffzc, 0, 1)
    hx2=[[-delta_row,0,-diffzr2[-ix-1,-iy-1]] for ix,iy in np.ndindex(diffzr2.shape)]
    hy2=[[0,-delta_col,-diffzc2[-ix-1,-iy-1]] for ix,iy in np.ndindex(diffzc2.shape)]
    cross_set2 = np.cross(hx2,hy2)
    area_set2 = np.sum([np.sqrt(a[0]**2+a[1]**2+a[2]**2) for a in cross_set2])/2
    area=area_set1+area_set2
    return area



def applyOperations(genes,image,dim):
    newImage = image.copy()
    lb,rb,tb,bb=genes; bfac=dim/2;
    xcrop=[ -1*int(crop) for crop in list('0'*round(lb*bfac)+'1'*round(rb*bfac))]
    ycrop=[ -1*int(crop) for crop in list('0'*round(tb*bfac)+'1'*round(bb*bfac))]
    for crop in xcrop:
        newImage=np.delete(newImage,int(crop),axis=0)
    for crop in ycrop:
        newImage=np.delete(newImage,int(crop),axis=1)
    newImage-=np.nanmin(newImage)
    return newImage

def genFromSplines(genes,splines,dim):
    lb,rb,tb,bb=genes
    Xl = np.linspace(lb,1-rb,dim)
    Yl = np.linspace(bb,1-tb,dim)
    newImage = splines(Xl,Yl)
    newImage-=np.nanmin(newImage)
    return newImage


def genFromSplines2(genes,splines,dim,hyp=.85):
    left,bott,thetaf,spanf = genes
    bi,bj = 0.3*left, 0.3*bott
    theta = getTheta(bi,bj,thetaf,hyp)
    span, span45 = getSpan(bi,bj,theta,spanf)
    
    Xl = np.linspace(bi,bi+span45,dim)
    Yl = np.linspace(bj,bj+span45,dim)
    XX, YY = np.meshgrid(Xl,Yl)
    
    XXp = bi + np.hypot(YY-bj,XX-bi)*np.cos(theta+np.arctan2(YY-bj,XX-bi))
    YYp = bj + np.hypot(YY-bj,XX-bi)*np.sin(theta+np.arctan2(YY-bj,XX-bi))
    XYp = np.vstack([XXp.ravel(), YYp.ravel()]).T
    
    newImage = splines(XYp[:,0],XYp[:,1],grid=False).reshape(dim,dim)
    newImage -= np.nanmin(newImage)
    return newImage.T

def getTheta(bi,bj,thetaf,hyp=0.85):
    bMag, bRad = np.hypot(bi,bj), np.arctan2(bi,bj)
    spant = (np.hypot(hyp,hyp)-bMag*np.cos(bRad-45))
    spant45 = spant*np.cos(np.deg2rad(45))
    cons=(np.arcsin(bj/spant45),
          np.arcsin(-bi/spant45),
          (np.deg2rad(-45)+np.arccos(-bi/spant)),
          (np.deg2rad(-45)+np.arcsin(-bj/spant)),
          (np.deg2rad(-45)+np.arccos((1-bj)/spant)),
          (np.deg2rad(-45)+np.arcsin((1-bi)/spant)))
    thetaMin = max(np.radians(-3),np.nanmax([con for con in cons if con < 0]))
    thetaMax = min(np.radians(3),np.nanmin([con for con in cons if con > 0]))
    return thetaf*(thetaMax - thetaMin)+thetaMin

def thetaSpan(bi,bj,hyp=0.85):
    bMag, bRad = np.hypot(bi,bj), np.arctan2(bi,bj)
    spant = (np.hypot(hyp,hyp)-bMag*np.cos(bRad-45))
    spant45 = spant*np.cos(np.deg2rad(45))
    cons=(np.arcsin(bj/spant45),
          np.arcsin(-bi/spant45),
          (np.deg2rad(-45)+np.arccos(-bi/spant)),
          (np.deg2rad(-45)+np.arcsin(-bj/spant)),
          (np.deg2rad(-45)+np.arccos((1-bj)/spant)),
          (np.deg2rad(-45)+np.arcsin((1-bi)/spant)))
    thetaMin = max(np.radians(-3),np.nanmax([con for con in cons if con < 0]))
    thetaMax = min(np.radians(3),np.nanmin([con for con in cons if con > 0]))
    return thetaMin, thetaMax

def getSpan(bi,bj,theta,spanf):
    cons = ((1-bi)/np.cos(theta+np.deg2rad(45)),
            (1-bj)/np.sin(theta+np.deg2rad(45)),
            (1-bi)/(np.cos(theta)*np.cos(np.deg2rad(45))),
            (1-bj)/(np.cos(theta)*np.cos(np.deg2rad(45))),
            (-bj)/(np.cos(theta)*np.sin(np.deg2rad(45))),
            (-bi)/(np.cos(theta+np.deg2rad(90))*np.sin(np.deg2rad(45))))
    spanMax = np.nanmin([con for con in cons if con > 0])
    spanMin = spanMax*.9
    span = spanf*(spanMax - spanMin)+spanMin
    return span, span*np.cos(np.deg2rad(45))

def lossPercent(chromosome):
    genes = binaryDecode(chromosome)
    return [((1-(gene[0]+gene[1])/2)*(1-(gene[2]+gene[3])/2)) for gene in genes]

def binaryDecode(chromosome,alleleCount=8):
    '''
    Interperates a given binary string as design vectors for fitting each 
    image
    '''
    if not len(chromosome)%alleleCount:
        chromeLen = int(len(chromosome)); alleleLen = int(len(chromosome)/alleleCount); 
        alleles=tuple(chromosome[i:i+alleleLen] for i in range(0, chromeLen, alleleLen))
        geneVector=tuple(int(gene,2)/int('1'*alleleLen,2) for gene in alleles)
        return geneVector[:4],geneVector[4:]
    return


#%% plotting
    

def plot_surf(array):
    # Max Min
    dmax=np.nanmax(array)
    dmin=np.nanmin(array)

    # get array shape
    Dshape=np.shape(array)
    
    # get array of x and y
    x0, x1 = np.meshgrid(np.linspace(0,1,Dshape[1]),np.linspace(0,1,Dshape[0]))
    
    # Open figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  
    
    # Plot surface
    surf = ax.plot_surface(x0, x1, array, cmap=cm.viridis,linewidth=0, antialiased=1 ,vmax=dmax,vmin=dmin)
    
    # Customize the z axis.
    ax.set_zlim(dmin-0.1*dmin, dmax+0.1*dmax)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig

def plot_curf(array):
    # Max Min
    dmax=np.nanmax(array)
    dmin=np.nanmin(array)

    # get array shape
    Dshape=np.shape(array)
    
    # get array of x and y
    x0, x1 = np.meshgrid(np.linspace(0,1,Dshape[1]),np.linspace(0,1,Dshape[0]))
    
    # Open figure
    fig, ax = plt.subplots()  
    # Plot surface
    curf = ax.contourf(x0, x1, array, levels=900, cmap=cm.gray,vmax=dmax,vmin=dmin)
    
    # Customize the z axis.
    # ax.set_zlim(dmin-0.1*dmin, dmax+0.1*dmax)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(curf, shrink=0.5, aspect=5)
    return fig

def plot_2curf(prior,post,key,color_map=cm.gray):
    # Max Min
    dmax=np.nanmax([np.nanmax(prior),np.nanmax(post)])
    dmin=np.nanmin([np.nanmin(prior),np.nanmin(post)])
    
    difference = np.abs(post-prior)
    rmsediff = np.sqrt(np.abs(np.average(post**2-prior**2)))

    # get array shape
    PRshape=np.shape(prior)
    POshape=np.shape(post)
    
    # get array of x and y
    PRx0, PRx1 = np.meshgrid(np.linspace(0,1,PRshape[1]),np.linspace(0,1,PRshape[0]))
    POx0, POx1 = np.meshgrid(np.linspace(0,1,POshape[1]),np.linspace(0,1,POshape[0]))
    
    # Open figure
    fig, (ax1, ax2, ax3, cax) = plt.subplots(1,4,figsize=(12,4),gridspec_kw={"width_ratios":[1,1,1,0.05]}) 
    plt.suptitle(key)
    
    # Plot surface
    curf_prior = ax1.contourf(PRx0, PRx1, prior, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
    ax1.set_title('prior')
    curf_post = ax2.contourf(POx0, POx1, post, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
    ax2.set_title('deformed')
    #curf_diff = ax3.contourf(POx0, POx1, (post-prior)**3, levels=900, cmap=cm.gray,vmax=np.nanmax((post-prior)**3),vmin=np.nanmin((post-prior)**3))
    curf_diff = ax3.contourf(POx0, POx1, difference, levels=900, cmap=color_map,vmax=dmax,vmin=dmin) #-.5*np.average(post-prior)
    ax3.set_title('difference')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(curf_prior, cax=cax, shrink=0.5, aspect=10)
    #plt.tight_layout()
    return fig, ax1, ax2, ax3, rmsediff

def plot_2curf2(prior,post,key,color_map=cm.gray):
    # Max Min
    dmax=np.nanmax([np.nanmax(prior),np.nanmax(post)])*0.9
    dmin=np.nanmin([np.nanmin(prior),np.nanmin(post)])*1.1
    
    difference = np.abs(post-prior)
    dmax_diff=np.nanmax(difference)*0.9
    dmin_diff=np.nanmin(difference)*1.1
    rmsediff = np.sqrt(np.abs(np.average(post**2-prior**2)))

    # get array shape
    PRshape=np.shape(prior)
    POshape=np.shape(post)
    
    # get array of x and y
    PRx0, PRx1 = np.meshgrid(np.linspace(0,1,PRshape[1]),np.linspace(0,1,PRshape[0]))
    POx0, POx1 = np.meshgrid(np.linspace(0,1,POshape[1]),np.linspace(0,1,POshape[0]))
    
    # Open figure
    fig, (ax1, ax2, cax1, ax3, cax2) = plt.subplots(1,5,figsize=(15,4),gridspec_kw={"width_ratios":[1,1,0.05,1,0.05]}) 
    #plt.suptitle(key)
    
    # Plot surface
    curf_prior = ax1.contourf(PRx0, PRx1, prior, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
    ax1.set_title('prior')
    curf_post = ax2.contourf(POx0, POx1, post, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
    ax2.set_title('deformed')
    #curf_diff = ax3.contourf(POx0, POx1, (post-prior)**3, levels=900, cmap=cm.gray,vmax=np.nanmax((post-prior)**3),vmin=np.nanmin((post-prior)**3))
    curf_diff = ax3.contourf(POx0, POx1, difference, levels=900, cmap=color_map,vmax=dmax_diff,vmin=dmin_diff) #-.5*np.average(post-prior)
    ax3.set_title('difference')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(curf_prior, cax=cax1, shrink=0.5, aspect=10)
    fig.colorbar(curf_diff, cax=cax2, shrink=0.5, aspect=10)
    plt.tight_layout()
    return fig, ax1, ax2, ax3, rmsediff


def plot_2curfall(ims,color_map=cm.gray):
    labels=[]; priorl=[]; postl=[];
    for key , (pre,post) in ims.items():
        labels.append(key)
        priorl.append(pre)
        postl.append(post) 
    # Max Min
    dmax = float(-100)
    dmin = float(100)
    for prior, post in zip(priorl,postl):
        dmaxt=np.nanmax([np.nanmax(prior),np.nanmax(post)])*0.9
        if dmaxt > dmax:
            dmax = dmaxt
        dmint=np.nanmin([np.nanmin(prior),np.nanmin(post)])*1.1
        if dmint < dmin:
            dmin = dmint
    
    dmax_diff = float(-100)
    dmin_diff = float(100)       
    for prior, post in zip(priorl,postl):
        dmaxt=np.nanmax(np.abs(post-prior))*.9
        if dmaxt > dmax_diff:
            dmax_diff = dmaxt
        dmint=np.nanmin(np.abs(post-prior))*1.1
        if dmint < dmin_diff:
            dmin_diff = dmint
    figlist=[]
    for prior, post in zip(priorl,postl):
        difference = np.abs(post-prior)
    
        # get array shape
        PRshape=np.shape(prior)
        POshape=np.shape(post)
        
        # get array of x and y
        PRx0, PRx1 = np.meshgrid(np.linspace(0,1,PRshape[1]),np.linspace(0,1,PRshape[0]))
        POx0, POx1 = np.meshgrid(np.linspace(0,1,POshape[1]),np.linspace(0,1,POshape[0]))
        
        # Open figure
        fig, (ax1, ax2, cax1, ax3, cax2) = plt.subplots(1,5,figsize=(15,4),gridspec_kw={"width_ratios":[1,1,0.05,1,0.05]}) 
        #plt.suptitle(key)
        
        # Plot surface
        curf_prior = ax1.contourf(PRx0, PRx1, prior, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
        ax1.set_title('prior')
        ax1.axes.xaxis.set_visible(False);ax1.axes.yaxis.set_visible(False);
        curf_post = ax2.contourf(POx0, POx1, post, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
        ax2.axes.xaxis.set_visible(False);ax2.axes.yaxis.set_visible(False);
        ax2.set_title('deformed')
        #curf_diff = ax3.contourf(POx0, POx1, (post-prior)**3, levels=900, cmap=cm.gray,vmax=np.nanmax((post-prior)**3),vmin=np.nanmin((post-prior)**3))
        curf_diff = ax3.contourf(POx0, POx1, difference, levels=900, cmap=color_map,vmax=dmax_diff,vmin=dmin_diff) #-.5*np.average(post-prior)
        ax3.axes.xaxis.set_visible(False);ax3.axes.yaxis.set_visible(False);
        ax3.set_title('difference')
        
        # Add a color bar which maps values to colors.
        fig.colorbar(curf_prior, cax=cax1, shrink=0.5, aspect=10)
        fig.colorbar(curf_diff, cax=cax2, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(key+'.png', dpi=300)
        figlist.append(fig)
    return figlist

def plot_2curfall2(ims,color_map=cm.gray):
    labels=[]; priorl=[]; postl=[];
    for key , (pre,post) in ims.items():
        labels.append(key)
        priorl.append(pre)
        postl.append(post) 
    # Max Min
    dmax = float(-100)
    dmin = float(100)
    for prior, post in zip(priorl,postl):
        dmaxt=np.nanmax([np.nanmax(prior),np.nanmax(post)])*0.8
        if dmaxt > dmax:
            dmax = dmaxt
        dmint=np.nanmin([np.nanmin(prior),np.nanmin(post)])*1.1
        if dmint < dmin:
            dmin = dmint
    
    dmax_diff = float(-100)
    dmin_diff = float(100)  
    diffl = []     
    for prior, post in zip(priorl,postl):
        diff = np.abs(post-prior)
        diffl.append(diff)
        dmaxt=np.nanmax(diff)*.9
        if dmaxt > dmax_diff:
            dmax_diff = dmaxt
        dmint=np.nanmin(diff)*1.1
        if dmint < dmin_diff:
            dmin_diff = dmint
    
    # crop 
    
    dmean, dspan_diff = (dmax + dmin)/2 ,(dmax_diff - dmin_diff)/2
    dmax, dmin = min([dmean + 2*dspan_diff,10]), 0#max([dmean - 2*dspan_diff,0])
    dmin_diff=0
    print(dmax,dmin)
    tspan=(round(dmin,1),round(dmax,1))
    
    images=sorted([ imset for imset in zip(priorl,postl,diffl)],key=lambda x : np.nanmax(x[2]))
    
    fig, axess = plt.subplots(3,5,figsize=(6,4.2),gridspec_kw={"width_ratios":[1,1,1,1,0.05],"height_ratios":[1,1,1]})
    #((ax1, ax2, ax3, ax4, cax1),(ax5, ax6, ax7, ax8, cax2),(ax9, ax10, ax11, ax12, cax3)
    
    axess[0,0].set_title('sample a')
    axess[0,1].set_title('sample b')
    axess[0,2].set_title('sample c')
    axess[0,3].set_title('sample d')
    
    
    for i, (prior, post, diff) in enumerate(images):
        # get array shape
        PRshape=np.shape(prior)
        POshape=np.shape(post)
        
        # get array of x and y
        PRx0, PRx1 = np.meshgrid(np.linspace(0,1,PRshape[1]),np.linspace(0,1,PRshape[0]))
        POx0, POx1 = np.meshgrid(np.linspace(0,1,POshape[1]),np.linspace(0,1,POshape[0]))
        
        # Plot surface
        curf_prior = axess[0,i].contourf(PRx0, PRx1, prior, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
        axess[0,i].axes.xaxis.set_visible(False);axess[0,i].axes.yaxis.set_visible(False);
        
        curf_post = axess[1,i].contourf(POx0, POx1, post, levels=900, cmap=color_map,vmax=dmax,vmin=dmin)
        axess[1,i].axes.xaxis.set_visible(False);axess[1,i].axes.yaxis.set_visible(False);
        #ax2.set_title('deformed')
        #curf_diff = ax3.contourf(POx0, POx1, (post-prior)**3, levels=900, cmap=cm.gray,vmax=np.nanmax((post-prior)**3),vmin=np.nanmin((post-prior)**3))
        curf_diff = axess[2,i].contourf(POx0, POx1, diff, levels=900, cmap=color_map,vmax=dmax_diff,vmin=dmin_diff) #-.5*np.average(post-prior)
        axess[2,i].axes.xaxis.set_visible(False);axess[2,i].axes.yaxis.set_visible(False);
        #ax3.set_title('difference')
    
    # Add a color bar which maps values to colors.
    axess[0,0].set_ylabel('prior')
    axess[1,0].set_ylabel('deformed')
    axess[2,0].set_ylabel('difference')
    sclmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmax=dmax,vmin=dmin),cmap=color_map)
    diffmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmax=dmax_diff,vmin=dmin_diff),cmap=color_map)
    fig.colorbar(sclmap, cax=axess[0,4], shrink=0.5, aspect=10, ticks=list(np.linspace(tspan[0],tspan[1],num=5)))
    fig.colorbar(sclmap, cax=axess[1,4], shrink=0.5, aspect=10, ticks=list(np.linspace(tspan[0],tspan[1],num=5)))
    fig.colorbar(diffmap, cax=axess[2,4], shrink=0.5, aspect=10, ticks=list(np.linspace(0.00,.25,num=5)))
    plt.tight_layout()
    plt.savefig(key+'.png', dpi=300)
        
    return fig

#%% Post Processing 

def f_Callable(func,rangex=(0,1),rangey=(0,1),resolution=400):
    x = np.linspace(rangex[0], rangex[1], resolution)
    y = np.linspace(rangey[0], rangey[1], resolution)
    X, Y = np.meshgrid(x, y)
    return RectBivariateSpline(x,y,func(X,Y))
 
def surfaceArea2(funcdata,rangex=(0,1),rangey=(0,1),resolution=400):
    func=funcdata
    if str(type(funcdata))=="<class 'numpy.ndarray'>":
        func = interpolateSplines(funcdata)
    if str(type(funcdata))=="<class 'function'>":
        func = f_Callable(funcdata)
    if str(type(func))=="<class 'scipy.interpolate.fitpack2.RectBivariateSpline'>":
        x, dx = np.linspace(rangex[0], rangex[1], resolution),(rangex[1]-rangex[0])/resolution
        y, dy = np.linspace(rangey[0], rangey[1], resolution),(rangey[1]-rangey[0])/resolution
        X, Y = np.meshgrid(x, y)
        XY=np.vstack([X.ravel(), Y.ravel()]).T
        dfdx=np.square(func(XY[:,0],XY[:,1],dx=1,grid=False))
        dfdy=np.square(func(XY[:,0],XY[:,1],dy=1,grid=False))
        return sum((np.sqrt(dfdx+dfdy+1))*dx*dy)

def surfaceIntegrationPlane(Z):
    x = np.linspace(0, 1, num=Z.size)
    y = np.linspace(0, 1, num=Z.size)
    X, Y = np.meshgrid(x, y)
    # X, Y = X.ravel(), Y.ravel()
    atotal=0
    for index, _ in np.ndenumerate(X):
        xi,yi = index
        if np.max([xi,yi]) >= len(X.ravel())**.5-1:
            # print('out of range')
            None
        else:
            #print('in range')
            x0, y0, x1, y1 = X[xi,yi], Y[xi,yi], X[xi,yi+1], Y[xi+1,yi]
            dy, dx = abs(y0-y1), abs(x0-x1)
            z00, z10, z01, z11 = Z[xi,yi], Z[xi+1,yi], Z[xi,yi+1], Z[xi+1,yi+1]
            # fore triangle
            foreL10 = [z10-z00,dx,0]
            foreL01 = [z01-z00,0,dy]
            area_fore = abs(np.linalg.norm(np.cross(foreL10,foreL01))/2)
            # aft triangle
            aftL10 = [z10-z11,dx,0]
            aftL01 = [z01-z11,0,dy]
            area_aft = abs(np.linalg.norm(np.cross(aftL10,aftL01))/2)
            # area accumulator
            atotal += area_fore + area_aft
    return atotal


def surfaceArea3(funcdata,rangex=(0,1),rangey=(0,1),resolution=400):
    func=funcdata
    if str(type(funcdata))=="<class 'numpy.ndarray'>":
        func = interpolateSplines2(funcdata)
    if str(type(funcdata))=="<class 'function'>":
        func = f_Callable(funcdata)
    if str(type(func))=="<class 'scipy.interpolate.fitpack2.RectBivariateSpline'>":
        x,dx = np.linspace(rangex[0], rangex[1], resolution),(rangex[1]-rangex[0])/resolution
        y,dy = np.linspace(rangey[0], rangey[1], resolution),(rangey[1]-rangey[0])/resolution
        X, Y = np.meshgrid(x, y)
        XY=np.vstack([X.ravel(), Y.ravel()]).T
        dfdx=np.square(func(XY[:,0],XY[:,1],dx=1,grid=False))
        dfdy=np.square(func(XY[:,0],XY[:,1],dy=1,grid=False))
        return sum((np.sqrt(dfdx+dfdy+1))*dx*dy)

def deNoise(Z):
    # zshape = np.shape(Z)
    # splines=interpolateSplines2(Z)
    # Xl = np.linspace(0,1,zshape[0])
    # Yl = np.linspace(0,1,zshape[1])
    # newImage = splines(Xl,Yl)
    # newImage -= np.nanmin(newImage)
    return ndi.median_filter(Z,size=5,mode='nearest')


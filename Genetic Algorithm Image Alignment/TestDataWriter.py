# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:57:52 2021

@author: AVEREEN
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random as random


plt.close('all')

def plot_2curf(prior,post,key):
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
    curf_prior = ax1.contourf(PRx0, PRx1, prior, levels=900, cmap=cm.gray,vmax=dmax,vmin=dmin)
    ax1.set_title('prior')
    curf_post = ax2.contourf(POx0, POx1, post, levels=900, cmap=cm.gray,vmax=dmax,vmin=dmin)
    ax2.set_title('deformed')
    #curf_diff = ax3.contourf(POx0, POx1, (post-prior)**3, levels=900, cmap=cm.gray,vmax=np.nanmax((post-prior)**3),vmin=np.nanmin((post-prior)**3))
    curf_diff = ax3.contourf(POx0, POx1, difference, levels=900, cmap=cm.gray,vmax=dmax,vmin=dmin) #-.5*np.average(post-prior)
    ax3.set_title('difference')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(curf_prior, cax=cax, shrink=0.5, aspect=10)
    #plt.tight_layout()
    return fig, ax1, ax2, ax3, rmsediff
def f(x, y):
    return np.cos(0.5*(2*np.pi)*x)*np.cos((2*np.pi)*y) +1

def post(x, y,bi=.02,bj=.05,gamma=1.03,theta=0.0174533):
    xs = bi + np.hypot(gamma*y[:],gamma*x[:])*np.cos(theta+np.arctan2(gamma*y[:],gamma*x[:]))
    ys = bj + np.hypot(gamma*y[:],gamma*x[:])*np.sin(theta+np.arctan2(gamma*y[:],gamma*x[:]))
    Z=f(xs, ys)
    Zcir=Z[:]
    for xsi, ysi in np.ndindex(Z.shape):
        if(xs[xsi,ysi]**2+ys[xsi,ysi]**2)<.5 and (xs[xsi,ysi]**2+ys[xsi,ysi]**2)>.45 and np.arctan2(ysi,xsi)>0.261799 and np.arctan2(ysi,xsi)<1.309:
            Zcir[xsi, ysi]=Z[xsi, ysi]+0.3
        else:
            Zcir[xsi, ysi]=Z[xsi, ysi]
    return Zcir

def noise(Z):
    Zn = Z[:]
    for xsi, ysi in np.ndindex(Z.shape):
        Zn[xsi, ysi] += (random.random()-.5)*.1
    return Zn
    

def prior(x, y):
    xs=x[:]
    ys=y[:]
    Z=f(xs, ys)
    Zcir=Z[:]
    for xsi, ysi in np.ndindex(Z.shape):
        if(xs[xsi,ysi]**2+ys[xsi,ysi]**2)<.5 and (xs[xsi,ysi]**2+ys[xsi,ysi]**2)>.45 and np.arctan2(ysi,xsi)>0.261799 and np.arctan2(ysi,xsi)<1.309:
            Zcir[xsi, ysi]=Z[xsi, ysi]+0.2
        else:
            Zcir[xsi, ysi]=Z[xsi, ysi]
    return Zcir

def loadScanData(sample,scan_path = "Scans\cropCSV\PN",state_var = ['p','d']):
    return [np.genfromtxt(scan_path+str(sample)+state+'.csv',dtype=float,delimiter=',') for state in state_var]

def writeTestData(parameters,scan_path = r"Scans\testData2\PN",state_var = ['p','d']):
    samp_id, bi, bj, gamma ,theta = parameters
    x = np.linspace(0, 1, 600)
    y = np.linspace(0, 1, 600)
    X, Y = np.meshgrid(x, y)
    Deform = noise(post(X, Y, bi, bj, gamma ,theta))
    Prior = noise(f(X, Y))
    plot_2curf(Prior,Deform,'Sample id:'+samp_id)
    plt.savefig(scan_path+samp_id+'.png')
    plt.close('all')
    np.savetxt(scan_path+samp_id+state_var[0]+'.csv', Prior, delimiter=',')
    np.savetxt(scan_path+samp_id+state_var[1]+'.csv', Deform, delimiter=',')
    return print(samp_id)

if __name__ == "__main__":
    samp_id=[str(i+1) for i in range(4)]
    bias_i=[0,0.03,0.02,0.02]
    bias_j=[0,0.02,0.01,0.02]
    gamma=[1.03,1.01,1.01,1]
    theta=[0,-0.0349066,0.0174533,0.0349066]
    for parameter in list(zip(samp_id, bias_i, bias_j, gamma ,theta)):
        writeTestData(parameter)
    


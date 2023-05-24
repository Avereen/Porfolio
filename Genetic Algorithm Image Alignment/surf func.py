# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:23:23 2021

@author: AVEREEN
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D 
import scipy.ndimage as ndi
import random

#%% function library
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

def loadScanData(sample,scan_path = "Scans\CSV\PN",state_var = ['p','d']):
    return [np.genfromtxt(scan_path+str(sample)+state+'.csv',dtype=float,delimiter=',') for state in state_var]

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

def interpolateSplines(array):

    # get array shape
    Dshape=np.shape(array)
    Xl=np.linspace(0,1,Dshape[0])
    Yl=np.linspace(0,1,Dshape[1])

    splines = RectBivariateSpline(Xl,Yl,array)
    return splines

def cosWobble(x, y):
    return np.cos(0.5*(2*np.pi)*x)*np.cos((2*np.pi)*y) +1

def noise(Z):
    Zn = Z[:]
    for xsi, ysi in np.ndindex(Z.shape):
        if random.random()>.7:
            Zn[xsi, ysi] += (random.random()-.5)*.1
        else:
            if random.random()>.9:
                Zn[xsi, ysi] += (random.random()-.5)*.5
            else:
                Zn[xsi, ysi] += 0
    return Zn


def noiseWobble(x, y):
    return noise(cosWobble(x, y))


def deNoise(Z):
    return ndi.uniform_filter(Z,mode='nearest')

def conWobble(x, y):
    return deNoise(noise(cosWobble(x, y)))
def cosWobbleN(x, y):
    return deNoise(cosWobble(x, y))

def linearPlane(x, y):
    z=np.ones(x.shape)
    for xsi, ysi in np.ndindex(x.shape):
        z[xsi, ysi]=(x[xsi,ysi]+y[xsi,ysi])/2
    return z

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)

plot_surf(linearPlane(X, Y))
print(surfaceArea(linearPlane(X, Y)))

def f_Callable(func,rangex=(0,1),rangey=(0,1),resolution=400):
    x = np.linspace(rangex[0], rangex[1], resolution)
    y = np.linspace(rangey[0], rangey[1], resolution)
    X, Y = np.meshgrid(x, y)
    return RectBivariateSpline(x,y,func(X,Y))


lin = f_Callable(linearPlane)

def f_surf_area(funcdata,rangex=(0,1),rangey=(0,1),resolution=400):
    func=funcdata
    if str(type(funcdata))=="<class 'numpy.ndarray'>":
        func = interpolateSplines(funcdata)
    if str(type(funcdata))=="<class 'function'>":
        func = f_Callable(funcdata)
    if str(type(func))=="<class 'scipy.interpolate.fitpack2.RectBivariateSpline'>":
        x,dx = np.linspace(rangex[0], rangex[1], resolution),np.double(rangex[1]-rangex[0])/resolution
        y,dy = np.linspace(rangey[0], rangey[1], resolution),np.double(rangey[1]-rangey[0])/resolution
        X, Y = np.meshgrid(x, y)
        XY=np.asarray(np.vstack([X.ravel(), Y.ravel()]).T,dtype=np.double)
        dfdx=np.square(np.asarray(func(XY[:,0],XY[:,1],dx=1,grid=False),dtype=np.double))
        dfdy=np.square(np.asarray(func(XY[:,0],XY[:,1],dy=1,grid=False),dtype=np.double))
        return sum((np.sqrt(dfdx+dfdy+1))*dx*dy)




plot_surf(cosWobble(X,Y))
plot_surf(cosWobbleN(X,Y))
plot_surf(conWobble(X,Y))
print(f_surf_area(np.ones((100,100)),resolution=400))
print((f_surf_area(linearPlane,resolution=100)-np.sqrt(.5**2 +.5**2 + 1))/np.sqrt(.5**2 +.5**2 + 1))
print((f_surf_area(linearPlane,resolution=200)-np.sqrt(.5**2 +.5**2 + 1))/np.sqrt(.5**2 +.5**2 + 1))
print((f_surf_area(linearPlane,resolution=300)-np.sqrt(.5**2 +.5**2 + 1))/np.sqrt(.5**2 +.5**2 + 1))
#print((f_surf_area(noiseWobble,resolution=1000)-f_surf_area(cosWobble,resolution=1000))/f_surf_area(cosWobble,resolution=1000))
print((f_surf_area(conWobble,resolution=400)-f_surf_area(cosWobble,resolution=400))/f_surf_area(cosWobble,resolution=400))
print((f_surf_area(conWobble,resolution=400)-f_surf_area(cosWobbleN,resolution=400))/f_surf_area(cosWobbleN,resolution=400))



sample1=loadScanData(1)
sample1P,sample1D=sample1[0],sample1[1]

plot_surf(sample1P)
plot_surf((deNoise(sample1P)-sample1P)/np.average(sample1P))
plot_surf((deNoise(sample1D)-sample1D)/np.average(sample1D))







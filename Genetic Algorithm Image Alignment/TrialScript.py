# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:47:31 2021

@author: alexv
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')
import random
import imageUtilityFunctions as im
import GAUtilityFunctions as ga 
from matplotlib import cm
from imageDataClass import imageDataClass
from GApopulation import GApopulation
import matplotlib.pyplot as plt
import time as time
plt.close('all')



def getSurfaceProps(population,dimsamp):
    popOrder, fit = population.populationScores()
    dims = ga.areaPercent(popOrder[0])
    dim = sum([dims[0]**.5,dims[1]**.5])/2
    designs = population.generateDesign(popOrder[0])
    sa1 = im.surfaceArea3(designs[0], rangex=(0,dim*dimsamp[0]), rangey=(0,dim*dimsamp[1]),resolution=1000)
    sa2 = im.surfaceArea3(designs[1], rangex=(0,dim*dimsamp[0]), rangey=(0,dim*dimsamp[1]),resolution=1000)
    return sa1, sa2

epochs_set=[1,50,200]
tribe = 0
elites={}
gadict={}
history={}
samples={'Test':['t1','t2','t3','t4']}
data='Test'

if __name__ == "__main__":
    print('Dataset used is the '+data+'set')
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        elites.update({sample:[]})
        gadict.update({key:GApopulation(sample)})
        history.update({key:[]})
    for epochs in epochs_set: 
        for sample in samples[data]:
              for i in range(epochs):
                  key = 's'+sample+'t'+str(tribe)
                  t1=time.time()
                  history[key].append(gadict[key].propagate())
                  t=time.time()-t1
                  print('time: '+str(t))
    
        for sample in samples[data]:
            key = 's'+sample+'t'+str(tribe)
            pop,fit=gadict[key].populationScores()
            elites[sample].append(pop[0])
            image1, image2 = gadict[key].generateDesign(pop[0],downsample=.3)
            im.plot_2curf(image1,image2,sample+' score: '+str(fit[0]))
            if data=='Test':
                plt.savefig(r'output\Sample '+sample+'gen'+str(gadict[key].epoch)+'.png')
            plt.close('all')
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        eps, fits = [], []
        for ep, fitTop in history[key]:
            for fit in fitTop:
                eps.append(ep)
                fits.append(fit)
        fig, axs = plt.subplots(figsize=(7,4))
        plt.scatter(eps,fits)
        plt.title('Sample '+sample+' history')
        axs.set_xlabel('epochs')
        axs.set_ylabel('top performers (loss)')
        if data=='Test':
            plt.savefig(r'output\Sample '+sample+' history.png')
    
    f = open('epochs'+str(sum(epochs_set))+'_set_'+str(data)+".txt", "w")
    f.write(str(elites))
    f.close()
    
    cap = [1.261030390597417,1.0223432773262373,1.16854206196601,1.013121742645992]
    dimsamp = {'1':[1,1],'2':[1,1],'3':[1,1],'4':[1,1],
               't1':[1,1],'t2':[1,1],'t3':[1,1],'t4':[1,1]} 
    propout={}
    
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        surfv = getSurfaceProps(gadict[key],dimsamp[sample])
        delA = abs(surfv[0]-surfv[1])/surfv[0]
        dcoc = (surfv[1]**2)/(surfv[0]**2)-1 
        propout.update({sample:'dcoc: '+str(dcoc)+'      normal area change: '+str(delA)})
        print(sample)
        print('Change in area:'+str(delA) )
        print('Change in cap:'+str(dcoc)+'\n')
    f = open('epochs'+str(sum(epochs_set))+'predicted_props'+".txt", "w")
    f.write(str(propout))
    f.close()
    plt.close('all')
    images={}
    keys=[]
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        keys.append(key)
        pop,fit=gadict[key].populationScores()
        elites[sample].append(pop[0])
        image1,image2=gadict[key].generateDesign(pop[0],downsample=1)
        images.update({key:(image1,image2)})
    fig=im.plot_2curfall2(images,cm.gray)
    fig.savefig('final_output.png',dpi=300)   
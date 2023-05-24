# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:47:31 2021

@author: alexv
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')
import random
import imageUtilityFunctions3 as im
from matplotlib import cm
import GAUtilityFunctions3 as ga 
from imageDataClass3 import imageDataClass
from GApopulation3 import GApopulation
import matplotlib.pyplot as plt
import time as time
import ast
plt.close('all')


# def getSurfaceProps(population):
#     popOrder, fit = population.populationScores()
#     dims = ga.areaPercent(popOrder[0])
#     dim = sum([dims[0]**.5,dims[1]**.5])/2
#     designs = population.generateDesign(popOrder[0])
#     sa1 = im.surfaceArea2(designs[0], rangex=(0,dim), rangey=(0,dim),resolution=400)
#     sa2 = im.surfaceArea2(designs[1], rangex=(0,dim), rangey=(0,dim),resolution=400)
#     return sa1, sa2
def getSurfaceProps(population,dimsamp):
    popOrder, fit = population.populationScores()
    dims = ga.areaPercent(popOrder[0])
    dim = sum([dims[0]**.5,dims[1]**.5])/2
    designs = population.generateDesign(popOrder[0])
    sa1 = im.surfaceArea3(designs[0], rangex=(0,dim*dimsamp[0]), rangey=(0,dim*dimsamp[1]),resolution=1000)
    sa2 = im.surfaceArea3(designs[1], rangex=(0,dim*dimsamp[0]), rangey=(0,dim*dimsamp[1]),resolution=1000)
    return sa1, sa2

epochs_set=[1]#,70,100,100,100,100]#,100,100,100,100,100,100,100,100,100,100]
            #100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
            #100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
            #100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
            #100,100]
tribe = 0
elites = {}
gadict = {}
history = {}
samples = {'Scan':['1','2','3','4'],'Test':['t1','t2','t3','t4']}
data = ['Scan','Test']
data = data[0]  #  0 for real data || 1 for Testdata
res_epoch = 4500
path=r"epochs"+str(res_epoch)+"_set_"+data+".txt"
with open(path) as file:
    stock = file.read()
stock = dict(ast.literal_eval(stock))


if __name__ == "__main__": 
    print('Dataset used is the '+data+'set')
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        elites.update({sample:[]})
        gadict.update({key:GApopulation(sample)})
        gadict[key].epoch = res_epoch
        gadict[key].resumeFromStock(stock[sample])
        gadict[key].epoch = res_epoch
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
            image1,image2=gadict[key].generateDesign(pop[0],downsample=.3)
            im.plot_2curf(image1,image2,sample+' score: '+str(fit[0]))
            if data=='Test':
                plt.savefig(r'trialFuncSet\Sample '+sample+'gen'+str(gadict[key]).epoch+'.png')
            if data=='Scan':
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
            plt.savefig(r'trialFuncSet\Sample '+sample+str(gadict[key].epoch)+' history.png')
        if data=='Scan':
            plt.savefig(r'output\Sample '+sample+str(gadict[key].epoch)+' history.png')
    
    f = open('epochs'+str(gadict[key].epoch)+'_set_'+str(data)+".txt", "w")
    f.write(str(elites))
    f.close()
    
    
    cap = [1.261030390597417,1.0223432773262373,1.16854206196601,1.013121742645992]
    dimsamp = {'1':[1,1],#[.845,.845],
               '2':[1,1],#[.845,.845],
               '3':[1,1],#[.845,.845],
               '4':[1,1],#[.845,.845],
               't1':[1,1],#
               't2':[1,1],
               't3':[1,1],
               't4':[1,1]}
    
    f = open('epochs'+str(sum(epochs_set))+'_set_'+str(data)+".txt", "w")
    f.write(str(elites))
    f.close()
    
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
        
    for sample in samples[data]:
        key = 's'+sample+'t'+str(tribe)
        pop,fit=gadict[key].populationScores()
        elites[sample].append(pop[0])
        image1,image2=gadict[key].generateDesign(pop[0],downsample=1)
        im.plot_2curf2(image1,image2,sample+' score: '+str(fit[0]),cm.viridis)
        if data=='Test':
            plt.savefig(r'trialFuncSet\Sample '+sample+'gen'+str(gadict[key]).epoch+'.png')
        if data=='Scan':
            plt.savefig(r'output\aSample '+sample+'gen'+str(gadict[key].epoch)+'.png')        
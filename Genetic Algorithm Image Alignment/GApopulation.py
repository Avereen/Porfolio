# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:16:36 2021

@author: AVEREEN
"""
import random
import imageUtilityFunctions as im
import GAUtilityFunctions as ga 
from imageDataClass import imageDataClass
import matplotlib.pyplot as plt
import time as time

class GApopulation(imageDataClass):
    
    def __init__(self,sample,cl=64,alc=8,popcount=100,elitism=True,elites=.1,proles=.6,mutation=0.99,areaMin=.5,downsample=0):
        imageDataClass.__init__(self,sample)
        self.cl , self.al, self.alc = cl, int(cl/alc), alc
        self.elitism, self.elites, self.proles = elitism, round(elites*popcount), round(proles*popcount)
        self.mutation, self.mutateCDF = mutation, ga.genRankCDF(self.al)
        self.population = ga.initialPopulation(cl,popcount,areaMin)
        self.popcount = popcount
        self.areaMin = areaMin
        self.extinct = set()
        self.epoch, self.sample = 0 , sample
        self.crc=4
        self.popMax=1
        self.ds = downsample
    
    def propagate(self):
        'get fitness'
        self.epoch += 1
        popOrder, fit = self.populationScores(min(self.ds+.000005*(self.epoch),1))
        print('Fitness Scored')
        fitPut = fit[:10]
        
        'breeding plan'
        self.population={}
        if self.elitism:
            for i in range(self.proles):
                self.extinct.add(popOrder.pop(-1));fit.pop(-1);
            
            CDF = ga.genRankCDF(len(fit))
            
            while len(self.population)<self.popcount-self.proles-self.elites:
                parents=[popOrder[ga.rollFromCDF(random.random(),CDF)] for i in range(2)]
                if parents[0] != parents[1]:
                    children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                    children=[ga.mutation(child,self.mutation,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                    for child in children:
                        if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                            self.population.update({child:1})
                            
            for i in range(self.elites):
                self.population.update({popOrder.pop(0):fit.pop(0)})
            
            while len(self.population)<self.popcount:
                immigrant=''.join(map(str,[str(round(random.random())) for gene in range(self.cl)]))
                if all(area>self.areaMin for area in ga.areaPercent(immigrant)) and not immigrant in self.extinct:
                    self.population.update({immigrant:1})
            
        else:
            CDF = ga.genRankCDF(len(fit))
            while len(self.population)<self.popcount:
                parents=[popOrder[ga.rollFromCDF(random.random(),CDF)] for i in range(2)]
                if parents[0] != parents[1]:
                    children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                    children=[ga.mutation(child,self.mutation,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                    for child in children:
                        if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                            self.population.update({child:1})
    
        for member in popOrder:
            self.extinct.add(member)
        print('generation: '+str(self.epoch)+'   sample: '+str(self.sample))
        print('best member: '+str(self.popMax))
        print('population average: '+str(self.popAverage))
        
        return self.epoch, fitPut 
    
    def tournamentPropagate(self,participants):
        'get fitness'
        self.epoch += 1
        popOrder, fit = self.populationScores(self.ds)

        'breeding plan'
        self.population={}
        for i in range(self.proles):
            self.extinct.add(popOrder.pop(-1));fit.pop(-1);
        
        CDF = ga.genRankCDF(len(fit))
        
        while len(self.population)<self.popcount-self.proles-self.elites:
            parents=[popOrder[ga.rollFromCDF(random.random(),CDF)] for i in range(2)]
            if parents[0] != parents[1]:
                children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                children=[ga.mutation(child,self.mutation,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                for child in children:
                    if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                        self.population.update({child:1})
                        
        for i in range(self.elites):
            self.population.update({popOrder.pop(0):fit.pop(0)})
        
        while len(self.population)<self.popcount:
            parents=[random.choice(participants) for i in range(2)]
            if parents[0] != parents[1]:
                children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                children=[ga.mutation(child,0,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                for child in children:
                    if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                        self.population.update({child:1})
                    
        for member in popOrder:
            self.extinct.add(member)
        print('generation: '+str(self.epoch)+'   sample: '+str(self.sample))
        print('best member: '+str(self.popMax))
        print('population average: '+str(self.popAverage))
              
        return None
    
    def resumeFromStock(self,stockp):
        stock=stockp[:]
        'get fitness'
        self.epoch += 1
        popOrder, fit = self.populationScores(min(self.ds+.0001*(self.epoch),.7))
        print('Fitness Scored')
        fitPut = fit[:10]
        
        'breeding plan'
        self.population={}
        if self.elitism:
            for i in range(self.proles):
                self.extinct.add(popOrder.pop(-1));fit.pop(-1);
            
            CDF = ga.genRankCDF(len(fit))
            while len(stock)>0 and len(self.population)<self.popcount-self.proles-self.elites:
                child=stock.pop()
                self.population.update({child:1})
            while len(self.population)<self.popcount-self.proles-self.elites:
                parents=[popOrder[ga.rollFromCDF(random.random(),CDF)] for i in range(2)]
                if parents[0] != parents[1]:
                    children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                    children=[ga.mutation(child,self.mutation,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                    for child in children:
                        if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                            self.population.update({child:1})
                            
            for i in range(self.elites):
                self.population.update({popOrder.pop(0):fit.pop(0)})
            
            while len(self.population)<self.popcount:
                immigrant=''.join(map(str,[str(round(random.random())) for gene in range(self.cl)]))
                if all(area>self.areaMin for area in ga.areaPercent(immigrant)) and not immigrant in self.extinct:
                    self.population.update({immigrant:1})
            
        else:
            CDF = ga.genRankCDF(len(fit))
            while len(self.population)<self.popcount:
                parents=[popOrder[ga.rollFromCDF(random.random(),CDF)] for i in range(2)]
                if parents[0] != parents[1]:
                    children=ga.crossover(parents[0],parents[1],alleleCount=self.alc,crossCount=round(self.crc))
                    children=[ga.mutation(child,self.mutation,self.alc,round(min(24,max(1/self.crc,1)))) for child in children]
                    for child in children:
                        if all(area>self.areaMin for area in ga.areaPercent(child)) and not child in self.extinct:
                            self.population.update({child:1})
    
        for member in popOrder:
            self.extinct.add(member)
        print('generation: '+str(self.epoch)+'   sample: '+str(self.sample))
        print('best member: '+str(self.popMax))
        print('population average: '+str(self.popAverage))
        
        return self.epoch, fitPut 
        
    def populationScores(self,downsample=0):
        'get fitness'
        for member in ga.dictKeys(self.population):
            if self.population[member] > 0:
                self.population.update({member:ga.lossFunction(self.scoreChromosome(member,downsample))})
        popOrder, fit = ga.sortFitness(self.population)
        self.popAverage=sum(fit)/len(fit)
        if self.popMax==max(fit):
            self.crc*=.99 #increase variance if stagnate
        self.popMax=max(fit)
        
        return popOrder, fit      

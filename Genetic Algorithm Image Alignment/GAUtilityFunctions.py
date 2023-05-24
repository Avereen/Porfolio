# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:29:32 2021

@author: alexv
"""
import random
import numpy as np
import imageUtilityFunctions as im


#%% INITIALIZATION 
def initialPopulation(chromeLen=64,popCount=100,areaMin=.1):
    '''
    Create an population a of strings within an intial region of the 4D state space 
    '''
    population={}
    while (len(population) < popCount):
       chromosome=''.join(map(str,[str(round(random.random())) for gene in range(chromeLen)]))
       if all(area>areaMin for area in areaPercent(chromosome)):
           population.update({chromosome:1})
    print('Population Initialized')
    return population

#%% SCORING
'''
Assorted utils for decoding a and scoring population strings
'''
def sortFitness(dictionary):
    popScores=dictItems(dictionary)
    popfit = np.asarray(sorted(popScores, key = lambda x: -x[1]))
    popOrder, fit = popfit[:,0], popfit[:,1]
    fit = np.array(fit,dtype=np.float32)
    return list(popOrder), list(fit)

def lossFunction(scoreTuple):
    return sum(scoreTuple)

def areaPercent(chromosome):
    genes = binaryDecode(chromosome)
    Area=[]
    for gene in genes:
        left,bott,thetaf,spanf = gene
        bi,bj = 0.3*left, 0.3*bott
        theta = im.getTheta(bi,bj,thetaf)
        span, span45 = im.getSpan(bi,bj,theta,spanf)
        Area.append(span45**2)
    return Area

def aspectRatio(chromosome):
    genes = binaryDecode(chromosome)
    ratios=[]
    for gene in genes:
        ratios.append((1-(gene[0]+gene[1]))/(1-(gene[2]+gene[3])))
        ratios.append((1-(gene[2]+gene[3]))/(1-(gene[0]+gene[1])))
    return ratios

#%% BREEDING    
'''
Assorted utils for generating child strings for the next generation
'''
def crossover(parent1,parent2,alleleCount=8,crossCount=4):
    if not len(parent1)%alleleCount:
        cl = int(len(parent1)); al = int(len(parent1)/alleleCount)
        alleles1=tuple(parent1[i:i+al] for i in range(0, cl, al))
        alleles2=tuple(parent2[i:i+al] for i in range(0, cl, al))
        cross = 0
        if crossCount>0:
            while cross < crossCount:
                cross = 0
                child1,child2=[],[]
                for allele in range(al):
                    if random.random()>.50:
                        split = int(round((al-2)*random.random())+1)
                        child1+=alleles1[allele][:split]+alleles2[allele][split:]
                        child2+=alleles2[allele][:split]+alleles1[allele][split:]
                        cross+=1
                    else:
                        child1+=alleles1[allele]
                        child2+=alleles2[allele]
            return strList(child1),strList(child2)
        else:
            return parent1, parent2

def mutation(child,mutation=.05,alleleCount=8,mutationCount=1):
    if random.random() < mutation:
        al=int(len(child)/alleleCount)
        mutchild = list(child)[:]
        mutations = 0
        while mutationCount > mutations:
            mutations += 1
            allele = int(round((alleleCount-1)*random.random()))*(al)
            gene = rollFromCDF(random.random(),genRankCDF(al))
            while not gene:
                gene=rollFromCDF(random.random(),genRankCDF(al))
            selection = allele+gene
            if selection>len(mutchild):
                selection-=gene+1
                mutchild[selection]=bitFlip(mutchild[selection])
            elif selection<0:
                selection=0
                mutchild[selection]=bitFlip(mutchild[selection])
            else:
                mutchild[selection]=bitFlip(mutchild[selection])
        return strList(mutchild)
    return child


#%% ROULETTE ROLLS
def rollFromCDF(roll,CDF):
    if not CDF or CDF[0] > roll:
        return 0
    for member, cumprob in enumerate(CDF):
        if cumprob > roll:
            return member

def genRankCDF(ranks):
    PDF = [(i+1)/sum([s+1 for s in range(ranks)]) for i in range(ranks)]
    return [sum(PDF[:i+1]) for i in range(ranks)]

def genPropCDF(pop):
    if type(pop)==dict:
        fit = [pop[key] for key in dictKeys(pop)]
    else:
        fit = pop
    PDF = [fit[i]/sum([fit[s] for s in range(len(fit))]) for i in range(len(fit))]
    return [sum(PDF[:i+1]) for i in range(len(fit))]



#%% DATA UTILITY FUNCTIONS

def binaryDecode(chromosome,alleleCount=8,limiter=.32):
    '''
    Interperates a given binary string as design vectors for fitting each 
    image
    '''
    if not len(chromosome)%alleleCount:
        chromeLen = int(len(chromosome)); alleleLen = int(len(chromosome)/alleleCount); 
        alleles=tuple(chromosome[i:i+alleleLen] for i in range(0, chromeLen, alleleLen))
        geneVector=tuple(limiter*int(gene,2)/int('1'*alleleLen,2) for gene in alleles)
        return geneVector[:4],geneVector[4:]
    return

def strList(lis):
    return ''.join(map(str,lis))

def dictKeys(dictionary):
    return [str(item[0]) for item in list(dictionary.items())]

def dictItems(dictionary):
    return [[str(item[0]),item[1]] for item in list(dictionary.items())]

def bitFlip(bitString):
    return str(int(not bool(int(bitString))))


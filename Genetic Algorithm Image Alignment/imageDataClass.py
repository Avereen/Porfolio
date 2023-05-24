# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:36:22 2021

@author: AVEREEN
"""
import numpy as np
import imageUtilityFunctions as im
import GAUtilityFunctions as ga 


class imageDataClass:
    
    def __init__(self,sample):
        self.initialize(sample)
        
    def initialize(self,sample):
        '''
        On initialization reinterpolate the images as n by n where n is the
        maximum length of any axis of the two images 
        '''
        # clean out negatives
        if sample[0]=='t':
            images = im.loadScanData(sample[1],scan_path=r"Scans\testData\PN")
        else:
            images = im.loadScanData(sample,scan_path=r"Scans\CSV\PN")#cropCSV1
        prior, post = images[0].copy(), images[1].copy()
        prior[prior<0], post[post<0] = np.nan, np.nan
        prior, post = im.deNoise(im.trimNans(prior)), im.deNoise(im.trimNans(post))
        
        self.dim=max([max(np.shape(prior)),max(np.shape(post))])
        self.baseImagePrior,self.baseImagePost=im.getInterpolates(prior,post)
        
        
        
    def scoreChromosome(self,chromosome,downsample=.5):
        '''
        returns the score criteria
        '''
        prior, post= self.generateDesign(chromosome,downsample)
        a1, a2 = ga.areaPercent(chromosome)
        multi_objective_function = -np.average((np.subtract(post,prior))**2), -4*np.var(50*np.subtract(post,prior)), -2*((min(a1,.95)**-1)-2*(min(a2,.95)**-1))
        return multi_objective_function
    
    def generateDesign(self,chromosome,downsample=1):
        '''
        Generates design from given chromosome
        '''
        genesPr, genesPo = ga.binaryDecode(chromosome)
        newImagePrior = im.genFromSplines2(genesPr,self.baseImagePrior,max(int(round(self.dim*downsample)),100))
        newImagePost = im.genFromSplines2(genesPo,self.baseImagePost,max(int(round(self.dim*downsample)),100))
        return newImagePrior, newImagePost
    
    
    
    

        
        
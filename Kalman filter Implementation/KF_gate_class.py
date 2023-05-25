# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:48:39 2023

@author: AVEREEN
"""
import numpy as np
from scipy.stats.distributions import chi2



chol = lambda x : np.linalg.cholesky(x)
inv = lambda x : np.linalg.inv(x)
chi2inv = lambda alpha,df : chi2.ppf(1-alpha/2, df)
app_series = lambda series,value : np.append(series, np.atleast_2d(value.ravel()).T,axis=1)
mat = lambda arr : np.atleast_2d(arr)
vsl = lambda arr : mat(arr).T

class SKF_gated:
    # constant velocity  
    def __init__(self,x0=np.zeros((2,1)),y0=np.zeros((1,1)),w=0.01,v=0.002,Ts=0.0051, alpha=.99):
        self.P=np.zeros((2,2))
        self.x_hist,self.y_hist=x0,y0
        self.P_hist_post,self.P_hist_prior=np.zeros((4,1)),np.zeros((4,1))
        self.x, self.y, self.w, self.v, self.Ts = x0, y0, w, v, Ts
        self.sys_mat()
        self.thresh = chi2inv(alpha, len(y0))
        
        
    
    def __call__(self,Y,u=np.zeros((1,1))):
        y=mat(Y)
        # Update the true and estimated state
        self.y_hist = app_series(self.y_hist, y )
        self.x_hist = app_series(self.x_hist, self.A@self.x_hist[:,-1] )#+ self.B@u)

        # Update the error covariance matrix
        self.P = self.A@self.P@self.A.T + self.Q
        self.P_hist_prior = app_series(self.P_hist_prior,self.P)
        
        # Chi square test
        inno = y-self.C@self.x_hist[:,-1]
        e = inno.T@inv(self.C@self.P@(self.C.T) + self.R)@inno
        
        self.e=e
        for s in range(len(y)):
            #% Compute Kalman gain
            if e <= self.thresh:
                L = self.P@vsl(self.C[s,:])/(vsl(self.C[s,:]).T@self.P@(vsl(self.C[s,:]))+self.R[s,s])
            elif e > self.thresh:
                L = np.zeros((2,1))
            else:
                L = np.zeros((2,1))
            self.L=L
            #% Get expected measurement and Correct state estimate
            self.update= np.add(vsl(self.x_hist[:,-1]) , vsl(L*(self.y_hist[s,-1]-mat(self.C[s,:])@vsl(self.x_hist[:,-1]))).T).ravel()
            self.x_hist[:,-1] = self.update

            #% Update the error covariance matrix
            self.P = (np.identity(2) - mat(L@mat(self.C[s,:])))@self.P
        

        # Store current P
        self.P_hist_post = app_series(self.P_hist_post,self.P)

        
        return self.x_hist[:,-1]
    
    
    def sys_mat(self):
        self.A=np.array([[0.9872,-0.0254],[0.0051,0.9999]])
        self.B=np.array([[0.0051],[0]])
        self.C=np.array([[1],[0]]).T
        self.D=np.array([[0]])
        self.Q=np.array([[self.w,0],[0,self.w]])
        self.R=np.array([[self.v]])
        
    def get_state(self,i=1):
        return self.x_hist[0,-i] 
    
    def get_state_hist(self):
        return self.x_hist[0,:] 
    
    def get_std_dev(self,i=1):
        return self.P_hist_pos[0,-i]
    
    def get_std_dev_hist(self,i=1):
        return self.P_hist_pos[0,-i]
        
    
            
            
        
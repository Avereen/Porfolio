# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:48:39 2023

@author: AVEREEN
"""
import numpy as np
from scipy.stats.distributions import chi2



chol = lambda x : np.linalg.cholesky(x)
inv = lambda x : np.linalg.inv(x)
chi2inv = lambda alpha,df : chi2(1-alpha/2, df)


class SKF_gated:
    # constant velocity  
    def __init__(self,x0=np.zeros((1,1)),y0=np.zeros((1,1)),w=0.01,v=0.002,Ts=0.0051, alpha=.99):
        self.P=np.zeros((2,2))
        self.x_hist,self.P_hist_post,self.P_hist_prior=np.zeros((2,1)),np.zeros((4,1)),np.zeros((4,1))
        self.x, self.y, self.w, self.v, self.Ts = x0, y0, w, v, Ts
        self.sys_mat()
        self.thresh = chi2inv(alpha, len(y0))
        
        
    
    def __call__(self,y,u=0):
        i=self.x_hist.shape[1]
        # Update the true and estimated state
        self.x_hist[:,i] = self.A @ self.x_hist[:,i-1] #+ self.B*u

        # Update the error covariance matrix
        self.P = self.A*self.P*self.A.T + self.Q
        self.P_hist_prior[:, i] = self.P.ravel()
        
        # Chi square test
        inno = y-self.C*self.x_hist[:,i]
        e = inno.T*inv(self.C*self.P*self.C.T + self.R)*inno
    
        for s in range(y):
            #% Compute Kalman gain
            if e <= self.thresh:
                L = self.P*self.C[s,:].T/(self.C[s,:]*self.P*self.C[s,:].T+self.R[s,s])
            elif e > self.thresh:
                L = 0
            
            #% Get expected measurement and Correct state estimate
            self.x_hist[:,i] = self.x_hist[:,i] + L*(y[s,i]-self.C[s,:]*self.x_hist[:,i])

            #% Update the error covariance matrix
            self.P = (np.identity(2) - L*self.C[s,:])*self.P
        

        # Store current P
        self.P_hist_post[:, i] = self.P.ravel()

        
        return self.x_hist[:,i]
    
    
    def sys_mat(self):
        self.A=np.exp(np.array([[1,self.Ts],[0,1]]))
        self.B=np.array([[1],[0]]).T
        self.C=np.array([[1],[0]])
        self.D=np.array([[0]])
        self.Q=np.array([[self.w,0],[0,self.w]])
        self.R=np.array([[self.v]])
        
    def get_state(self,i=1):
        return self.x_hist[:,-i] 
    
    def get_std_dev(self,i=1,ds=False):
        j=1
        if ds: 
            j=4 
        return self.P_hist_pos[j,-i]
        
    
            
            
        
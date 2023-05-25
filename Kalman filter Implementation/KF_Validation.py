# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:46:38 2023

@author: AVEREEN
"""


# import IPython as IP
# IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft

import numpy as np
import KF_gate_class as kf
mat = lambda arr : np.atleast_2d(arr)
beam_length= 0.3525

#estimated
estimated_LEMP_bayes= np.load('est_pin_loc_LEMP.npy')
estimated_GE=np.load('est_pin_loc_GE.npy')
estimated_LEMP=np.load('est_pin_loc_LEMP_no_Bayes.npy')

estimated_LEMP_bayes_lin_reg= np.load('est_pin_loc_LEMP_lin_reg.npy')
estimated_GE_lin_reg=np.load('est_pin_loc_GE_lin_reg.npy')
estimated_LEMP_lin_reg=np.load('est_pin_loc_LEMP_no_Bayes_lin_reg.npy')

estimated_pin_location_len1 = estimated_LEMP_bayes*beam_length
estimated_pin_location_len2 = estimated_GE*beam_length
estimated_pin_location_len3 = estimated_LEMP*beam_length
estimated_pin_location_len4 = estimated_LEMP_bayes_lin_reg*beam_length
estimated_pin_location_len5 = estimated_GE_lin_reg*beam_length
estimated_pin_location_len6 = estimated_LEMP_lin_reg*beam_length

#measured
measured_pin= np.load('measured_pin_loc.npy')
measured_pin_location=measured_pin

np.savetxt('measure_true.txt',measured_pin_location)
np.savetxt('measure_est.txt',estimated_pin_location_len3)

#time
est_time= np.load('est_time.npy')
measured_time= np.load('measured_time.npy')
np.savetxt('measure_time.txt',measured_time)


kfilter=kf.SKF_gated(x0=np.array([[measured_pin[0]],[0]]),y0=np.array([[measured_pin[0]]]))
for y in estimated_LEMP:
    kfilter(y)

rx1=np.linspace(0,45,len(estimated_pin_location_len1))
rx2=np.linspace(0,45,len(estimated_pin_location_len1)+1)
rx3=np.linspace(0,45,len(measured_pin))
plt.figure()
plt.plot(rx1,estimated_LEMP*beam_length,'--',label='raw lemp')
plt.plot(rx2,kfilter.get_state_hist()*beam_length,'--',label='filter')           
plt.plot(rx3,measured_pin,c='k',label='true')
plt.legend()
         
                 




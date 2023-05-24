# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:39:01 2023

@author: AVEREEN
"""

#%% Clear and close everything
clear; close all; clc;

#%% 0. Start by entering your last name
beam_length  = 0.3525;
lemp_measure = importdata("measure_lemp.txt","\n");
true_measure = importdata("measure_true.txt","\n")/beam_length;
time_true = importdata("measure_time.txt","\n");

%% clean nans

for iter = 1:length(true_measure)
    if isnan(true_measure(iter,1))
        p1 = true_measure(iter-1,1);
        true_measure(iter,1)=p1;
    end
end

#%% Initialize system constants
time_steps = length(lemp_measure);
time_lemp = linspace(min(time_true),max(time_true),time_steps)';
Ts = max(time_lemp)/(time_steps-1);
%delete(time_true)


X_true = [true_measure'];
X_true = resample(X_true,time_steps,length(true_measure))';
%delete(true_measure)
z = [lemp_measure'];
%delete(lemp_measure)
%u = resample(X_true,time_steps,length(true_measure));
u = diff(X_true);


#%% 1. Initialize the system
#% Initialize system constants

b= 2.5; k=5; m=1; 

A = [-b, -k; 1, 0];
B = [1; 0];
C = [1,0];
D = 0;

continous_sys = ss(A,B,C,D);
discrete_sys = c2d(continous_sys,Ts);
sysd = discrete_sys; %for convenience

#%% 2. Square-Root filter
#% Initialize the variables to store the estimated state,
#% the error covariance matrix, the process and the measurement noise
#% covariance

X_true = [X_true';[0,diff(X_true')]/Ts];
y = z;


X_est_sr = zeros([2, time_steps]);
X_est_sr(:, 1) = [0; 0];

#% State var store
P = [1, 0; 0, 2]; % state error covariance
SR_P = chol(P, 'lower'); 
P_store_post_sr = zeros([4, time_steps]); % Variable to store the elements of P
P_store_post_sr(:, 1) = P(:);

#% Noise sources
w_variance = 0.01; v_variance = 0.002;

Q = [w_variance, 0; 0, w_variance]; % process noise covariance
R = [v_variance]; % measurement noise covariance

SR_Q = chol(Q, 'lower');
SR_R = chol(R, 'lower');

#% Estimate storage
X_est_seq = zeros([2, time_steps]);
X_est_seq(:, 1) = X_true(:, 1);

#% State var store
P = [.17, 0; 0, .17]; % state error covariance
P_store_post_seq = zeros([4, time_steps]); % Variable to store the elements of P
P_store_post_seq(:, 1) = P(:);

P_store_prior_seq = zeros([4, time_steps]); % Variable to store the elements of P
P_store_prior_seq(:, 1) = P(:);

#% Chi square test
thresh = chi2inv(1-0.9,length(y(:,1)))
e_max=0;

#% Simulate the system for all provided time steps while collecting the
#% estimated states, a posteriori error covariance

for i = 2:time_steps
    % Update the true and estimated state
    X_est_seq(:,i) = sysd.A * X_est_seq(:,i-1); %+ sysd.B*u(i-1);

    % Update the error covariance matrix
    P = sysd.A*P*sysd.A' + Q;
    P_store_prior_seq(:, i) = P(:);
    
    % Chi square test
    inno = y(:,i)-sysd.C*X_est_seq(:,i);
    e = inno'*inv(sysd.C*P*sysd.C' + R)*inno;
    e_max = max([e_max,e]);
    
    for s = 1:length(y(:,i-1))
        #% Compute Kalman gain
        if e <= thresh:
            L = P*sysd.C(s,:)'/(sysd.C(s,:)*P*sysd.C(s,:)'+R(s,s));
        elif e > thresh:
            L = 0;
        
        #% Get expected measurement and Correct state estimate
        X_est_seq(:,i) = X_est_seq(:,i) + L*(y(s,i)-sysd.C(s,:)*X_est_seq(:,i));

        #% Update the error covariance matrix
        P = (eye(2) - L*sysd.C(s,:))*P;
    

    % Store current P
    P_store_post_seq(:, i) = P(:);

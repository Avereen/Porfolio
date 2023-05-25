# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:06:13 2022

@author: alexv
"""

#%% imports
import IPython as IP
IP.get_ipython().magic('reset -sf')
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%% Object delcaration

class Dual:
    def __init__(self, value, grad=[]):
        self.value = value
        self.grad = grad

    def __add__(self, other):
        value = self.value + other.value    
        grad = ((self, 1),(other, 1))
        return Dual(value, grad)
    
    def __mul__(self, other):
        value = self.value * other.value
        grad = ((self, other.value),(other, self.value))
        return Dual(value, grad)
    
    def __sub__(self, other):
        # negate
        nvalue = -1 * other.value
        ngrad = ((other, -1),)
        nother=Dual(nvalue, ngrad)
        # add
        value = self.value + nother.value    
        grad = ((self, 1),(nother, 1))
        return Dual(value, grad)

    def __truediv__(self, other):
        # invert
        invvalue = 1. / other.value
        invgrad = ((other, -1 / other.value**2),)
        invother = Dual(invvalue, invgrad)
        # multiply
        value = self.value * invother.value
        grad = ((self, invother.value),(invother, self.value))
        return Dual(value, grad)

#%% function delcarations

# Convert NumPy array of Numeric objects into an array of Dual objects:
array_as_duals = np.vectorize(lambda x : Dual(x))

# Convert array of Dual objects into a NumPy array of Numeric objects:
duals_as_value = np.vectorize(lambda dual : dual.value)

# Backprop 
def reverse_mode_auto_diff(dual):
    """ Compute the adjoint of the duals 
    w.r.t children.
    """
    # Stores the graph of adjoints after recursion calls on 
    gradients = defaultdict(lambda: 0)
    
    def reverse_mode_gradients(dual, propgate_value):
        for child, grad in dual.grad:
            # "Multiply the edges of a path":
            adjoint = propgate_value * grad
            # "Add together the different paths":
            gradients[child] += adjoint
            # recurse through graph:
            reverse_mode_gradients(child, adjoint)
    
    reverse_mode_gradients(dual, propgate_value=1)
    # propgate_value=1 is the always the output differentiated w.r.t. itself 
    # eg. y=h(z) --> dy/dh(z)=dy/dy*dy/dh(z)
    return gradients

# activation functions 
def tanh(x):
    value = np.tanh(x.value)
    grad = ((x, 1-np.tanh(x.value)**2), )
    return Dual(value, grad)

act_tanh = np.vectorize(lambda dual : tanh(dual))
    
def sigmoid(x):
    #sigm = lambda x : 1/(1+np.exp(-x))
    value = 1/(1+np.exp(-x.value))
    grad = ((x,value*(1-value)), )
    return Dual(value, grad)

act_sig = np.vectorize(lambda dual : sigmoid(dual))

# Binary threshold function 
def thresh(x,thresh=.5):
    # Maps the values greater than threshold to 1 else 0
    value = (np.sign(x.value-thresh)+1)/2
    grad = ((x,1-np.tanh(x.value)**2),)
    return Dual(value, grad)

heaviside = np.vectorize(lambda dual : thresh(dual))

# Drives error down peicewise continous and 
def ReLU(x):
    # Maps the values greater than threshold to 1 else 0
    value = max([0,x.value])
    grad = ((x,(1+np.sign(x.value))/2),)
    return Dual(value, grad)

act_ReLU = np.vectorize(lambda dual : ReLU(dual))

# ReLU but smooth and twice differentiable  
def softPlus(x, cheddar=1):
    value = np.log(1+np.exp(x.value*cheddar))
    grad = ((x,1/(1+np.exp(-x.value))),)
    return Dual(value, grad)

act_softPlus = np.vectorize(lambda dual : ReLU(dual))

# Convolve

def convolve2d(image,weights,pad=0):
    assert (len(np.unique(weights.shape))==1)and(len(weights.shape)==2)
    kernal_size = weights.shape[0]
    ks=kernal_size
    output_size = (image.shape[0]-ks+2*pad+1,
                   image.shape[1]-ks+2*pad+1)
    os = output_size
    input_size = (image.shape[0]+2*pad,
                  image.shape[1]+2*pad)
    ins = input_size
    # Prep
    if pad:
        input_image = array_as_duals(np.zeros(ins))
        for ind, dual in np.ndenumerate(image):
            input_image[ind[0]+1,ind[1]+1] = dual
    else:
        input_image=image
    
    # Convolute
    output_image = array_as_duals(np.zeros(os))
    for ind, dual in np.ndenumerate(output_image):
        output_image[ind[0],ind[1]] = np.sum(input_image[ind[0]:ind[0]+ks,ind[1]:ind[1]+ks] @ weights)
    
    return output_image



#%% Learning 
def grad_descent_weights(weights, gradients, learning_rate):
    for _, weight in np.ndenumerate(weights):
        weight.value -= learning_rate * gradients[weight]
        
        
#%% Plotting
        
def to_scalar(value):
    # Maps the values [[0,0],[0,1],[1,0],[1,1]] to the scalars [0.5, 0.0, 1.0, 0.75]
    value = ((-2*value[1]+2*value[0])+value[1]*value[0]+2)/4
    return value
    
def descision_bound_plot(weights, bias, X_span=((-2,2),(-2,2)),side=200):
    pts = side**2
    XX1, XX2 = np.meshgrid(np.linspace(X_span[0][0],X_span[0][1],num=side),
                           np.linspace(X_span[1][0],X_span[1][1],num=side))
    X_dist_ = np.array([XX1.ravel().T,XX2.ravel().T]).T
    X_dist = [array_as_duals(x) for x in X_dist_]
    
    Y_surf_bin = np.zeros(pts)
    for i in range(pts):
        Y_surf_bin[i] = to_scalar(duals_as_value(heaviside(act_sig( X_dist[i] @ weights + bias))))
       
    Y_surf_bin=Y_surf_bin.reshape((side,side)).T
    
    fig,ax =plt.subplots(1,figsize=(8,4))
    ax.contourf(XX1, XX2, Y_surf_bin,cmap="Set1")
    return (fig,ax)

def regression_plot(weights1,weights2,bias1,bias2):
    X_span=np.linspace(-1,1,num=200)
    
    X_dist=array_as_duals(X_span)
    
    Y_model = np.zeros_like(X_span)
    for i in range(len(X_dist)):
        Y_model[i] = duals_as_value(act_tanh( act_tanh( np.array([X_dist[i]])@weights1 + bias1)@weights2+ bias2))

    fig,ax =plt.subplots(1,figsize=(8,4))
    ax.plot(X_span,Y_model)
    ax.scatter(X,Y,color='k')
    return (fig,ax)



#%% Single layer neural net
if __name__ == "__main__":
    np.random.seed(0)
    
    X=np.array([[0.1 , 0.7 , 0.8 , 0.8 , 1.0 , 0.3 , 0.0 , -0.3 , -0.5 , -1.5 ],
                [1.2 , 1.8 , 1.6 , 0.6 , 0.8 , 0.5 , 0.2 ,  0.8 , -1.5 , -1.3 ]],dtype=float)
    
    Y=np.array([[1 , 1 , 1 , 0 , 0 , 1 , 1 , 1 , 0 , 0 ],
                [0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 ]],dtype=float)
    
    input_shape = min(np.shape(X))
    output_shape = min(np.shape(Y))
    samples = max(np.shape(X))
    epochs = 101
    learning_rate = 0.1
    
    X_var = [array_as_duals(X[:,i]) for i in range(samples)]
    Y_true_var = [array_as_duals(Y[:,i]) for i in range(samples)]
    weights = array_as_duals(np.random.random((input_shape, output_shape)))
    bias = array_as_duals(np.random.random(output_shape))
    
    loss_history = []
    for epoch in range(epochs):
        loss_average = []
        for i in range(samples):
            Y_pred = act_sig( X_var[i] @ weights + bias)
            loss = np.sum((Y_true_var[i] - Y_pred) * (Y_true_var[i] - Y_pred))# Square error
            loss_average.append(loss.value)
            gradients = reverse_mode_auto_diff(loss)
            grad_descent_weights(weights, gradients, learning_rate)
            grad_descent_weights(bias, gradients, learning_rate)
        loss_history.append(np.mean(loss_average))
        if epoch in [3,10,100]:
            (fig,ax)=descision_bound_plot(weights,bias)
            ax.scatter(X[1,:],X[0,:],color='k')
            plt.title('B.P1(2) decision boundary after {} epochs of training'.format(epoch))
            plt.savefig('decision_boundary_{}_epochs.png'.format(epoch),dpi=300)
    
    fig,ax = plt.subplots(1,figsize=(8,4))
    ax.plot([e for e in range(epochs)],loss_history)
    plt.title('B.P1(1) training error vs epoch number')
    ax.set_ylabel('Loss MSE')
    ax.set_xlabel('epochs')
    plt.savefig('multiclass_training_error_vs_epoch.png',dpi=300)
    
    for i in range(samples):
            Y_dec = heaviside(act_sig( X_var[i] @ weights + bias))
            print('Input:')
            print(duals_as_value(X_var[i]))
            print('Predicted Group:')
            print(duals_as_value(Y_dec))
            print('Target Group:')
            print(duals_as_value(Y_true_var[i]))
            print('\n')
            
#%% Two layer Neural Network
    
    X = np.asarray([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],dtype=float)
    Y = np.asarray([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134,
                    -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396,
                    0.345,0.182, -0.031, -0.219, -0.321],dtype=float)
    
    
    input_shape = 1
    hidden_layer = 40
    output_shape = 1
    samples = len(X)
    epochs = 1001
    learning_rate = 0.045
    
    X_var = array_as_duals(X)
    Y_true_var = array_as_duals(Y)
    weights1 = array_as_duals(np.random.random((input_shape, hidden_layer)))
    weights2 = array_as_duals(np.random.random((hidden_layer, output_shape)))
    bias1 = array_as_duals(np.random.random((input_shape,hidden_layer)))
    bias2 = array_as_duals(np.random.random(output_shape))
    params = [weights1,weights2,bias1,bias2]
    
    loss_history = []
    for epoch in range(epochs):
        loss_average = []
        for i in range(samples):
            Y_pred = act_tanh( act_tanh( np.array([X_var[i]])@weights1 + bias1)@ weights2 +bias2)
            Y_true = np.array([Y_true_var[i]])
            loss = np.sum(( Y_true - Y_pred) * (Y_true - Y_pred))# Square error
            loss_average.append(loss.value)
            gradients = reverse_mode_auto_diff(loss)
            for param in params:
                grad_descent_weights(param, gradients, learning_rate)
        loss_history.append(np.mean(loss_average))
        if epoch in [10, 100, 200, 400, 1000, 2000, 5000]:
            (fig,ax)=regression_plot(weights1,weights2,bias1,bias2)
            plt.title('B.P2(2) regression model: hidden layer {}, learning rate {}, epochs {}'.format(hidden_layer,learning_rate,epoch))
            plt.savefig('regression_model_{}_epochs.png'.format(epoch),dpi=300)
    
    fig,ax = plt.subplots(1,figsize=(8,4))
    ax.plot([e for e in range(epochs)],loss_history)
    plt.title('B.P2(1) training error vs epoch number')
    ax.set_ylabel('Loss MSE')
ax.set_xlabel('epochs')
plt.savefig('regression_training_error_vs_epoch.png',dpi=300)


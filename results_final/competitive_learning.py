#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:36:54 2019

@author: uddhav
"""

import numpy as np
import numpy.matlib
import math
import matplotlib
import matplotlib.pyplot as plt

letters = np.genfromtxt ('letters.csv', delimiter=",")

[n,m]  = np.shape(letters)                    # number of pixels and number of training data
norm_train = np.sqrt(np.diag(letters.T.dot(letters)))

n_train = letters / np.matlib.repmat(norm_train,n,1)

eta    = 0.06                               # learning rate for the winners
eta_l  = 0.00006                            # learning rate for the losers
eta_d  = 0.03                             
winit  = 1                                  # parameter controlling magnitude of initial conditions
alpha = 0.999

tmax   = 10000
digits = 16

# W = winit * np.random.rand(digits,n)        # Weight matrix (rows = output neurons, cols = input neurons)
# normW = np.sqrt(np.diag(W.dot(W.T)))
# normW = normW.reshape(digits,-1)            # reshape normW into a numpy 2d array

# W = W / normW                               # normalise using numpy broadcasting -  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html


counter = np.zeros((1,digits))              # counter for the winner neurons
wCount = np.ones((1,tmax+1)) * 0.25         # running avg of the weight change over time

W = np.zeros((digits,n))
for t in range(digits):
    i = math.ceil(m * np.random.rand())-1
    y = n_train[:,i]
    W[t,:] = y

normW = np.sqrt(np.diag(W.dot(W.T)))
normW = normW.reshape(digits,-1)        

W = W / normW

for t in range(1,tmax):
    i = math.ceil(m * np.random.rand())-1   # get a randomly generated index in the input range
    x = n_train[:,i]                          # pick a training instance using the random index

    h = W.dot(x)/digits                     # get output firing
    h = h.reshape(h.shape[0],-1)            # reshape h into a numpy 2d array

    output = np.max(h)                      # get the max in the output firing vector
    k = np.argmax(h)                        # get the index of the firing neuron

    counter[0,k] += 1                       # increment counter for winner neuron

#    dw = eta * (x.T - W[k,:])               # calculate the change in weights for the k-th output neuron
#                                            # get closer to the input (x - W)
#    W[k,:] = W[k,:] + dw
    
    for q in range(digits):
        if (q == k):
            dw = eta * (x.T - W[q,:])
            W[q,:] = W[q,:] + dw
        if (q != k):
            dw = eta_l * (x.T - W[q,:])
            W[q,:] = W[q,:] + dw       


    eta = (eta * ((eta_d/eta) ** (t / (tmax))) - 1/tmax)  

    wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)

# Plot a prototype

width = 10
height = 10
fig = plt.figure(figsize=(18, 18))
columns = 4
rows = 4

# prep (x,y) for extra plotting
xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
ys = np.abs(np.sin(xs))           # absolute of sine

# ax enables access to manipulate each of subplots
ax = []

for i in range( columns*rows ):
    img = np.random.randint(10, size=(height,width))
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("ax:"+str(i))  # set title
    plt.imshow(W[i,:].reshape((88,88), order = 'F'),interpolation = 'nearest', cmap='inferno')
plt.savefig('prototype.png')  # finally, render the plot

plt.loglog(wCount[0,0:tmax], linewidth=2.0, label='rate')
plt.xlabel('Iterations')
plt.ylabel('Change in Weight')
plt.suptitle('Change in weight over time')
plt.savefig('dw.png')

ching = np.corrcoef(W)
plt.matshow(ching)
plt.colorbar(fraction = 0.046)
plt.savefig('cor.png')

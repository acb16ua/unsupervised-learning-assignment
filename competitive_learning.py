#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:36:54 2019

@author: Uddhav Agarwal
"""

import numpy as np
import numpy.matlib
import math
import matplotlib
import matplotlib.pyplot as plt

######################################################################################################
######################%%%% Function that return the normalised training data %%%%#####################
######################################################################################################
def getNormInput(URL):
    letters = np.genfromtxt (URL, delimiter=",")            # extracting the training data

    [n,m]  = np.shape(letters)                              # number of pixels and number of train data
    norm_train = np.sqrt(np.diag(letters.T.dot(letters)))
    n_train = letters / np.matlib.repmat(norm_train,n,1)    # normalising the input data

    return n, m, n_train

######################################################################################################
##################################%%%% THE LEARNING FUNCTION %%%%#####################################
######################################################################################################
def competitive(train, eta, eta_l, eta_d, digits, weight, leaky, decay):
    
    alpha = 0.999
    tmax   = 10000                                          # number of iterations the algorithm runs for
    winit = 1
    counter = np.zeros((1,digits))                          # counter for the winner neurons
    wCount = np.ones((1,tmax+1)) * 0.25                     # running avg of the weight change over time
    
    if (weight == 'random'):
        W = winit * np.random.rand(digits,n)                # Random weight matrix
    elif (weight == 'input'):
        W = np.zeros((digits,n))                            # Weight matrix from input
        for t in range(digits):                             # randomly assigning input weights to output
            i = math.ceil(m * np.random.rand())-1
            y = n_train[:,i]
            W[t,:] = y

    normW = np.sqrt(np.diag(W.dot(W.T)))
    normW = normW.reshape(digits,-1)                        # reshape normW into a numpy 2d array
    W = W / normW
    
    for t in range(1,tmax):
        i = math.ceil(m * np.random.rand())-1               
        x = n_train[:,i]                                    # randomly pick a training instance 

        h = W.dot(x)/digits                                 # get output firing
        h = h.reshape(h.shape[0],-1)                        
        
        output = np.max(h)                                  # get the max in the output firing vector
        k = np.argmax(h)                                    # get the index of the firing neuron

        counter[0,k] += 1                                   # increment counter for winner neuron

        if (leaky == 'not-leaky'):                          # only update the winner
            dw = eta * (x.T - W[k,:])                       # calculate the change in weights for the k-th output neuron
            W[k,:] = W[k,:] + dw                            # get closer to the input (x - W)
        elif (leaky == 'leaky'):
            # leaky learning loop
            for q in range(digits):                             
                if (q == k):
                    dw = eta * (x.T - W[q,:])               # calculate the change in weights for the k-th output neuron
                    W[q,:] = W[q,:] + dw                    # change weights to get closer to a cluster
                if (q != k):
                    dw = eta_l * (x.T - W[q,:])
                    W[q,:] = W[q,:] + dw       


        if (decay == 'decay'):
            eta = (eta * ((eta_d/eta) ** (t / (tmax))) - 1/tmax) # learning rate decay:
                                                                 # uses a combination of exponential 
                                                                 # and harmonic decay
        elif (decay == 'not-decay'):
            eta = eta                                                         

        wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)
        
        
    return counter, tmax, W, wCount
        

######################################################################################################
##########################%%%% FUNCTION TO DISPLAY AND SAVE RESULTS %%%%##############################
######################################################################################################
def results(train, eta=0.06, eta_l=0.00006, eta_d=0.03, digits=16, weight='random', leaky='not-leaky', decay='not-decay'):    
    
    token = []
           
    for numb in range(5):
        counter, tmax, W, wCount = competitive(train, eta, eta_l, eta_d, digits, weight, leaky, decay)
        checker = []
        
        for k in counter:
            for j in k:
                check = (j/tmax)*100
                if (check < 4):
                    checker.append(check)
            
        bing = len(checker)
        token.append(bing)
        
        
        #### Plotting and saving the prototypes ####
        width = 10
        height = 10
        fig = plt.figure(figsize=(18, 18))
        columns = 4
        rows = 4
        xs = np.linspace(0, 2*np.pi, 60)
        ys = np.abs(np.sin(xs))           
        ax = []
    
        for i in range( columns*rows ):
            img = np.random.randint(10, size=(height,width))
            ax.append( fig.add_subplot(rows, columns, i+1) )
            ax[-1].set_title("ax:"+str(i))
            plt.imshow(W[i,:].reshape((88,88), order = 'F'),interpolation = 'nearest', cmap='inferno')
        plt.savefig(weight+'_'+leaky+'_'+decay+'_'+'prototype('+str(numb)+').png')
        plt.show()
        plt.clf()
            
        #### Plotting and saving the change in weight ####    
        plt.loglog(wCount[0,0:tmax], linewidth=2.0, label='rate')
        plt.xlabel('Iterations')
        plt.ylabel('Change in Weight')
        plt.suptitle('Change in weight over time')
        plt.savefig(weight+'_'+leaky+'_'+decay+'_'+'change_in_weight('+str(numb)+').png')
        plt.show()
        plt.clf()
    
        #### Plotting and saving the correlation matrix ####
        cor_matrix = np.corrcoef(W)
        plt.matshow(cor_matrix)
        plt.colorbar(fraction = 0.046)
        plt.savefig(weight+'_'+leaky+'_'+decay+'_'+'cor_matrix('+str(numb)+').png')
        plt.show()
        plt.clf()
    
    #### Calculating the mean and stdv ####
    meanCalc = sum(token) / len(token)
    stdCalc = np.std(token)
    
    #### Saving the mean and stdv in a text file ####
    file = open('results.txt', 'a+')
    file.write('\n\n-----------------------------------------')
    file.write('\nINITIAL CONDITIONS: \neta = %01.2f \neta (for losers in Leaky) = %01.6f \n' % (eta, eta_l))
    file.write('eta (lower limit in eta decay) = %01.2f \n\nResults for %s_%s_%s: \n' % (eta_d, weight, leaky, decay))
    file.write('Mean: %01.3f \nStandard Deviation: %01.12f' %(meanCalc, stdCalc))
    file.close()
    
    print('Mean:'+ ' ' + str(meanCalc))
    print('STD:'+ ' ' + str(stdCalc))
    

######################################################
############%%%% EXAMPLE TEST CASES %%%%##############    
######################################################
    
n, m, n_train = getNormInput('letters.csv')
   
results(n_train, 0.07, 0.00007, 0.03, 16, 'input', 'leaky', 'decay')
#results(n_train, 0.08, 0.00009, 0.04, 16, 'random', 'leaky', 'not-decay')  
#results(n_train, 0.06, 0.00005, 0.02, 16, 'input', 'not-leaky', 'decay')
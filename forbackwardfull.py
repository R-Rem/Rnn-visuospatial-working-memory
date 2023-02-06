# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 23:06:19 2022

@author: Robin
"""

import numpy as np



def forbackwardfull2(inputt,target,hprev,nItem,Wxh,Whh,Why,bh,by,loss2,epoch,nInput,learning_rate,ti):

    
  # Initial incoming state.
  tI, rInput, xs, hs, ys, ps = {}, {}, {}, {}, {}, {}

  hs[-1] = np.copy(hprev)

  loss = 0

  


  for t in range(0,nItem):

     # Input at time step t is xs[t]. Prepare a encoded vector of shape


     rInput = inputt[:,t]

     rInput.shape = (nInput,1)

     xs[t] = rInput # encode in 1-of-k representation


     # Compute h[t] from h[t-1] and x[t]

     hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)


     # Compute ps[t]

     ys[t] = np.dot(Why, hs[t]) + by

     # Compute ys[t]
      
     ps[t] = np.tanh(ys[t])

      # Backward pass: compute gradients going backwards.

      # Gradients are initialized to 0s, and every time step contributes to them.
      
     dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)


     dbh, dby = np.zeros_like(bh), np.zeros_like(by)



     # Initialize the incoming gradient of h to zero; this is a safe assumption for

     # a sufficiently long unrolling.

     dhnext = np.zeros_like(hs[0])



    # The backwards pass iterates over the input sequence backwards.
     # Backprop through the gradients of loss and softmax.

     
     tI = ti[:,t]
     
     tI.shape = (nInput,1)
     
     
     E = ps[t] - tI
     
     dy = (1- ps[t] * ps[t])*E
     
     
     err = np.sum(np.abs(E))

     loss += err


     # Compute gradients for the Why and by parameters.

     dWhy = np.dot(dy, hs[t].T)

     dby = dy


     # Compute gradients for the hidden layer.

     dh = np.dot(Why.T, dy) + dhnext

    

     # Backprop through tanh.

     dhraw = (1 - hs[t] * hs[t]) * dh

     # Compute gradients for the dby, dWxh, Whh parameters.

     dbh = dhraw

     dWxh = np.dot(dhraw, xs[t].T)

     dWhh = np.dot(dhraw, hs[t-1].T)

    

     # Backprop the gradient to the incoming h, which will be used in the

     # previous time step.

     dhnext = np.dot(Whh.T, dhraw)

    

     

     # Memory variables for Adagrad.

     mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

     mbh, mby = np.zeros_like(bh), np.zeros_like(by)


     # Perform parameter update with Adagrad

     for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],

                                [dWxh, dWhh, dWhy, dbh, dby],

                                [mWxh, mWhh, mWhy, mbh, mby]):

      mem += dparam * dparam

      adaptativelr= learning_rate / np.sqrt(mem + 0.4)

      param -= adaptativelr * dparam

    


  loss2[epoch] = np.sum(loss)

    

  return loss2

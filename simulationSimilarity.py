# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:58:54 2022

@author: Robin
"""

import numpy as np

import matplotlib.pyplot as plt

from forbackwardfull import forbackwardfull

from testBipolar import testBipolar

from genSimDis import genSimDis

global loss



# parameters

nHidden = 200 

letters = 7

nItem = (2*letters )+1

nSeq = 400

nEpoch = 150

nInput = 21

learning_rate = 0.001


# generate sequence to learn
gen2 = genSimDis(nSeq,letters,nItem,nInput)

inputs = gen2[0]
targets = gen2[1]

testInput2 = gen2[2]
testOutput2 = gen2[3]


# Initial incoming state.
hprev = np.zeros((nHidden,1))
loss = 0

# Store the loss
loss2 = np.zeros((nEpoch))

# create the x axis
x = np.array(range(0,nEpoch))


# weight matrix

Wxh = np.random.randn(nHidden, nInput) * 0.01       # input to hidden

Whh = np.random.randn(nHidden, nHidden) * 0.01      # hidden to hidden

Why = np.random.randn(nInput, nHidden) * 0.01       # hidden to output


bh = np.zeros((nHidden, 1))     # hidden bias

by = np.zeros((nInput, 1))      # output bias


# perform forward and backward pass
 
for epoch in range (0,nEpoch):

    for Seq in range(0,nSeq):
    
    
        inputt = inputs[Seq].T
    
        ti = targets[Seq].T
            
        target = targets[Seq]
    
        target = np.where(target==1)[1]
    
        loss2 = forbackwardfull(inputt,target,hprev,nItem,Wxh,Whh,Why,bh,by,loss2,epoch,nInput,learning_rate,ti)


# Plot error
plt.plot(x,loss2)

plt.show()






#TEST
nSeq=100
mscore = testBipolar(gen2,hprev,letters,nInput,nSeq,Wxh,Whh,Why,by,bh,nItem,learning_rate)

# Plot the score
axeX = np.arange(1,letters +1)

# Retrieve the similar and dissimilar condition
idsampleTest = gen2[4]

scrSim = mscore[2][idsampleTest[0:int(nSeq/2)]]

scrDis = mscore[2][idsampleTest[int(nSeq/2):nSeq]]


perDis = np.zeros((letters))
perSim = np.zeros((letters))

for i in range(0,letters):

    perDis[i] = len(np.where(scrDis[:,i]==i)[0])
    perSim[i] = len(np.where(scrSim[:,i]==i)[0])
    

perDis = perDis/int(nSeq/2)*100

perSim = perSim /int(nSeq/2)*100  


plt.plot(axeX,perDis , label='Dissimilar')
plt.plot(axeX,perSim , label='Similar')
plt.legend()
plt.show()












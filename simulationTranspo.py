

import numpy as np

import matplotlib.pyplot as plt

from forbackwardfull import forbackwardfull

from genStim import genStim

from genBipolar import genBipolar

from testBipolar import testBipolar

global loss



# parameters

nHidden = 200 

letters = 7

nItem = (2*letters )+1

nSeq = 500

nEpoch = 300

nInput = 21

learning_rate = 0.001


# generate sequence to learn
gen = genStim(nSeq,letters,nItem,nInput)

inputs = gen[0]
targets = gen[1]

testInput = gen[2]
testOutput = gen[3]

# Transform one hot encoding to bipolar encoding
gen2 = genBipolar(nInput,inputs,targets,testInput,testOutput,nItem,letters,nSeq)

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
gen2[2] = gen2[2][0:100]
gen2[3] = gen2[3][0:100]

nSeq=100

mscore = testBipolar(gen2,hprev,letters,nInput,nSeq,Wxh,Whh,Why,by,bh,nItem,learning_rate)


#Plot the score
correctpos = np.zeros((nSeq,letters))

correctpos[:,0:7] = [0,1,2,3,4,5,6]

transposcore = correctpos - mscore[2]

eachpos = np.zeros((1,13))

idxdis = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]

for i in range(0,13):
    
    eachpos[0,i] = len(np.where(transposcore== idxdis[i])[0])
   

per =eachpos/700*100

axeX = np.arange(1,14)
axeY = per

plt.plot(axeX,axeY)
plt.xticks(axeX,['-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6'])
plt.show()

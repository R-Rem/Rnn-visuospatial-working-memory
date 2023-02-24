


import numpy as np

import matplotlib.pyplot as plt

from forbackwardfull import forbackwardfull

from genStim import genStim

from listlenght import listlenght

from genBipolar import genBipolar

from testListLenght import testListLenght

from genBipolarlenght import genBipolarlenght

global loss


# parameters

nHidden = 200 

letters = 8

nItem = (2*letters )+1

nSeq = 500

nEpoch = 300

nInput = 21

learning_rate = 0.001

nId = 14

# generate sequence to learn
gen = genStim(nSeq,letters,nItem,nInput)

inputs = gen[0]
targets = gen[1]

testInput = gen[2]
testOutput = gen[3]

# generate every test sequence to test
testlenght = listlenght(nInput,nItem)


# transform one hot encoding to bipolar encoding
gen2 = genBipolar(nInput,inputs,targets,testInput,testOutput,nItem,letters,nSeq)

inputs = gen2[0]
targets = gen2[1]

testInput2 = gen2[2]
testOutput2 = gen2[3]

# transform one hot encoding to bipolar encoding
testlenbipolar = genBipolarlenght(nInput,testlenght)


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

# Save the weight
Whhsave = np.copy(Whh)
Whysave  = np.copy(Why)
Wxhsave  = np.copy(Wxh)

bhsave = np.copy(bh)
bysave = np.copy(by)


mscore = np.zeros((len(testlenbipolar[0])))

nSeq = 100

for span in range(len(testlenbipolar[0])):
    
    # Reinitialize weight before each test lenght
    Whh = np.copy(Whhsave)
    Why = np.copy(Whysave)
    Wxh = np.copy(Wxhsave)

    bh = np.copy(bhsave)
    by = np.copy(bysave)
    
    #readapt the dimension
    nItem = len(testlenbipolar[0][span][0])
    letters = int((nItem-1)/2)
    
    
    mscore[span] = testListLenght(span,testlenbipolar,hprev,letters,nInput,nSeq,Wxh,Whh,Why,by,bh,nItem,learning_rate)


# Plot the score
axeX = np.arange(1,len(testlenbipolar[0])+1)
axeY = mscore

plt.plot(axeX,axeY)
plt.show()

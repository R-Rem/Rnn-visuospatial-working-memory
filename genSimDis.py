# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:02:14 2022

@author: Robin
"""

from genStim3 import genStim3

from simil import simil

from genBipolar2 import genBipolar2

import numpy as np

import random as rand

def genSimDis(nSeq,letters,nItem,nInput):
    
  nSeqsim = int(nSeq/2)#200
  
  nTest = 100
  nTest2 = int(nTest/2)
 
  #Pre allocate memory
  inputs = np.zeros((nSeq,nItem,nInput))  
  targets = np.zeros((nSeq,nItem,nInput))
  
  testInput = np.zeros((nSeq,nItem,nInput))
  testOutput = np.zeros((nSeq,nItem,nInput))
  
  testIn = np.zeros((nTest,nItem,nInput))
  testOut = np.zeros((nTest,nItem,nInput))
  
  # generate sequence to learn  
  gen = genStim3(nSeq,letters,nItem,nInput)


  # generate similar sequence
  genSim = simil(nSeq,nItem,nInput,gen,letters)

 
  #randomize the set training
  idsample = rand.sample(range(0,nSeq),nSeq)
    
  inputs[idsample[0:nSeqsim]] = gen[0][0:nSeqsim]
  inputs[idsample[nSeqsim:nSeq]] =  genSim[0][0:nSeqsim]
    
  targets[idsample[0:nSeqsim]] = gen[1][0:nSeqsim]
  targets[idsample[nSeqsim:nSeq]] =  genSim[1][0:nSeqsim]
  

  testInput[0:nSeqsim] = genSim[2][0:nSeqsim] #0 to 200 similarity
  testInput[nSeqsim:nSeq] = gen[2][0:nSeqsim] #200 to 400 dissimilarity 

  testOutput[0:nSeqsim] = genSim[3][0:nSeqsim] 
  testOutput[nSeqsim:nSeq] = gen[3][0:nSeqsim] 
  
  
  
  # transform one hot encoding to bipolar encoding
  gen2 = genBipolar2(nInput,inputs,targets,testInput,testOutput,nItem,letters,nSeq)
  
  simIn = gen2[2][0:nSeqsim]
  dissIn = gen2[2][nSeqsim:nSeq]
  
  simOut = gen2[3][0:nSeqsim]
  dissOut = gen2[3][nSeqsim:nSeq]
  
  
  #randomize the set test
  idsampleTest = rand.sample(range(0,nTest),nTest)
  
  testIn[idsampleTest[0:nTest2]] = simIn[0:nTest2]#0 to 50 sim
  testIn[idsampleTest[nTest2:nTest]] = dissIn[0:nTest2] 

  testOut[idsampleTest[0:nTest2]] = simOut[0:nTest2] 
  testOut[idsampleTest[nTest2:nTest]] = dissOut[0:nTest2]
  
  gen2[2] = testIn
  
  gen2[3] = testOut
  
  gen2[4] = idsampleTest
  
  return gen2
  
  
  
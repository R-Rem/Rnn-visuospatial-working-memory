# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:23:44 2022

@author: Robin
"""


import numpy as np

from scipy import spatial

from forbackward_test import forbackward_test

def testBipolar(gen2,hprev,letters,nInput,nSeq,Wxh,Whh,Why,by,bh,nItem,learning_rate):

   # Reinitialize state
   mscore= {}
   allSim = {}

   # Select the testing set
   tests = gen2[2]
   targets = gen2[3]
   
   
   # Pre allocate memory
   strictScore = np.zeros((nSeq,nItem))
   transpo = np.zeros((nSeq,nItem))
   sim = np.zeros((nItem,letters))
   PsRep = np.zeros((nItem,nInput))
      
   # Perform only the forward pass to test the network
   for t in range(nSeq):
        
       # extract the correct answer
       exact = gen2[3][t][0:letters,:]
        
       inputt = tests[t].T
       
       ti = targets[t].T
       
       # Perform the 5 epoch correction
       for j in range(0,5):
           ps = forbackward_test(inputt,hprev,nItem,Wxh,Whh,Why,bh,by,nInput,learning_rate,ti)
           
       for seq in range(0,nItem):
           for i in range(letters):
           
               sim[seq,i] = 1- spatial.distance.cosine(exact[i,:],ps[seq])
               
           PsRep[seq] = exact[np.argmax(sim[seq,:])]
           
           transpo[t,seq] = np.argmax(sim[seq,:])
           
       # compare the score with the exact score
       strictScore[t] = np.all(PsRep == gen2[3][t],axis=1)
      
       allSim[t] = sim
      
   # Compute the mean    
   mscoreEncodage = np.mean(strictScore[:,0:letters],axis = 0)
   mscoreRappel = np.mean(strictScore[:,letters:2*letters],axis = 0)
   mscorespan = np.mean(strictScore[:,letters:2*letters],axis = 1)
   nb = (len(np.where(mscorespan==1)[0])/nSeq)*100
    
   mscore[0] = mscoreEncodage
   mscore[1] = mscoreRappel
   mscore[2] = transpo[:,letters:letters*2]
   mscore[3] = nb
   mscore[4] = strictScore[:,letters:2*letters]
    
   return mscore
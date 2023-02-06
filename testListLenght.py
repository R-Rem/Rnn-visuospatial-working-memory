# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:39:22 2022

@author: Robin
"""


import numpy as np

from scipy import spatial

from forbackward_test import forbackward_test

def testListLenght(span,testlenbipolar,hprev,letters,nInput,nSeq,Wxh,Whh,Why,by,bh,nItem,learning_rate):

   
   

   # Select the testing set
   tests = testlenbipolar[0][span]
   targets = testlenbipolar[1][span]
   
   
   # Pre allocate memory
   strictScore = np.zeros((nSeq,nItem))
   transpo = np.zeros((nSeq,nItem))
   sim = np.zeros((nItem,letters))
   PsRep = np.zeros((nItem,nInput))
   
      
   # Perform only the forward pass to test the network
   for t in range(nSeq):
        
       # extract the correct answer
       exact = testlenbipolar[1][span][t][0:letters,:]
        
       inputt = tests[t].T
       
       ti = targets[t].T
       
      
       for j in range(0,5):
        
            ps = forbackward_test(inputt,hprev,nItem,Wxh,Whh,Why,bh,by,nInput,learning_rate,ti)
            
           
       for seq in range(0,nItem):
           for i in range(letters):
           
               sim[seq,i] = 1- spatial.distance.cosine(exact[i,:],ps[seq])
               
           PsRep[seq] = exact[np.argmax(sim[seq,:])]
           
           transpo[t,seq] = np.argmax(sim[seq,:])
           
       # compare the score with the exact score
       strictScore[t] = np.all(PsRep == testlenbipolar[1][span][t],axis=1)
      
      
      
   # Compute the mean    
   mscorespan = np.mean(strictScore[:,letters:2*letters],axis = 1)
   
   #Number of entire sequence correct
   nb = int((len(np.where(mscorespan==1)[0])/nSeq)*100)
    
    
   mscore = nb
   
    
   return mscore
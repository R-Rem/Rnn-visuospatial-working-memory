# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:12:49 2022

@author: Robin
"""

import numpy as np
import random as rand

def listlenght (nInput,nItem):

    letters = 2
    nLocation = nInput-1
    
    #axe x and axe y
    idcons1 = [0,1,2,3,4,5,6,7,8,9]
    idcons2 = [10,11,12,13,14,15,16,17,18,19]
   
    genInput = {}
    genOutput = {}
    gen = {}
    
    nSeq = 100
 
    
    for f in range (0,6):
        
        letters = letters+1
        nItem = (2*letters )+1
        letters_output = np.zeros((nSeq,nItem,nInput))
        letters_input = np.zeros((nSeq,nItem,nInput)) 
   
        for i in range (0,nSeq):
            
            # generate random index
            id1 = rand.sample(idcons1,letters)
            id2 = rand.sample(idcons2,letters)
            
            letters_input[i][letters:nItem,nLocation] = 1 
            letters_output[i][letters:nItem,nLocation] = 1
            
            for j in range (0, letters):
                letters_output[i][j,id1[j]] = 1
                letters_output[i][j,id2[j]] = 1
            
            letters_input[i][0:letters,:] = letters_output[i][0:letters,:]
                  
            letters_output[i][letters:letters*2,:] = letters_output[i][0:letters,:]
            
        
        genInput[f] = letters_input
        genOutput[f] = letters_output
   
    gen[0] = genInput
    gen[1] = genOutput
   
    return gen
   
   
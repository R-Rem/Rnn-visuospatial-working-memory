# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:50:42 2022

@author: Robin
"""

import numpy as np
import random as rand

def genStim(nSeq,letters,nItem,nInput):

   
    
    nLocation = nInput-1
   
   
    
    idcons1 = [0,1,2,3,4,5,6,7,8,9]#axe x
    idcons2 = [10,11,12,13,14,15,16,17,18,19]#axe y
    
    # pre allocate memory
    list_input = np.zeros((nSeq,nItem,nInput))
    list_output = np.zeros((nSeq,nItem,nInput))
    test_list_input = np.zeros((nSeq,nItem,nInput))
    test_list_output = np.zeros((nSeq,nItem,nInput))
    gen = {}
    
    
   
    for i in range (0,nSeq):
        
        #generate random index
        id1 = rand.sample(idcons1,letters)
        id2 = rand.sample(idcons2,letters)
        
        #pre allocate memory
        letters_output = np.zeros((nItem,nInput))
        letters_input = np.zeros((nItem,nInput)) 
        test_let_input = np.zeros((nItem,nInput)) 
        test_let_output = np.zeros((nItem,nInput))
        
        letters_input[letters:nItem,nLocation] = 1 
        test_let_input[letters:nItem,nLocation] = 1
        test_let_output[letters:nItem,nLocation] = 1
        letters_output[letters:nItem,nLocation] = 1
        
        for j in range (0, letters):
            letters_output[j,id1[j]] = 1
            letters_output[j,id2[j]] = 1
        
        letters_input[0:letters,:] = letters_output[0:letters,:]
        
       
        idperm = rand.sample(range(0,letters),4)
        test_let_input[0:letters,:] = letters_input[0:letters]
        test_let_input[[idperm[0],idperm[1]]] = test_let_input[[idperm[1],idperm[0]]]
        test_let_input[[idperm[2],idperm[3]]] =  test_let_input[[idperm[3],idperm[2]]]
        
        test_let_output[0:letters,:] = test_let_input[0:letters,:]
        test_let_output[letters:letters*2,:] = test_let_output[0:letters,:]
        letters_output[letters:letters*2,:] = letters_output[0:letters,:]
        
        list_output[i] = letters_output
        list_input[i] = letters_input
        test_list_input[i] = test_let_input
        test_list_output[i] = test_let_output
    
    gen[0] = list_input
    gen[1] = list_output
    gen[2] = test_list_input
    gen[3] = test_list_output
    
    return gen
   
   
        
        

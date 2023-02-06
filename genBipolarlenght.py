# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:57:56 2022

@author: Robin
"""


import numpy as np


def genBipolarlenght(nInput,testlenght):
  
    
    letters = 2
    testlenbipolar,genInput,genOutput = {},{},{}
    nSeq =100
    
    # transform each one encoding test set on bipolar test set
    for h in range(0,6):
        
        
        letters = letters+1
        nItem = (2*letters)+1
        
        
        input2 = np.zeros((nSeq,nItem,nInput))
        targets2 = np.zeros((nSeq,nItem,nInput))
        
        for i in range(0,nSeq):
            
        
            inp = testlenght[0][h][i]
            tar = testlenght[1][h][i]
                
            indInp = np.where(inp[0:letters]==1)
            indTar = np.where(tar[0:letters*2]==1)
                
            inp = inp-1
            tar = tar-1
                          
            inp[indInp[0],indInp[1]] = 1
            tar[indTar[0],indTar[1]] = 1
                
            inp[np.where(inp==0)] = 1
            tar[np.where(tar==0)] = 1
    
            for j in range(0,letters*2):
                try:
                        
                    inp[indInp[0][j],indInp[1][j]+1] = 1
                    inp[indInp[0][j],indInp[1][j]-1] = 1
                        
                    tar[indTar[0][j],indTar[1][j]+1] = 1
                    tar[indTar[0][j],indTar[1][j]-1] = 1
                    
                except Exception:
                    pass
                    
                tar[letters:letters*2,:] = tar[0:letters,:] 
                
            
            
            input2[i] = inp
            targets2[i] = tar
                
        genInput[h] = input2
        genOutput[h] = targets2
        
    testlenbipolar[0] = genInput 
    testlenbipolar[1] = genOutput

    return testlenbipolar
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:50:13 2022

@author: Robin
"""
import numpy as np

import random as rand

def simil(nSeq,nItem,nInput,gen,letters):

    gen3 = {}    

    inp = np.zeros((nSeq,nItem,nInput))
    inpTest = np.zeros((nSeq,nItem,nInput))
    
    inpShrink = np.zeros((nSeq,nItem,nInput))
    outShrink = np.zeros((nSeq,nItem,nInput))
    
    inpTestShrink = np.zeros((nSeq,nItem,nInput))
    outTestShrink = np.zeros((nSeq,nItem,nInput))
    
    
    for i in range(0,nSeq):
        inp[i] = gen[0][i]
        inpTest[i] = gen[2][i]
    
    for j in range(0,nSeq):
        
        indic = np.where(inp[j][0:letters]==1)[1]
        indic[1:letters*2:2] = indic[1:letters*2:2]-10
        
        indicTest = np.where(inpTest[j][0:letters]==1)[1]
        indicTest[1:letters*2:2] = indicTest[1:letters*2:2]-10
        
        
        
        shrink = np.int32(np.round(indic*0.5))
        shrink1 = shrink[1:letters*2:2]
        shrink2 = shrink[0:letters*2:2]+10
        
        shrinkTest = np.int32(np.round(indicTest*0.5+2))
        shrink1Test = shrinkTest[1:letters*2:2]
        shrink2Test = shrinkTest[0:letters*2:2]+10
        
        
        for k in range(0,letters):
            
            inpTestShrink[j][k,shrink1Test[k]] = 1
            inpTestShrink[j][k,shrink2Test[k]] = 1
            
            inpShrink[j][k,shrink1[k]] = 1
            inpShrink[j][k,shrink2[k]] = 1
        
        inpShrink[j][letters:nItem,nInput-1] = 1
        
        inpTestShrink[j][letters:nItem,nInput-1] = 1
        
        outShrink[j][0:letters] = inpShrink[j][0:letters]
        outShrink[j][letters:nItem-1] = inpShrink[j][0:letters]
        outShrink[j][nItem-1,nInput-1] = 1
        
        outTestShrink[j][0:letters] = inpTestShrink[j][0:letters]
        outTestShrink[j][letters:nItem-1] = inpTestShrink[j][0:letters]
        outTestShrink[j][nItem-1,nInput-1] = 1
        
    
    gen3[0] = inpShrink
    gen3[1] = outShrink
    gen3[2] = inpTestShrink 
    gen3[3] = outTestShrink

    return gen3
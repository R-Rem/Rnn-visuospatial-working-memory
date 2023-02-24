
import numpy as np


def genBipolar(nInput,inputs,targets,testInput,testOutput,nItem,letters,nSeq):
    
    
    #Pre-allocate memory
    input2 = np.zeros((nSeq,nItem,nInput))
    targets2 = np.zeros((nSeq,nItem,nInput))

    testInput2 = np.zeros((nSeq,nItem,nInput))
    testOutput2 = np.zeros((nSeq,nItem,nInput))
    
    
    gen2 ={}
    
    for i in range(0,nSeq):
        
        inp = inputs[i]
        tar = targets[i]
        testIn = testInput[i]
        testOut = testOutput[i]
        
        indInp = np.where(inputs[i][0:letters]==1)
        indTar = np.where(targets[i][0:letters*2]==1)
        indtestIn = np.where(testIn[0:letters]==1)
        indTestOut = np.where(testOut[0:letters*2]==1)
    
        inp = inp-1
        tar = tar-1
        testIn = testIn-1
        testOut = testOut-1
            
        inp[indInp[0],indInp[1]] = 1
        tar[indTar[0],indTar[1]] = 1
        testIn[indtestIn[0],indtestIn[1]] = 1
        testOut[indTestOut[0],indTestOut[1]] = 1
            
        inp[np.where(inp==0)] = 1
        tar[np.where(tar==0)] = 1
        testIn[np.where(testIn==0)] = 1
        testOut[np.where(testOut==0)] = 1
            

        for j in range(0,letters*2):
            
            try:
                    
                inp[indInp[0][j],indInp[1][j]+1] = 1
                inp[indInp[0][j],indInp[1][j]-1] = 1
                
                tar[indTar[0][j],indTar[1][j]+1] = 1
                tar[indTar[0][j],indTar[1][j]-1] = 1
                
                testIn[indtestIn[0][j],indtestIn[1][j]+1] = 1
                testIn[indtestIn[0][j],indtestIn[1][j]-1] = 1
                
                testOut[indTestOut[0][j],indTestOut[1][j]+1] = 1
                testOut[indTestOut[0][j],indTestOut[1][j]-1] = 1
                
            except Exception:
                pass
                
            tar[letters:letters*2,:] = tar[0:letters,:]
            testOut[letters:letters*2,:] = testOut[0:letters,:]
        
        input2[i] = inp
        targets2[i] = tar
        
        testInput2[i] = testIn
        testOutput2[i] = testOut

    gen2[0] = input2
    gen2[1] = targets2
    gen2[2] = testInput2
    gen2[3] = testOutput2

    return gen2


###############################################

To launch the simulations, just choose :

- List length effect: simulationListLenght
- PrimRecency effect: simulationPrimRecence
- Transposition gradient effect: simulationTranspo
- Similarity effect: simulationSimilarity

###############################################

Training algorithm :

- forbackwardfull: to train the network during the training phase
- forbackward_test : to train the network during the test phase (5 epochs)

###############################################

Algorithm for the test phase:

- testBipolar: for the simulation of the primacy/recency, transposition and similarity effect
- testListLenght : for the simulation of the list length effect

################################################

Algorithm to generate the stimuli :

- genStim: generates the set of random sets (training/test) in one encoding
- genBipolar : transforms the sets into bipolar encoding
- listlenght: generates each test set for each different length
- genBipolarlenght : transforms the sets into bipolar encoding
- genSimDis: generate similar and dissimilar stimuli randomized in a set
- similar: generates similar stimuli

##############################################
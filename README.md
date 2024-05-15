# Machine Learning Based Prediction of Polaron-Vacancy Patterns on the TiO$_2$(110) Surface - Data and Source Code

All data and source code to reproduce the results are contained in this repository and are structured as follows:
- ./dataset/ 
    - pristine POSCARs of the 6x4 and 12x2 cell 
    - polaronic and defect configuartions as extracted from the performed DFT calculations in .pkl files 
    - an example for reading and processing the data from the pickle file into ML representations is given in preprocess.py
- ./params/ 
    - pretrained ML-model weights for the 6x4 cell 
    - pretrained ML-model weights for the combined 12x2 and 6x4 cell 
- ./configurational_ML/ 
    - all python source code to perform the preprocessing, ML-model training and simulated annealing
    - ./preprocess.py generates descriptors and performs data augmentation
    - ./train.py performs gradient based optimization of the model weights
    - ./search.py reads in optimized weights and optimizes a large area model of the surface
- ./env.yml contains the used conda environment for generating the results

The three provided Feature Selection Datasets are:


Breast-w: From https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original  
With 9 features/columns
Letter-r from https://archive.ics.uci.edu/dataset/59/letter+recognition  
With 16 features/columns
Credit-a from https://archive.ics.uci.edu/dataset/27/credit+approval  
	With 15 features/columns
Read the accuracy aspect of the HDF5 files. Each row is the fitness function for the decimal representation that corresponds that you have to convert to binary to know which features/columns are active in the given instance. You’re supposed to calculate the mean of all columns for every row as that instance’s fitness function. The HDF5 files start with index 1, as no active  features for training of a machine learning model is not interesting for their use case. 


The synthetic problem is a triangle problem. You should use n = 16, m = 1, and s = 4, as parameters. Info on the triangle problem is found in the project description, at this link: https://dl.acm.org/doi/epdf/10.1145/3638530.3654156 or during the project lecture.

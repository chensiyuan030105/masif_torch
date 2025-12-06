### data_preparation

Contains the protocol followed to prepare protein structures for MaSIF, in order of execution:

+ *00-pdb_download.py*: Download pdb file from Protein DataBank 

+ *01-pdb_extract_and_triangulate.py*: Extract the PDB chains analyzed and triangulate them

+ *04-masif_precompute.py*: Decompose proteins into patches for input into the neural network.

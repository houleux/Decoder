Carefully look at RELDEC folder. It logs continuous baselines and results on some algorithms for layered BP decoding of LDPC matrices. 

We have added two new matrices: H_Mackay_96_48.csv and H_AB_3_7_196.csv


Now we want to evaluate all methods on these 2 matrices and save the data in continuous runs like exisiting methods. 
- Keep z = 6 for @file:H_Mackay_96_48.csv and z= 7 for @file:H_AB_3_7_196.csv 
- Certain algorithms use a fixed z for now. make sure you create generalized versions of these first (dont change exiting algos but add new ones)
- Note that there exists another AB code (@file:H_AB_LDPC_500.csv  ) Many present files refer to it simply as AB. ENSURE THAT YOU DO NOT GET CONFUSED BY THIS. The new AB matrix is different and unless explicitly mentioned AB_3_7, assume it is for the older matrix
- run training first and then evaluation. Ensure both have checkpint are stored like continuous runs 
- when run is successfull plot results in a notebook (eval_2021_matrices.ipynb)
- first do a smoke run and then a full run
- there may be references to a SLURM/HPC/ADA cluster. No need to use it unless explicitly specified by user
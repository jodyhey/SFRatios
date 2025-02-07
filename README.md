# SF_Ratios Documentation
by Jody Hey, 2025

The SF_Ratios archive contains materials associated with the manuscript "Isolating selective from non-selective forces using site frequency ratios" by Jody Hey and Vitor Pavinato. Included in this archive are the main scripts for estimating selection parameters, as well as other utility scripts for manipulating data, making figures and assessing performance. 

This archive is also a copy of a Visual Studio Code Workspace, including a launch.json file if anyone wants to use it. 

## Main Scripts
* SFRatios.py  - selection model fitting for ratios of Site Frequency Spectra, Selected/Neutral
* SFRatios_functions.py - various functions called by SFRatios.py and other scripts in this archive 

SFRatios.py has an option (-g) for more thorough optimization on scipy basinhopping and dualannealing. This is quite slow,  and some runs may take a day or so,  but it is often worth it. 

## Subfolder Contents
### ./Drosophila_SFS_pipeline
Summary and scripts of the pipeline for building the Drosophila data sets

### ./performance
Scripts and folders for assessing estimator performance.
* Estimation_on_WrightFisher_SF_simulations.py - runs ROC, Power and Chi^2 comparison analyses on data simulated under PRF.  
* Estimation_on_SFS_with_SLiM.py - does PRF-Ratio model fitting on data previously simulated using SLiM
* Simulate_SFS_with_SLiM.py - runs SLiM simulations using models and functions found in the *slim_work* folder
* Results_WrightFisher_SF_simulations - the default folder for output from Estimation_on_WrightFisher_SF_simulations.py.  Contains the results of ROC,Power and Chi^2 comparison analyses presented in the paper
* Results_SFS_with_SLiM - the default folder for output from Estimation_on_SFS_with_SLiM.py. Contains the results for various demographic models that were presented in the paper. 

### ./data 
Data files for North Caroline (DGRP2) and Zambia (DPGP3) samples. All files have the neutral SFS based on short introns first, followed by the selected SFS.  All SFSs begin with bin 0.  

### ./utilities
* get_SF_Ratio_output_summaries.py - a script that can read a bunch of output files from SF_Ratio.py and generate a .csv file with main results
* make_2Ns_distribution_plot.py - a script that can make a figure from a SF_Ratio.py output file
* SFS_modifications.py - has several utilities for handling SFSs
* twoDboxplot.py - called by Estimation_on_SFS_with_SLiM.py when run using a lognormal or gamma density.  Can be run as a standalone on an output file from Estimation_on_SFS_with_SLiM.py.
* compare_ratio_poissons_to_ratio_gaussians.py - simulate the ratio of two poisson random variables, and plot the histogram. Also plot the corresponding density of the ratio of two gaussians using ex (1) of Díaz-Francés, E. and F. J. Rubio (2013). "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables." Statistical Papers 54: 309-323.
* simulate_WF_SFSs_for_SF_Ratios.py - simulate a data set for SF_Ratios.py 
### ./slim_work
Contains folders and files used for generating simualted data sets with SLiM.  These are used by  ./performance/Estimation_on_SLiM_SFS_simulations.py  and ./performance/Simulate_SFS_with_SLiM.py.


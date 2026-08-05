# SF_Ratios Documentation
by Jody Hey, 2025

The SF_Ratios archive contains materials associated with the manuscript "Isolating selective from non-selective forces using site frequency ratios" by Jody Hey and Vitor Pavinato. Included in this archive are the main scripts for estimating selection parameters, as well as other utility scripts for manipulating data, making figures and assessing performance. 

This archive is also a copy of a Visual Studio Code Workspace, including a launch.json file if anyone wants to use it. 

## Main Scripts
* SFRatios.py  - selection model fitting for ratios of Site Frequency Spectra, Selected/Neutral
* SFRatios_functions.py - various functions called by SFRatios.py and other scripts in this archive 
* utilities/run_multiple_SFRatios_jobs.py - runs SFRatios.py on many SFS pairs in parallel

SFRatios.py has an option (-g) for more thorough optimization on scipy basinhopping and dualannealing. This is quite slow,  and some runs may take a day or so,  but it is often worth it. 

## SFRatios.py

`SFRatios.py` fits a selection model to the ratio of a selected SFS to a neutral SFS.

Example:

```bash
python SFRatios.py -a data/example_SFS.txt -f unfolded -d fixed2Ns -r results -p example
```

### Input file format

The input file has four lines:

1. Header text
2. Neutral SFS, beginning with bin 0
3. Blank or ignored line
4. Selected SFS, beginning with bin 0

The neutral and selected SFSs must have the same number of bins. Bin 0 is read but ignored in the calculations.

### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `-a SFSFILENAME` | yes | none | Input SFS file. |
| `-f FOLDSTATUS` | yes | none | SFS folding mode. Use `unfolded` for an unfolded SFS, `foldit` to fold an unfolded SFS before fitting, or `isfolded` when the input SFS is already folded. |
| `-d DENSITYOF2NS` | no | `fixed2Ns` | Selection model. Current public options are `fixed2Ns`, `gamma`, `lognormal`, and `normal`. |
| `-p POPLABEL` | no | empty string | Prefix used when constructing the output filename. |
| `-r OUTDIR` | no | current directory | Output directory. Created if it does not exist. |
| `-c FIX_THETA_RATIO` | no | estimated | Fix the mutation-rate ratio `thetaS/thetaN` to this value instead of estimating it. |
| `-g` | no | off | Run additional global optimization using basinhopping and dual annealing. This can be much slower but may find a better optimum. |
| `-m SETMAX2NS` | no | `0` | Fixed maximum 2Ns value for `gamma` and `lognormal` models. |
| `-t` | no | off | Estimate the maximum 2Ns value for `gamma` or `lognormal` instead of using `-m`. |
| `-M MAXI` | no | all bins | Maximum bin index to include in the likelihood. |
| `-x` | no | off | If the output file already exists, stop instead of making a numbered filename. |
| `-z` | no | off | Include an estimated point mass at 2Ns = 0. |
| `-Q LOW HIGH` | no | automatic | Constrain the estimated `thetaS/thetaN` ratio to the range `LOW` to `HIGH`. |
| `-v MINDENOM` | no | off | Set a minimum neutral-SFS denominator count. When used with `-w`, bins after the first sparse bin are summed into subranges until each subrange has at least this neutral count. |
| `-w` | no | off | Sum sparse high-frequency bins into subranges. Requires `-v`. Useful for sparse SFS tails, especially for continuous `gamma` and `lognormal` models. |

### Output

The output filename is built from the population label, model, sample size, and selected options, and ends in `_estimates.out`. The file records the command line, fitted likelihood/AIC, parameter estimates with confidence intervals, and a table comparing observed and fitted SFS ratios.

## utilities/run_multiple_SFRatios_jobs.py

`utilities/run_multiple_SFRatios_jobs.py` runs `SFRatios.py` repeatedly on a file containing many SFS pairs. Most options are passed through to `SFRatios.py`; the batch script adds parallelization and controls how each SFS pair is interpreted.

Example:

```bash
python utilities/run_multiple_SFRatios_jobs.py \
  -a many_SFS_pairs.txt \
  -f unfolded \
  -d fixed2Ns \
  -r results \
  -p run1 \
  -j 10
```

### Input file format

The input file is made of repeated four-line blocks with no blank lines between blocks:

1. Title for the first SFS
2. First SFS values
3. Title for the second SFS
4. Second SFS values

By default, the first SFS in each block is treated as selected and the second as neutral. Use `--first-sfs neutral` if the first SFS is neutral.

For each block, the batch script writes a temporary single-pair input file in the format expected by `SFRatios.py`, then runs one SFRatios job.

### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `-a INPUTFILE` | yes | none | Input file containing multiple four-line SFS-pair blocks. |
| `-f FOLDSTATUS` | yes | none | Passed to `SFRatios.py`. Use `unfolded`, `foldit`, or `isfolded`. |
| `-p POPLABEL` | yes | none | Base output prefix. The selected-SFS title is sanitized and appended for each job. |
| `-j PARALLEL_JOBS` | no | `1` | Number of SFRatios jobs to run at the same time. |
| `--first-sfs {selected,neutral}` | no | `selected` | Meaning of the first SFS in each input block. |
| `--debug` | no | off | Print detailed command and file-location information for troubleshooting. |
| `-d DENSITYOF2NS` | no | `SFRatios.py` default | Passed to `SFRatios.py`; one of `fixed2Ns`, `gamma`, `lognormal`, or `normal`. |
| `-r RESULTSDIR` | no | current directory | Output directory for all SFRatios result files. Relative paths are converted to absolute paths before launching jobs. |
| `-c FIX_THETA_RATIO` | no | estimated | Passed to `SFRatios.py`; fixes `thetaS/thetaN`. |
| `-g` | no | off | Passed to `SFRatios.py`; run additional global optimization. |
| `-m SETMAX2NS` | no | `SFRatios.py` default | Passed to `SFRatios.py`; fixed maximum 2Ns value for `gamma` and `lognormal`. |
| `-t` | no | off | Passed to `SFRatios.py`; estimate maximum 2Ns for `gamma` or `lognormal`. |
| `-M MAXI` | no | all bins | Passed to `SFRatios.py`; maximum bin index to use. |
| `-x` | no | off | Passed to `SFRatios.py`; stop if an output file already exists. |
| `-z` | no | off | Passed to `SFRatios.py`; estimate a point mass at 2Ns = 0. |
| `-Q LOW HIGH` | no | automatic | Passed to `SFRatios.py`; range for `thetaS/thetaN`. |
| `-v MINDENOM` | no | off | Passed to `SFRatios.py`; minimum denominator for sparse-bin handling. |
| `-w` | no | off | Passed to `SFRatios.py`; sum sparse bins into subranges. Requires `-v`. |

The batch script prints each launched command and writes the normal `SFRatios.py` `_estimates.out` files into the requested results directory.

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

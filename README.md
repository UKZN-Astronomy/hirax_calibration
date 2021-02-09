# hirax_corrcal
CorrCal development for HIRAX

The Code_without_draco folder contains scripts to calibrate using LogCal and CorrCal. The visibilities and covariance matrices are generated without using draco/driftscan. The CorrCal_LogCal_Notebook_perturbed_array.ipynb notebook compares gain amplitude errors obtained with LogCal and CorrCal. The Calibration_vs_Systematic_Error_Notebook.ipynb notebook compares calibration error and systematic error for different perturbed arrays using CorrCal.

The Code_parallelisation_hippo folder has a bash script that can be used to speed up code by running multiple noise realisations in parallel. 

The draco_corrcal_pipeline folder details work that has been done thus far to incorporate Radiocosmology packages (draco, driftscan, cora, etc) and CorrCal in an automated calibration pipeline.   

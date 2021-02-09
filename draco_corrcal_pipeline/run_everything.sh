#! /usr/bin/env bash

cora-makesky --nside 256 --freq 600.0 600.78125 2 --freq-mode edge --pol zero --filename pointsource_brightsources.h5 pointsource_bright --minflux 1 --maxflux 2 #1

cora-makesky --nside 256 --freq 600.0 600.78125 2 --freq-mode edge --pol zero --filename pointsource_allsources.h5 pointsource --maxflux 2 #2

drift-makeproducts run prod_params.yaml #3

caput-pipeline run config_draco.yaml #4

caput-pipeline run config_draco_brightsources.yaml #5

python draco_corrcal_file.py #6

caput-pipeline run config_draco_vis_after_calibration.yaml #7


# This bash file is used to calculate the relative visibilities for a perturbed array, whose recovered gains were calculated using CorrCal. The commands in the file are described as follows:
 
#1 Generates a point source map using ONLY the bright point sources in the catalogue provided in cora. Here, we chose a flux range between 1-2 Jy. Note we consider only the zero polarisation.

#2 Generates a point source map for ALL sources - unresolved sources below 0.1 Jy; simulated sources between 0.1 and 1 Jy; and real, bright sources from the catalogue between 1 and 2 Jy. 

#3 Generate beam transfer matrices, kl transform, etc.

#4 Generate timestream data with random gains and Gaussian noise. The noise includes a seed parameter. This is the visibility data that is input into CorrCal. The random gains are also input as the true/simulated gains. The parameters used in the Gaussian noise are also used to get the diagonal noise cov matrix in CorrCal.

#5 Generate visibility timestreams for each bright source (in the map generated in step 1) without noise or gains. Used to obtain the source vector in CorrCal.

#6 Python file to run CorrCal. The output is a gains_error file which contains the complex gain errors. These are in the shape (nfreq, 2*Ndish, ntime), as required for draco.

#7 Load the gain errors and apply these to the true visibilities, and thereafter add noise (with the same seed parameter as in #4). 


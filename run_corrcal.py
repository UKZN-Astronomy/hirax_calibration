''' This is a file that I ran on hippo when testing out the use of SLURM job arrays to speed up code'''


import numpy as np
import random
import h5py,time, matplotlib.pyplot as plt
from scipy.optimize import fmin_cg, minimize
from drift.core import manager
import corrcal
import sys
sys.path.insert(0,'/data/zahrakad/PIPELINE_3')
from hirax_transfer import core
import scipy as sp
from cora.util import hputil
from astropy.stats import gaussian_fwhm_to_sigma
from hirax_transfer.beams import separations
import healpy as hp
from cora.core import skysim
from cora.foreground import gaussianfg, galaxy
from cora.util import coord
from drift.core import visibility
sys.path.insert(0,'/data/zahrakad/hirax_tools/')
from hirax_tools import array_config
from LogCal_code import Bls_counts
from cora.foreground import poisson as ps
import concurrent.futures
from Functions import *


m = manager.ProductManager.from_config('prod_params_custom.yaml')

'''Load the arrays that will be used for the simulations'''

src_total = np.load('src_total_6Aug_flux_1Jy_10Jy.npy')
Vis_bright_sources = np.load('Vis_bright_sources_6Aug_flux_1Jy_10Jy.npy')


Vis_poisson = np.load('Vis_poisson_6Aug.npy')


Vis_total = Vis_poisson + Vis_bright_sources


vecs = np.load('vecs_6Aug.npy')

src_zeros = np.zeros(src_total.shape[1]) # used if the source positions are unknown


v1=np.zeros(src_total.shape[1])
v1[0::2]=1
v2=np.zeros(src_total.shape[1])
v2[1::2]=1
vecs_redundant = np.vstack([v1,v2])*1.e3 # used for the redundant case in CorrCal

''' Compute the recovered gains for each run using CorrCal'''

start = time.time()
rec_gains = fit_gains_base(m, sys.argv[1], src_total, vecs, Vis_total)
finish = time.time()
print (finish - start, 'time taken to finish')


import numpy as np
import random
import h5py,time, matplotlib.pyplot as plt
from scipy.optimize import fmin_cg, minimize
from drift.core import manager
import corrcal
import sys
sys.path.insert(0,'/home/zahra/PIPELINE')
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
sys.path.insert(0,'/home/zahra/hirax_tools/')
from hirax_tools import array_config
from LogCal_code import Bls_counts, Logcal_solutions
from cora.foreground import poisson as ps
from scipy.special import j1 as bessel_j1


def baseline_length_direction(m, x_scat, y_scat):
    '''
    Returns the baseline lengths and directions

    Parameters
    ----------
    m : Prod params file
        Config file from driftscan runs using NOMINAL dish positions
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position

    Returns
    -------
    bl_xy_lengths_mat : ndarray
        The x and y lengths of baselines  (Nbls,2), organised according to baseline redundancy
    bl_abs_mag_length_redundancy_ordered : ndarray
        The absolute magnitude of baselines, organised according to baseline redundancy (Nbls,)
    bl_descrip_list : str
        A description of the nominal baseline length and orientation along the EW and NS directions
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    bl_arr_dish_indices, _, _, _, _ = Bls_counts(m)
    x = t.feedpositions[:,0][:Ndish]
    y = t.feedpositions[:,1][:Ndish]
    x_scat_array = np.random.normal(0, x_scat, Ndish)
    y_scat_array = np.random.normal(0, y_scat, Ndish) # Save and load x_scat_array and y_scat_array if you want to ensure the same x and y
                                                    # positions for perturbed array

    x = x + x_scat_array    #if you want the perturbed array to remain exactly the same each time you run the code, save and load the
                            #x_scat_array and y_scat_array
    y = y + y_scat_array

    bl_xy_lengths_mat = np.zeros((Nbls,2))
    for i in range(len(bl_arr_dish_indices)):
        dish_0, dish_1 = bl_arr_dish_indices[i]
        bl_ind = [np.int(dish_0), np.int(dish_1)]
        x_sep = x[bl_ind[1]] - x[bl_ind[0]]
        bl_xy_lengths_mat[i,0] = x_sep
        y_sep = y[bl_ind[1]] - y[bl_ind[0]]
        bl_xy_lengths_mat[i,1] = y_sep
    bl_descrip_list = []
    for i in bl_xy_lengths_mat:
        x_len_rounded = np.int(np.round(i[0]))
        y_len_rounded = np.int(np.round(i[1]))
        bl_descrip = (str(x_len_rounded) + ' EW' + ', ' + str(y_len_rounded) + ' NS')
        bl_descrip_list = np.append(bl_descrip_list, bl_descrip)

    bl_abs_mag_length_redundancy_ordered = np.linalg.norm(bl_xy_lengths_mat, axis=1)
    return bl_xy_lengths_mat, bl_abs_mag_length_redundancy_ordered, bl_descrip_list


def Gauss_beam(m, freq, nside):
    '''
    Calculates the Gaussian beam model - This is Devin's code from hirax_transfer/hirax_transfer/beams.py
    We assume the same beam for each antenna

    Parameters
    ----------
    m : Prod params file - doesn't matter if this is the prod params file for the nominal or actual dish positions, we just use this to get zenith
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used

    Returns
    -------
    Beam : ndarray
        Beam model (npix,)
    '''
    t = m.telescope
    zenith = t.zenith
    wavelength = (3.e8)/(freq*10.**6)
    fwhm = 1.* wavelength/6.
    sigma_beam = gaussian_fwhm_to_sigma*fwhm
    angpos = hputil.ang_positions(nside)
    seps = separations(angpos, zenith)
    beam = np.exp(-seps**2/2/sigma_beam**2)
    return beam

# The Mu function follows the equation $\mu_n = \int S^{n-1} \frac{dN}{dS} dS$, with n=2 used for visibilities and n=3 for sky cov

def Mu(m, index):
    '''
    Calculates mu by assuming a differential source count, dN/dS, as per Santos et al, 2005

    Parameters
    ----------
    m : Prod params file - doesn't matter if this is the prod params file for the nominal or actual dish positions, we just use this to get Ndish
    index : int
        Input the index of source flux, S, depending on what you want to calculate:
        n = 2 for visibilities due to unresolved point sources following a Poisson distribution
        n = 3 for the covariance of the visibilities due to these point sources

    Returns
    -------
    mu_n : float
        A constant, computed as per the equation above
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)

    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish) #12 micro jansky sensitivity for full array, scaled for our size array, multiplied by 5 for
    # a 5 sigma detection
    alpha = 4000
    beta = 1.75
    mu_index = alpha/(index-beta) * (S_max**(index-beta))
    return mu_index

# Coded up equation 11 in Calibration_with_Pointsources.pdf 
def Visibilities_poisson(m, x_scat, y_scat, freq, nside):
    '''
    Calculates visibilities due to unresolved point sources below the S_{max} set by the
    telescope sensitivity. We assume sources follow a Poisson distribution, and do not account
    for source clustering.

    Parameters
    ----------
    m : Prod params file - doesn't matter if this is the prod params file for the nominal or actual dish positions, we just use this to get zenith
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used

    Returns
    -------
    vis_poisson : ndarray
        Visibilities of unresolved sources (Nbls,) in units [Jy]
    '''
    t = m.telescope
    zenith = t.zenith
    fringe = visibility.fringe
    angpos = hputil.ang_positions(nside)
    npix = hp.nside2npix(nside)
    beammodel = Gauss_beam(m, freq, nside)

    bl_xy_lengths_mat = baseline_length_direction(m, x_scat, y_scat)[0]
    wavelength = (3.e8)/(freq*10.**6)
    bl_arr = bl_xy_lengths_mat/wavelength

    bl_arr_dish_indices, _, _, _, _ = Bls_counts(m)

    point_source_vis = Mu(m, 2)
    pix_beam = np.where(beammodel>1e-10)[0] # Reduce the computation time by only considering pixels where the beam model is larger than 1e-10
    vis_poisson = np.array([])
    for i in range(len(bl_arr_dish_indices)):
        fringe_ind = fringe(angpos[pix_beam], zenith, bl_arr[i])
        vis_ind = 4 * np.pi/npix * point_source_vis * np.sum(beammodel[pix_beam]**2 * fringe_ind)
        vis_poisson = np.append(vis_poisson, vis_ind)
    return vis_poisson

# Coded up equation 14 in Calibration_with_Pointsources.pdf 
def Covariance_poisson(m, x_scat, y_scat, freq, nside):
    '''
    Covariance matrix computed using unresolved point sources following a Poisson distribution,
    and we compute the covariance at a single frequency, across different baselines.
    Covariance matrices are computed separately for each redundant block such that we do not
    compute a full (Nbls, Nbls) matrix. The redundant blocks are created for redundant baselines (i.e. baselines with the same length and direction)

    Parameters
    ----------
    m : Prod params file - doesn't matter if this is the prod params file for the nominal or actual dish positions, we just use this to get zenith
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used

    Returns
    -------
    vis_poisson : dict
        Separate n x n covariance matrices for each redundant block, with the matrix size
        depending on the baseline redundancy, in units [Jy^2]
    '''
    _, sum_counts, bl_counts, _, _ = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks
    t = m.telescope
    zenith = t.zenith

    angpos = hputil.ang_positions(nside)
    npix = hp.nside2npix(nside)
    point_source_cov = Mu(m, 3)

    bl_xy_lengths_mat = baseline_length_direction(m, x_scat, y_scat)[0]
    wavelength = (3.e8)/(freq*10.**6)
    bl_arr = bl_xy_lengths_mat/wavelength

    beammodel = Gauss_beam(m, freq, nside)
    pix_beam = np.where(beammodel>1e-10)[0] # Reduce the computation time by only considering pixels where the beam model is larger than 1e-10

    Cov_dic ={}
    for ubl_k in range(len(lims)-1):
        block_k = bl_counts[ubl_k]
        cov_k =np.zeros((block_k,block_k), dtype='complex')
        for bl_w in range(block_k):
            for bl_z in range(block_k):
                u_alph_bet = bl_arr[bl_w] - bl_arr[bl_z]
                fringes_ind = visibility.fringe(angpos[pix_beam], zenith, u_alph_bet)
                cov_k[bl_w][bl_z] = 4 * np.pi/npix * point_source_cov * np.sum(beammodel[pix_beam]**4*fringes_ind)
        Cov_dic[ubl_k]= cov_k
    return Cov_dic


def Vecs(m, x_scat, y_scat, freq, nside):
    '''
    Decomposes the covariance matrix into eigenvalues and eigenvectors, separately for each
    redundant block. This is the R vector that is input into CorrCal (if you are not using the
    redundant case in CorrCal). We set a threshold for the eigenvalues that are considered,
    which determines the number of vectors, nvec. Also R is separated into its real and
    imaginary components

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used

    Returns
    -------
    vecs : ndarray
        R vector (2*Nvec, 2*Nbls)
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)

    _, sum_counts, _, _, _ = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks

    Cov_dic = Covariance_poisson(m, x_scat, y_scat, freq, nside)
    thresh = 1.e-6

    ind_array=np.array([])
    for i in range(len(lims)-1):
        myeig, myvecs = np.linalg.eig(Cov_dic[i])
        ind = np.sum(myeig > thresh * myeig.max()) #Only eigenvalues with 10^{-6} times the max
        # eigenvalue are considered
        ind_array = np.append(ind_array, ind)

    nvec = np.int((ind_array).max())

    vecs = np.zeros((2*Nbls, 2*nvec))

    for i in range(len(lims)-1):
        myeig, myvecs = np.linalg.eig(Cov_dic[i])
        ind = myeig >thresh * myeig.max() # picking up an index
        myeig = myeig[ind] # pick max eigenvalue
        myvecs = myvecs[:,ind]

        for j in range(len(myeig)):
                myvecs[:,j]= myvecs[:,j]*np.sqrt(myeig[j])
                vecs[2*lims[i]:2*lims[i+1]:2, 2*j] = np.column_stack(myvecs[:,j].real)
                vecs[(2*lims[i]+1):(2*lims[i+1]+1):2, 2*j+1] = np.column_stack(myvecs[:,j].imag)
    #Note that every first row and first column contains real components,
    #every second row and column contains imaginary components
    vecs= vecs.T
    return vecs

# The next four functions are taken from CORA (based on Di Matteo et al 2002)- I modified this slightly to return values in units of Jy instead of K.

def source_count(flux):
    r"""Power law luminosity function."""

    gamma1 = 1.75
    gamma2 = 2.51
    S_0 = 0.88
    k1 = 1.52e3

    s = flux / S_0

    return k1 / (s**gamma1 + s**gamma2)

def spectral_realisation(flux, freq):
    r"""Power-law spectral function with Gaussian distributed index."""

    spectral_mean = -0.7
    spectral_width = 0.1
    spectral_pivot = 151.0

    ind = spectral_mean + spectral_width * np.random.standard_normal(flux.shape)

    return flux * (freq / spectral_pivot)**ind

def generate_population(flux_min, flux_max, area):
    r"""Create a set of point sources.
    Returns
    -------
    sources : ndarray
        The fluxes of the sources in the population.
    """

    rate = lambda s: flux_min*np.exp(s)*area*source_count(flux_min*np.exp(s))
    fluxes = flux_min * np.exp(ps.inhomogeneous_process_approx(np.log(flux_max/flux_min), rate))

    return fluxes

def getsky(freq, nside, flux_min, flux_max):
    """Simulate a map of point sources.
    Returns
    -------
    sky : ndarray [nfreq, npix]
        Map of the brightness temperature on the sky (in Jy).
    """

    npix = 12*nside**2

    nfreq = 1

    sky = np.zeros((nfreq, npix), dtype=np.float64)

    fluxes = generate_population(flux_min, flux_max, 4*np.pi)

    sr = spectral_realisation(fluxes[:,np.newaxis], freq)

    for i in range(sr.shape[0]):
        # Pick random pixel
        ix = int(np.random.rand() * npix)

        sky[:, ix] += sr[i,:]
    return sky


def Src_vector(m, x_scat, y_scat, freq, nside, map, thresh_max_vals_beam):
    '''
    Calculates the source vector, S, that is input into CorrCal (for known source positions),
    and contains the per visibility response to bright sources with known position.
    Vector S is then used to calculate the total visibility contribution from bright sources.
    Note that we have separated S into its real and imaginary but have not separated the
    visibilities into these components as yet

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used
    map : ndarray
        The map of bright point sources that you are using (npix,)
    thresh_max_vals_beam : float
        Lower limit for the product of the beam model and map of bright sources

    Returns
    -------
    Vis_bright_sources : ndarray
        Visibilities due to bright point sources (Nbls,) in units [Jy]
    src_total : ndarray
        Per visibility response to sources with known positions (2*Nsources, 2*Nbls)
        in units [Jy]
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)

    zenith = t.zenith
    fringe = visibility.fringe
    angpos = hputil.ang_positions(nside)
    beammodel = Gauss_beam(m, freq, nside)

    bl_xy_lengths_mat = baseline_length_direction(m, x_scat, y_scat)[0]
    wavelength = (3.e8)/(freq*10.**6)
    bl_arr = bl_xy_lengths_mat/wavelength

    bl_arr_dish_indices = Bls_counts(m)[0]

    map_vals = map
    vals_beam = map_vals*beammodel
    pix_valsbeam = np.where(vals_beam >thresh_max_vals_beam)[0] ## Reduce the computation time by only considering pixels where the product of
    #beam model and bright source is larger than some threshold - I typically set to 10^{-6}
    source_number = len(pix_valsbeam)
    print (source_number,'source number')
    src = np.zeros((len(bl_arr_dish_indices), source_number), dtype='complex')
    for n in range(source_number):
        non_zero_pix = pix_valsbeam[n]
        for i in range(len(bl_arr_dish_indices)):
            fringe_ind = fringe(angpos[non_zero_pix], zenith, bl_arr[i])
            vis_ind_source_ind = map_vals[non_zero_pix] * beammodel[non_zero_pix]**2 * fringe_ind
            src[i, n] = vis_ind_source_ind
    Vis_bright_sources = np.sum(src, axis=1) #Corresponds to equation 16 in Calibration_with_Pointsources.pdf 

    src_total = np.zeros((2*Nbls, 2*source_number))
    src_total[::2,::2] = src.real
    src_total[1::2,1::2] = src.imag
    src_total = src_total.T
    return Vis_bright_sources, src_total

def fit_gains(m, runs, src_array, vecs, visibilities):
    '''
    Computes the recovered gains. The simulated gains are assumed to be 1, so real components are amplitude and imaginary components are phase

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    runs : int
        Number of noise realisations you want to do (500 runs is sufficient)
    src_array : ndarray
        The source vector, S (2*Nsources, 2*Nbls) - The two options for the vector
        depend on whether or not source positions are known.
    vecs : ndarray
        The covariance vector, R (2*Nvec, 2*Nbls) - The two options for the vector
        depend on whether or not you are using the redundant case.
    visibilities : ndarray
        Noiseless visibilities (Nbls,)

    Returns
    -------
    recovered gains : ndarray
        Recovered gains that have alternate real (amplitude) and imaginary (phase) components (2*Ndish)
    '''

    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)

    bl_arr_dish_indices, sum_counts, bl_counts, _, _ = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks

    ant1 = bl_arr_dish_indices[:,0].astype(int)
    ant2 = bl_arr_dish_indices[:,1].astype(int)

    gain = np.ones(Ndish)
    sim_gains = np.ones(2*Ndish)
    sim_gains[::2] = gain.real
    sim_gains[1::2] = gain.imag

    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish)
    sigma = S_max*0.01 # we include a small amount of per visibility noise
    diag = sigma**2*np.ones(2*Nbls)
    random_gain = np.random.normal(0,1.e-3, 2*Ndish) #initial guess input into corrcal
    gvec = sim_gains + random_gain

    get_chisq = corrcal.get_chisq
    get_gradient = corrcal.get_gradient
    mat = corrcal.Sparse2Level(diag,vecs,src_array,2*lims)

    rec_gains = np.zeros((runs,Ndish*2))
    for ind_run in range(runs):
        print (ind_run)
        Noise_array = np.random.normal(0, sigma, 2*Nbls)
        data = np.zeros(2*visibilities.size)
        data[0::2] = visibilities.real + Noise_array[::2]
        data[1::2] = visibilities.imag + Noise_array[1::2]
        fac=1.;
        normfac=1.
        results = fmin_cg(get_chisq, gvec*fac, get_gradient,(data,mat,ant1,ant2,fac,normfac))
        fit_gains_run = results/fac
        rec_gains[ind_run,:] = fit_gains_run
        print (fit_gains_run[::2])
    return rec_gains #Every second value is recovered gain amplitudes, i.e. rec_gains[::2]


def fit_gains_base(m, run, src_array, vecs, visibilities):
    '''
    This is slightly different from the above function in that this function doesn't contain a for loop over the runs.

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    run : int
        Input the integer corresponding to the run number so that the recovered gains can be saved for each run
    src_array : ndarray
        The source vector, S (2*Nsources, 2*Nbls)
    vecs : ndarray
        The covariance vector, R (2*Nvec, 2*Nbls)
    visibilities : ndarray
        Noiseless visibilities (Nbls,)

    Returns
    -------
    recovered gains : ndarray
        Recovered gains separated in real (amplitude) and imaginary (phase) components (2*Ndish)
    '''

    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)

    bl_arr_dish_indices, sum_counts, bl_counts, _, _ = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks

    ant1 = bl_arr_dish_indices[:,0].astype(int)
    ant2 = bl_arr_dish_indices[:,1].astype(int)

    gain = np.ones(Ndish)
    sim_gains = np.ones(2*Ndish)
    sim_gains[::2] = gain.real
    sim_gains[1::2] = gain.imag

    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish)
    sigma = S_max*0.01 # we include a small amount of per visibility noise
    diag = sigma**2*np.ones(2*Nbls)
    random_gain = np.random.normal(0,1.e-3, 2*Ndish) #initial guess input into corrcal
    gvec = sim_gains + random_gain

    get_chisq = corrcal.get_chisq
    get_gradient = corrcal.get_gradient
    mat = corrcal.Sparse2Level(diag,vecs,src_array,2*lims)

    rec_gains = np.zeros(Ndish*2)
    data = np.zeros(2*visibilities.size)

    Noise_array = np.random.normal(0, sigma, 2*Nbls)
    data[0::2] = visibilities.real + Noise_array[::2]
    data[1::2] = visibilities.imag + Noise_array[1::2]

    fac=1.;
    normfac=1.
    results = fmin_cg(get_chisq, gvec*fac, get_gradient,(data,mat,ant1,ant2,fac,normfac))
    fit_gains_run = results/fac
    rec_gains = fit_gains_run
    #np.save('/home/zahra/hirax_corrcal/gain_amps_' + str(run), fit_gains_run[::2])
    print (fit_gains_run[::2], 'rec gain amps')
    return rec_gains

def hist_rel_err_mean_std(rec_gains_amp_or_phase, sim_gains_amp_or_phase):
    '''
    Histogram of mean and standard deviation of the amplitude/phase relative error
    for a number of noise realisations. The error is computed by taking the recovered gains and
    subtracting out the true gains.

    Parameters
    ----------
   rec_gains_amp_or_phase  : ndarray # - specify shape here
       The recovered gains for either the amplitude or phase
    sim_gains_amp_or_phase : ndarray
       The corresponding true gains for either the amplitude or phase

    Returns
    -------
    rec_gains_std : ndarray
        Standard deviation of relative errors (Nruns,)
    rec_gains_mean : ndarray
        Mean of relative errors (Nruns,)
    '''
    rel_error = np.abs(rec_gains_amp_or_phase - sim_gains_amp_or_phase)
    rec_gains_mean = np.mean(rel_error, axis=1) #shape is number of runs
    rec_gains_std = np.std(rel_error, axis=1, ddof=1)
    return rec_gains_std, rec_gains_mean


def Measured_vis(sigma, visibilities): # I use this for LogCal
    '''
    Calculates visibilities with noise

    Parameters
    ----------
    sigma : float
        The detector noise
    visibilities :
        Noiseless visibilities (Nbls,)

    Returns
    -------
    Measured visibilities : ndarray
        Visibilities with noise (Nbls,) in units [Jy]
    '''
    Nbls = visibilities.shape[0]
    mu = 0
    N_real = np.random.normal(mu, sigma, Nbls)
    N_imag = np.random.normal(mu, sigma, Nbls)
    N_comp = np.array([])
    for i in range(len(N_real)):
        N_comp = np.append(N_comp,complex(N_real[i],N_imag[i]))
    meas_vis = visibilities + N_comp
    return meas_vis

def recovered_gains(m, runs, visibilities): # I use this for LogCal
    '''
    Recovered gain amplitudes computed. The phase component is not included here

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    run : int
        Input the integer corresponding to the run number so that the recovered gains can be computed for each run
    visibilities :
        Visibilities with noise (Nbls,)

    Returns
    -------
    rec_gains : ndarray
        Recovered gain amplitudes (Ndish,)
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)

    gain = np.ones(Ndish)
    sim_gains = np.ones(2*Ndish)
    sim_gains[::2] = gain.real
    sim_gains[1::2] = gain.imag


    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish) #12 micro jansky sensitivity for full array, scaled for our size array, multiplied by 5
    # for 5 sigma detection

    sigma = S_max*0.01 # - same sigma used for CorrCal
    rec_gains = np.zeros((runs,Ndish))
    for ind_run in range(runs):
        print (ind_run)
        meas_vis = Measured_vis(sigma, visibilities)
        rec_gains[ind_run,:] = Logcal_solutions(m,visibilities,visibilities, sim_gains[::2], meas_vis, sigma)[0][:Ndish]
        print (np.exp(rec_gains[ind_run,:]))
    return rec_gains

# The next three functions were used to do the plots for Calibration error vs Systematic error

def Vis_matrix_each_bl(m, visibilities, rec_gains_full_array):
    '''
    Computes corrected visibilities and visibility errors

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    visibilities : ndarray
        True visibilities (Nbls,)
    rec_gains_full_array : ndarray
        Alternating real and imaginary gain components (2*Ndish)
    Returns
    -------
    Vis_error_abs_array : ndarray
        Visibility error (Nbls,)
    Vis_recovered_array : ndarray
        Recovered visibilities (Nbls,)
    '''

    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    bl_arr_dish_indices = Bls_counts(m)[0]
    rec_gains = rec_gains_full_array[::2] + 1j * rec_gains_full_array[1::2]
    # Get visibilities with gains, by multiplying vis with gains
    Vis_error_abs_array = np.array([])
    Vis_recovered_array = np.array([])
    for i in range(len(visibilities)):
        dish_1 = np.int(bl_arr_dish_indices[i][0])
        dish_2 = np.int(bl_arr_dish_indices[i][1])
        Vis_true = visibilities[i]
        Vis_recovered = Vis_true * rec_gains[dish_1] * np.conj(rec_gains[dish_2])
        Vis_recovered_array = np.append(Vis_recovered_array, Vis_recovered)
        Vis_error_abs = np.abs(Vis_recovered - Vis_true)/np.abs(Vis_true)
        Vis_error_abs_array = np.append(Vis_error_abs_array, Vis_error_abs)
    return Vis_error_abs_array, Vis_recovered_array

def Median_range_vis_plots(m, Vis_error_input):
    '''
    Separates visibility errors into different blocks. These blocks are not the same as the redundant blocks used in the Covariance_poisson and
    Vecs functions. Here, all baselines with the same magnitude (regardless of direction) are grouped in the same block.
    Parameters
    ----------
    m : Prod params file - for NOMINAL dish positions
    Vis_error_input : ndarray
        Visibility errors (Nbls,)

    Returns
    -------
    median_vis_abs_err : ndarray
        Median visibility error in each block (Nblock,)
    bl_lengths_nominal : ndarray
        Nominal baseline length of each block (Nblock,)
    min_arr : ndarray
        Minimum visibility error in each block (Nblock,)
    max_arr : ndarray
        Maximum visibility error in each block (Nblock,)
    counts_bl_lengths_nominal : ndarray
        Number of baselines in each with the length given in bl_lengths_nominal (Nblock,)
    '''

    bl_arr_dish_indices, sum_counts, bl_counts, bl_abs_mag_lengths_nominal,_ = Bls_counts(m)
    bl_lengths_nominal, _, counts_bl_lengths_nominal = np.unique(bl_abs_mag_lengths_nominal, return_counts = True, return_index = True)
    full_ind_array = np.array([])
    for i in bl_lengths_nominal:
        ind_single_bl_length = np.where(bl_abs_mag_lengths_nominal==i)[0]
        full_ind_array = np.append(full_ind_array, ind_single_bl_length)
    full_ind_array = full_ind_array.astype(int)
    Vis_error_abs_increasing_bl_mag = Vis_error_input[full_ind_array]
    sum_counts_bl_lengths_nominal = np.cumsum(counts_bl_lengths_nominal)
    Vis_error_nominal_bl_lengths = np.split(Vis_error_abs_increasing_bl_mag, sum_counts_bl_lengths_nominal)

    median_vis_abs_err = np.array([])
    min_arr = np.array([])
    max_arr = np.array([])
    for i in Vis_error_nominal_bl_lengths:
        if len(i)>0:
            median_vis = np.median(i)
            median_vis_abs_err = np.append(median_vis_abs_err, median_vis)
            min_vis, max_vis = np.min(i), np.max(i)
            min_arr = np.append(min_arr, min_vis)
            max_arr = np.append(max_arr, max_vis)
        else:
            pass
    return median_vis_abs_err, bl_lengths_nominal, min_arr, max_arr, counts_bl_lengths_nominal

def Get_median_and_range(m, true_vis_pert, pert_level, median_colour, range_colour, Calibration_error=True, pert_gain_rec=None, true_vis_unpert=None, length=True, savefig=False,fig_name=None):
    '''
    Used to plot calibration error and systematic error as a function of the baseline length or baseline redundancy

    Parameters
    ----------
    m : Prod params file - for the nominal or actual dish positions
    true_vis_pert: ndarray
        True visibilities for the perturbed array (Nbls,)
    pert_level: float
        Single value for the level of perturbation in the dish positions
    median_colour: str
        colour of the dots that signify the medians of the visibility errors in each block
    range_colour: str
        colour of the bars that signify the full range of the visibility errors in each block
    Calibration_error: boolean
        True if the plot is for calibration error and false if it is for systematic error
    pert_gain_rec: ndarray
        Alternating real and imaginary gain components (2*Ndish) - input this for calibration error
    true_vis_unpert: ndarray
        True visibilities for the unperturbed array (Nbls,) - input this for systematic error
    length: boolean
        True if the x axis has lengths of baselines, false if x axis has baseline length redundancies
    savefig: boolean
        True to save the figure
    fig_name: str
        Name for the figure if you are saving it

    Returns
    -------
    Plot of calibration error or systematic error as a function of the baseline length or baseline redundancy
    '''
    if Calibration_error==True:
        vis_pert_recovered = Vis_matrix_each_bl(m, true_vis_pert, pert_gain_rec)[1]
        Vis_error = np.abs(vis_pert_recovered - true_vis_pert)/np.abs(true_vis_pert)
    else:
        Vis_error = np.abs(true_vis_pert-true_vis_unpert)/np.abs(true_vis_unpert)
    median_vis, bl_lengths_nominal, min_arr, max_arr, counts_bl_lengths_nominal = Median_range_vis_plots(m, Vis_error)
    print (max_arr.max())
    if length==True:
        plt.semilogy(bl_lengths_nominal, median_vis ,'o' + str(median_colour), markersize=4,label=str(pert_level) + ' cm')
        for i in range(len(bl_lengths_nominal)):
            if i==0:
                plt.vlines(bl_lengths_nominal[i], ymin=min_arr[i], ymax=max_arr[i], color=str(range_colour), label = str(pert_level)+ ' cm')
            else:
                plt.vlines(bl_lengths_nominal[i], ymin=min_arr[i], ymax=max_arr[i], color=str(range_colour))
        plt.xlabel('Baseline length (m)')
    else:
        plt.semilogy(counts_bl_lengths_nominal, median_vis ,'o' + str(median_colour), markersize=4,label=str(pert_level) + ' cm')
        for i in range(len(counts_bl_lengths_nominal)):
            if i==0:
                plt.vlines(counts_bl_lengths_nominal[i], ymin=min_arr[i], ymax=max_arr[i], color=str(range_colour), label = str(pert_level)+ ' cm')
            else:
                plt.vlines(counts_bl_lengths_nominal[i], ymin=min_arr[i], ymax=max_arr[i], color=str(range_colour))
        plt.xlabel('Baseline redundancy')
    if Calibration_error==True:
        plt.ylabel(r'$\frac{|V^{corrected}_{pert} - V^{true}_{pert}|}{|V^{true}_{pert}|}$')
        plt.title('Calibration error')
    else:
        plt.ylabel(r'$\frac{|V^{true}_{pert} - V^{true}_{unpert}|}{|V^{true}_{unpert}|}$')
        plt.title('Systematic error')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    if savefig==True:
        plt.savefig(str(fig_name)+'.png', dpi=300, bbox_inches = 'tight')
    else:
        pass

def Histograms_vis_matrix_each_bl(m, x_scat, y_scat, visibilities, rec_gains_full_array):
    '''
    Creates separate histograms for the calibration error for each redundant block. Here, the redundant blocks ARE the same as the blocks used
    in Covariance_poisson and Vecs, and ARE NOT the same as that used in Median_range_vis_plots.
    Parameters
    ----------
    m : Prod params file - for NOMINAL dish positions
    x_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the x position
    y_scat : float
            Single value that determines how much the dishes are offset from the nominal positions in the y position
    visibilities : ndarray
        True visibilities (Nbls,)
    rec_gains_full_array : ndarray
        Alternating real and imaginary gain components (2*Ndish)

    Returns
    -------
    Histogram of calibration error for each redundant block


    '''
    sum_counts = Bls_counts(m)[1]
    lims = np.append(0, sum_counts)
    bl_descript = baseline_length_direction(m, x_scat, y_scat)[2]
    Vis_error_abs_array = Vis_matrix_each_bl(m, visibilities, rec_gains_full_array)
    redundant_vis = np.split(Vis_error_abs_array, sum_counts)
    for blk in range(len(lims)-1):
        plt.hist(redundant_vis[blk], color='b')
        plt.xlabel('Relative error in visibility magnitude')
        plt.title(str(bl_descript[lims[blk]]) + ': ' + str(redundant_vis[blk].shape[0]) + ' baselines')
        #plt.savefig('/home/zahra/hirax_corrcal/Vis_errors_each_bl/Unique_bl_number_' + str(blk) +'.png', dpi=300, bbox_inches='tight')
        plt.show() # IF I DON'T INCLUDE THE PLT.SHOW, IT SEEMS TO STACK THE HISTOGRAMS ON TOP OF EACH OTHER SO THAT ALL HISTOGRAMS HAVE THE SAME MAX AMPLITUDE. I RUN THIS ON JUPYTER NOTEBOOK
        # SO I DON'T HAVE TO MANUALLY CLOSE EACH PNG OUTPUT

def Scatterplot(m):
    '''
    Scatterplot of array

    Parameters
    ----------
    m : Prod params file - for the NOMINAL dish positions

    Returns
    -------
    Annotated scatter plot of array
    '''

    t=m.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=np.int(Nfeeds/2)

    x=t.feedpositions[:,0] #these are x and y positions
    y=t.feedpositions[:,1]
    x = x[:Ndish] + np.load('/home/zahra/corrcal2/random_pt1_x_64.npy')[:Ndish] #load the arrays saved for the scatter in x and y positions
    y = y[:Ndish] + np.load('/home/zahra/corrcal2/random_pt1_y_64.npy')[:Ndish]
    print (x)
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='k')
    plt.ylabel('Distance (m)')
    plt.xlabel('Distance (m)')
    for i in range(0,Ndish):
        j=i+Ndish
        ax.annotate((i),(x[i],y[i]))
    plt.show()


def gain_error_scatterplot(m, rec_gains_full_array):
    '''
    Scatterplot of array with colour bar for gain errors

    Parameters
    ----------
    m : Prod params file - for the NOMINAL dish positions
    rec_gains_full_array : ndarray
        Alternating real and imaginary gain components (2*Ndish)
    Returns
    -------
    Coloured scatter plot of array to show distribution of gain errors
    '''

    t=m.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=np.int(Nfeeds/2)

    gain = np.ones(Ndish)
    sim_gains = np.ones(2*Ndish)
    sim_gains[::2] = gain.real
    sim_gains[1::2] = gain.imag
    abs_gains_rec = np.abs(rec_gains_full_array[::2] + 1j * rec_gains_full_array[1::2])
    rel_error = abs_gains_rec - sim_gains

    x=t.feedpositions[:,0][:Ndish] #these are x and y positions
    y=t.feedpositions[:,1][:Ndish]
    x = x + np.load('/home/zahra/corrcal2/random_pt1_x_64.npy')[:Ndish] #load the arrays saved for the scatter in x and y positions
    y = y + np.load('/home/zahra/corrcal2/random_pt1_y_64.npy')[:Ndish]

    plt.scatter(x, y, c=rel_error, s=200)
    plt.xlabel('Antenna x-location',fontsize=12); plt.ylabel('Antenna y-location',fontsize=12);plt.title('Relative amplitude error')
    plt.colorbar()
    plt.show()

'''
I altered the fit_gains function to be able to compute the recovered gains using multiple cores on my laptop

def fit_gains(run):
    visibilities = Vis_total
    src_array = src_total
    vecs_arr = vecs
    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish)
    sigma = S_max * 0.01
    diag = sigma**2 * np.ones(2*Nbls)
    random_gain = np.random.normal(0,1.e-3, 2*Ndish)
    gvec = sim_gains + random_gain
    mat = corrcal.Sparse2Level(diag, vecs_arr, src_array, 2*lims) #init
    get_chisq = corrcal.get_chisq
    get_gradient = corrcal.get_gradient
    data = np.zeros(2*Nbls)
    data[0::2] = visibilities.real + N_noise_arr[run,::2]
    data[1::2] = visibilities.imag + N_noise_arr[run,1::2]
    fac = 1.;
    normfac=1.
    results = fmin_cg(get_chisq, gvec*fac, get_gradient,(data,mat,ant1,ant2,fac,normfac))
    fit_gains_run = results/fac
    rec_gains = fit_gains_run[::2]
    return rec_gains

runs = 500
S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish)
sigma = S_max*0.01

N_noise_arr = np.zeros((runs, 2*Nbls))
for ind_run in range(runs):
    N_noise = np.random.normal(0, sigma, 2*Nbls)
    N_noise_arr[ind_run, :] = N_noise

fit_gains_arr = np.array([])

with concurrent.futures.ProcessPoolExecutor() as executor:
    runs_arr = np.arange(runs)
    results = executor.map(fit_gains, runs_arr)
    for result in results:
        print (result)
        fit_gains_arr = np.append(fit_gains_arr, result)


results_arr = fit_gains_arr.reshape(runs, Ndish)
'''

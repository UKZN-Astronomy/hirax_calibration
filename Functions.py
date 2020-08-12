
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
from LogCal_code import Bls_counts
from cora.foreground import poisson as ps
import numpy.random.normal as rand_normal


def baseline_arr(m, freq):
    '''
    Calculates baselines in uv coordinates

    Parameters
    ----------
    freq : float
        The frequency at which you are observing (in MHz).

    Returns
    -------
    baselines : ndarray
        The baselines in uv coordinates (Nbls,2)
        Organised according to baseline redundancy
    '''
    t = m.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    bl_arr_dish_indices, _, _ = Bls_counts(m)
    x = t.feedpositions[:,0][:Ndish]
    y = t.feedpositions[:,1][:Ndish]

    x = x + np.load('/home/zahra/corrcal2/random_pt1_x_64.npy')[:Ndish]
    y = y + np.load('/home/zahra/corrcal2/random_pt1_y_64.npy')[:Ndish]

    wavelength = (3.e8)/(freq*10.**6)
    bl_arr_uv = np.zeros((Nbls,2))
    for i in range(len(bl_arr_dish_indices)):
        dish_0, dish_1 = bl_arr_dish_indices[i]
        bl_ind = [np.int(dish_0), np.int(dish_1)]
        u_coord = (x[bl_ind[1]] - x[bl_ind[0]])/wavelength
        v_coord = (y[bl_ind[1]] - y[bl_ind[0]])/wavelength
        bl_arr_uv[i,:] += np.array([u_coord, v_coord])
    return bl_arr_uv


def Gauss_beam(m, freq, nside):
    '''
    Calculates the beam model
    We assume a Gaussian beam, and assume the same beam for each antenna

    Parameters
    ----------
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used

    Returns
    -------
    Beam : ndarray
        Beam model for each pixel (npix,)
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


# The Mu function follows the equation $\mu_n = \int S^n \frac{dN}{dS} dS$, with n=2 used for visibilities and n=3 for sky cov



def Mu(m, index):
    '''
    Calculates mu by assuming a differential source count, dN/dS, as per Santos et al, 2005

    Parameters
    ----------
    index : int
        Input the index of source flux, S, depending on what you which to calculate
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

    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish) #12 micro jansky sensitivity for full array,
    # scaled for our size array, multiplied by 5 for a 5\sigma detection
    alpha = 4000
    beta = 1.75
    mu_index = alpha/(index-beta) * (S_max**(index-beta))
    return mu_index


def source_count(flux):
    r"""Power law luminosity function."""

    gamma1 = 0.2#1.75
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

    Parameters
    ----------
    area : float
        The area the population is contained within (in sq degrees).

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


def Src_vector(m, freq, nside, map, thresh_max_vals_beam):
    '''
    Calculates the source vector, S, that is input into CorrCal (for known source positions),
    and contains the per visibility response to bright sources with known position.
    Vector S is then used to calculate the total visibility contribution from bright sources.
    Note that we have separated S into its real and imaginary but have not separated the
    visibilities into these components as yet

    Parameters
    ----------
    freq : float
        The frequency at which you are observing (in MHz).
    nside : int
        The dimensions of the HEALPix map used
    map : ndarray
        The map of bright point sources that you are using (npix,)

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
    bl_arr = baseline_arr(m, freq)
    bl_arr_dish_indices, _, _ = Bls_counts(m)

    map_vals = map
    vals_beam = map_vals*beammodel
    pix_valsbeam = np.where(vals_beam >thresh_max_vals_beam)[0] #reducing this value from 1e-3 gives bad results
    # for the 1-1.2 Jy case with known positions
    source_number = len(pix_valsbeam)
    src = np.zeros((len(bl_arr_dish_indices), source_number), dtype='complex')
    for n in range(source_number):
        non_zero_pix = pix_valsbeam[n]
        for i in range(len(bl_arr_dish_indices)):
            fringe_ind = fringe(angpos[non_zero_pix], zenith, bl_arr[i])
            vis_ind_source_ind = map_vals[non_zero_pix] * beammodel[non_zero_pix]**2 * fringe_ind
            src[i, n] = vis_ind_source_ind
    Vis_bright_sources = np.sum(src, axis=1)
    src_total = np.zeros((2*Nbls, 2*source_number))
    src_total[::2,::2] = src.real
    src_total[1::2,1::2] = src.imag
    src_total = src_total.T
    return Vis_bright_sources, src_total

def Visibilities_poisson(m, freq, nside):
    '''
    Calculates visibilities due to unresolved point sources below the S_{max} set by the
    telescope sensitivity. We assume sources follow a Poisson distribution, and do not account
    for source clustering.

    Parameters
    ----------
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
    bl_arr = baseline_arr(m, freq)
    bl_arr_dish_indices, _, _ = Bls_counts(m)

    point_source_vis = Mu(m, 2)
    pix_beam = np.where(beammodel>1e-10)[0]
    vis_poisson = np.array([])
    for i in range(len(bl_arr_dish_indices)):
        fringe_ind = fringe(angpos[pix_beam], zenith, bl_arr[i])
        vis_ind = 4 * np.pi/npix * point_source_vis * np.sum(beammodel[pix_beam]**2 * fringe_ind)
        vis_poisson = np.append(vis_poisson, vis_ind)
    return vis_poisson

def Covariance_poisson(m, freq, nside):
    '''
    Covariance matrix computed using unresolved point sources following a Poisson distribution,
    and we compute the covariance at a single frequency, across different baselines.
    Covariance matrices are computed separately for each redundant block such that we do not
    compute a full (Nbls, Nbls) matrix

    Parameters
    ----------
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
    _, sum_counts, bl_counts = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks
    t = m.telescope
    zenith = t.zenith

    angpos = hputil.ang_positions(nside)
    npix = hp.nside2npix(nside)
    point_source_cov = Mu(m, 3)
    bl_arr = baseline_arr(m, freq)
    beammodel = Gauss_beam(m, freq, nside)
    pix_beam = np.where(beammodel>1e-10)[0]

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




def Vecs(m, freq, nside):
    '''
    Decomposes the covariance matrix into eigenvalues and eigenvectors, separately for each
    redundant block. This is the R vector that is input into CorrCal (if you are not using the
    redundant case in CorrCal). We set a threshold for the eigenvalues that are considered,
    which determines the number of vectors, nvec. Also R is separated into its real and
    imaginary components

    Parameters
    ----------
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

    _, sum_counts, _ = Bls_counts(m)
    lims = np.append(0, sum_counts) #these are the edges of the redundant blocks

    Cov_dic = Covariance_poisson(m, freq, nside)
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

def fit_gains(runs, src_array, vecs, visibilities):
    '''
    Computes the recovered gains for both amplitude and phase. The inputs include the source
    vector, S. The two options for the vector depend on whether or not source positions are
    known.
    The vector, R, constructed from the visibility covariance, and the visibilities. The two
    options for the vector depend on whether or not you are using the redundant case.
    The visibilities input here are noiseless. Noise is input into the visibilities in this
    function, and visibilities are then separated into real and imaginary components. I either
    use the sum of unresolved and bright sources, or just the unresolved sources.

    Parameters
    ----------
    runs : int
        Number of noise realisations you want to do (500 runs is sufficient)
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
    S_max =  5 * 12 * 1.e-6 * np.sqrt(1024./Ndish)
    sigma = S_max*0.01 # we include a small amount of per visibility noise
    diag = sigma**2*np.ones(2*Nbls)
    random_gain = rand_normal(0,1.e-3, 2*Ndish) #initial guess input into corrcal
    gvec = sim_gains + random_gain

    get_chisq = corrcal.get_chisq
    get_gradient = corrcal.get_gradient
    mat = corrcal.Sparse2Level(diag,vecs,src_array,2*lims)

    rec_gains = np.zeros((runs,Ndish*2))
    for ind_run in range(runs):
        print (ind_run)
        Noise_array = rand_normal(0, sigma, 2*Nbls)
        data = np.zeros(2*visibilities.size)
        data[0::2] = visibilities.real + Noise_array[::2]
        data[1::2] = visibilities.imag + Noise_array[1::2]
        fac=1.;
        normfac=1.
        results = fmin_cg(get_chisq, gvec*fac, get_gradient,(data,mat,ant1,ant2,fac,normfac))
        fit_gains_run = results/fac
        rec_gains[ind_run,:] = fit_gains_run
        print (fit_gains_run[::2])
    return rec_gains

'''
I altered the above function to be able to compute the recovered gains using multiple cores on my laptop

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

    mu = 0
    N_real = np.random.normal(mu, sigma, Nbls)
    N_imag = np.random.normal(mu, sigma, Nbls)
    N_comp = np.array([])
    for i in range(len(N_real)):
        N_comp = np.append(N_comp,complex(N_real[i],N_imag[i]))
    meas_vis = visibilities + N_comp
    return meas_vis

def recovered_gains(runs): # I use this for LogCal
    '''
    Recovered gain amplitudes computed. The phase component is not included here

    Parameters
    ----------
    visibilities :
        Visibilities with noise (Nbls,)

    Returns
    -------
    rec_gains : ndarray
        Recovered gain amplitudes (Ndish,)
    '''
    visibilities = Vis_total
    sigma = S_max*0.01
    rec_gains = np.zeros((runs,Ndish))
    for ind_run in range(runs):
        print (ind_run)
        meas_vis = Measured_vis(sigma, visibilities)
        rec_gains[ind_run,:] = Logcal_solutions(m,visibilities,visibilities, gain,meas_vis, sigma)[0][:Ndish]
        print (np.exp(rec_gains[ind_run,:]))
    return rec_gains


import numpy as np
import healpy
import h5py, time
from drift.core import manager
from scipy.optimize import fmin_cg, minimize
import corrcal
import sys
sys.path.insert(0,'/home/zahra/hirax_corrcal/')
from LogCal_code import index_find, Bls_counts, Indices_for_reordering
from Functions import baseline_length_direction
import matplotlib.pyplot as plt


m = manager.ProductManager.from_config('/home/zahra/hirax_corrcal_desktop/draco_corrcal_pipeline/3by3_array/prod_params.yaml')
m_nominal = manager.ProductManager.from_config('/home/zahra/hirax_corrcal_desktop/draco_corrcal_pipeline/3by3_array/prod_params_nominal.yaml')


tstream_brightsources = h5py.File('/home/zahra/hirax_corrcal_desktop/draco_corrcal_pipeline/3by3_array/draco_synthesis/bright_sources_4_5Jy/tstream_true_0.h5','r')


tstream_allsources = h5py.File('/home/zahra/hirax_corrcal_desktop/draco_corrcal_pipeline/3by3_array/draco_synthesis/allsources_5Jy/tstream_gain_noise_0.h5','r')

rand_gains = h5py.File('/home/zahra/hirax_corrcal_desktop/draco_corrcal_pipeline/3by3_array/draco_synthesis/allsources_5Jy/rand_gains_pt1_0.h5','r')

def mmodes(m_nominal, freq):
    '''
    M modes corresponding to the length of the EW baseline, for each baseline in the unperturbed array. This is used when calling the visibility
    covariance due to unresolved point sources, in `fit_gains_base`.
    In reality, the EW baselines will not be the same for quasi-redundant baselines. I am currently not using m modes that correspond exactly to the
    bl lengths of the perturbed arrays.

    Parameters
    ----------
    m_nominal: prod_params file
        Prod params for the NOMINAL array.
    freq: float
        Observing frequency (MHz)

    Returns
    -------
    mmode_unperturbed : ndarray
        M mode for the EW baseline length of each unique baseline of the unperturbed array (Nblock,)
    '''

    sum_counts = Bls_counts(m_nominal)[1]
    lims = np.append(0, sum_counts)
    t_nom = m_nominal.telescope
    Ndish =np.int(t_nom.feedpositions.shape[0]/2)

    x_scat = np.zeros(Ndish)
    y_scat = x_scat
    EW_bl_length = baseline_length_direction(m_nominal, x_scat, y_scat)[0][:,0]
    wavelength = (3.e8)/(freq*10.**6)
    EW_u_coord = np.abs(EW_bl_length)/wavelength # u = b/lambda
    ellmode_nominal = EW_u_coord*2*np.pi # ell = 2 * np.pi * u
    mmode_unperturbed = ellmode_nominal[lims[:-1]] # m corresponding to ell
    return mmode_unperturbed


def Vecs(Cov_dic):
    '''
    Decomposes the covariance matrix into eigenvalues and eigenvectors, separately for each
    redundant block. This is the R vector that is input into CorrCal (if you are not using the
    redundant case in CorrCal). We set a threshold for the eigenvalues that are considered,
    which determines the number of vectors, nvec. Also R is separated into its real and
    imaginary components

    Parameters
    ----------
    Cov_dic: dict
        Covariance matrix generated separately for each redundant block

    Returns
    -------
    vecs : ndarray
        R vector (2*Nvec, 2*Nbls)
    '''
    bl_arr_dish_indices, sum_counts, _, _, _ = Bls_counts(m_nominal)
    lims = np.append(0, sum_counts)
    t=m.telescope
    Ndish =np.int(t.feedpositions.shape[0]/2)
    Nbls= np.int(Ndish*(Ndish-1)/2)

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

def fit_gains_base(m, m_nominal, tstream_allsources, tstream_allsources_with_noise = True, source_positions_unknown=True, tstream_brightsources=None, vecs_redundant = True, unity_gains = True, gains_file = None, save_gains_error = False):

    '''
    Calculates complex gain errors (recovered gains - true gains) using CorrCal. The gain errors are placed in a h5 file under a `gain` header.
    The shape is (nfreq, 2*Ndish, ntime) with zero padding for [:, Ndish:, :], so as to match the shape of draco visibility timestream data.

    Parameters
    ----------
    m: prod params file
        Prod params file for the perturbed array
    m_nominal: prod_params file
        Prod params for the unperturbed array - this file does not need to be used to generate beamtransfer matrices. It is rather a way of
        getting the x and y positions of each antenna in the unperturbed array, and is used for redundant blocking.
    tstream_allsources: h5 file
        Time stream visibility data that may contain gains and thermal (Gaussian) noise, for all sources (unresolved point sources AND bright sources).
        It is not required for the data to contain gains and/or noise.
    tstream_allsources_with_noise: boolean
        Specify if `tstream_allsources` contains per-visibility noise. If True, the noise covariance matrix input into CorrCal will use the same std
        that was used to generate the Gaussian per-visibility noise.
    source_positions_unknown: boolean
        Specify if the positions of bright sources are known. If True, the code will use a source vector containing only zeros. If False, the
        per-visibility response to known bright sources from the point source catalogue will be used here.
    tstream_brightsources: h5 file
        If `source_positions_unknown` is False, then the file containing the timestream data for known bright sources will be input here for the source
        vector.
    vecs_redundant: boolean
        Specify if the visibility covariance will be input into CorrCal. If True, the redundant case will be implemented, in which no knowledge of
        the covariance matrix is assumed. If False, the covariance due to unresolved point sources is called from drift.core.kltransform.
    unity_gains: boolean
        If True, we assume simulated/true gains of 1. This would be the case if no gains are applied to the timestream data `tstream_allsources`.
        If False, the `tstream_allsources` does contain gains that are not exactly 1.
    gains_file: h5 file
        Gains file if the true gains are not 1. Input this if `unity_gains` was set to False.
    save_gains_error: boolean
        If True, the gains error would be saved to an h5 file under a `gain` header, which is needed to run the last command in `run_everything.sh` to
        get the visibility error. To get the gain output without creating the h5 file, set this to False.
    Returns
    -------
    recovered gains: ndarray
        Complex recovered gains from CorrCal (nfreq, 2*Ndish, ntime)
    gains_error: h5 file
        Difference between complex recovered and true gains, which is stored in a h5 file (nfreq, 2*Ndish, ntime)

    '''

    get_chisq = corrcal.get_chisq
    get_gradient = corrcal.get_gradient

    bl_arr_dish_indices, sum_counts, _, _, _ = Bls_counts(m_nominal)
    lims = np.append(0, sum_counts)
    indices_vis = Indices_for_reordering(m_nominal, tstream_allsources)

    t=m.telescope
    Ndish =np.int(t.feedpositions.shape[0]/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    unique = t.uniquepairs
    indices_unique_bls = Indices_for_reordering(m_nominal, unique, h5_file=False) # unique[indices_unique_bls] = Bls_counts(m_nominal)[0]

    ant1 = bl_arr_dish_indices[:,0].astype(int)
    ant2 = bl_arr_dish_indices[:,1].astype(int)

    random_gain = np.load('/home/zahra/hirax_corrcal/saved_arrays/random_gain_1e-3_128val.npy')[:2*Ndish]

    freq_array = tstream_allsources['index_map']['freq']['centre']
    time_array = tstream_allsources['index_map']['time']['ctime']
    rec_gains_mat = np.zeros((len(freq_array), 2*Ndish, len(time_array)), dtype=complex)
    gains_error_mat = np.zeros((len(freq_array), 2*Ndish, len(time_array)), dtype=complex)

    for freq_ind in range(len(freq_array)):
        freq = freq_array[freq_ind]
        if vecs_redundant==True:
            v1=np.zeros(2*Nbls)
            v1[0::2]=1
            v2=np.zeros(2*Nbls)
            v2[1::2]=1
            vecs = np.vstack([v1,v2])*1.e3 # used for the redundant case in CorrCal
        else:
            mmode_unperturbed = mmodes(m_nominal, freq)
            klobj = m.kltransforms['kl_fg_0thresh']
            cv_fg = klobj.point_source_unresolved()
            Cov_dic = {}
            unique_m_unpert = np.unique(mmode_unperturbed)
            for i in unique_m_unpert:
                point_source_cov=m.beamtransfer.project_matrix_sky_to_telescope(i,cv_fg)[freq_ind,:unique.shape[0],freq_ind,:unique.shape[0]]
                point_source_cov_arranged = point_source_cov[indices_unique_bls][:,indices_unique_bls] # creates (Nbls,Nbls) cov matrix for
                # each unique m mode - the m modes correspond to the EW baselines of the nominal array.
                m_unpert_indices = np.where(mmode_unperturbed==i)[0]
                for j in m_unpert_indices:
                    Cov_dic[j] = point_source_cov_arranged[lims[j]:lims[j+1], lims[j]:lims[j+1]] # gives n x n covariance matrices for each
                    # redundant block, with the matrix size depending on the baseline redundancy

            vecs = Vecs(Cov_dic)

        sim_gains = np.zeros(2*Ndish)
        for time_ind in range(len(time_array)):
            if unity_gains==True:
                sim_gains[::2] = 1
                sim_gains_complex = sim_gains[::2] + 1j*sim_gains[1::2]
            else:
                sim_gains_complex = gains_file['gain'][freq_ind, :Ndish, time_ind]
                sim_gains[::2] = sim_gains_complex.real
                sim_gains[1::2] = sim_gains_complex.imag

            gvec = sim_gains + random_gain

            visibilities = tstream_allsources['vis'][freq_ind, :, time_ind]
            visibilities = visibilities[indices_vis]
            data = np.zeros(2*visibilities.size)
            data[::2] = visibilities.real
            data[1::2] = visibilities.imag
            if source_positions_unknown==True:
                src_array = np.zeros(2*Nbls)
            else:
                vis_brightsource = tstream_brightsources['vis'][:,freq_ind,:,time_ind]
                vis_brightsource = vis_brightsource[:,indices_vis]
                nsource, _ = vis_brightsource.shape
                src_array = np.zeros((nsource, 2*Nbls))
                src_array[:, ::2] = vis_brightsource.real
                src_array[:, 1::2] = vis_brightsource.imag

            dt = time_array[1] - time_array[0]
            df = tstream_allsources['index_map']['freq']['width'][0] * 1e6
            ndays = 1
            nsamp = int(ndays * dt * df)
            std = m.telescope.tsys_flat / np.sqrt(2 * nsamp)
            if tstream_allsources_with_noise==True:
                diag = std**2*np.ones(2*Nbls)
            else:
                diag = (std*1e-3)**2*np.ones(2*Nbls)

            mat = corrcal.Sparse2Level(diag, vecs, src_array, 2*lims)

            fac=1.;
            normfac=1.
            results = fmin_cg(get_chisq, gvec*fac, get_gradient,(data,mat,ant1,ant2,fac,normfac))
            rec_gains = results/fac
            rec_gains_complex = rec_gains[::2] + 1j*rec_gains[1::2]
            gain_error_complex = rec_gains_complex - sim_gains_complex
            rec_gains_mat[freq_ind, :Ndish, time_ind] = rec_gains_complex
            gains_error_mat[freq_ind, :Ndish, time_ind] = gain_error_complex

    if save_gains_error ==True:
        gain_error_result = h5py.File('gains_error.h5')
        gain_error_result.create_dataset('gain', data = gains_error_mat)
    else:
        gain_error_result = gains_error_mat
    return rec_gains_mat, gain_error_result

mat = fit_gains_base(m, m_nominal, tstream_allsources, tstream_allsources_with_noise=True, source_positions_unknown=False, tstream_brightsources=tstream_brightsources, vecs_redundant = False, unity_gains = False, gains_file = rand_gains)

'''As it stands, the fit_gains_base function doesn't use the config file for the perturbed array (m) because we have only considered the mmodes
for the nominal array.'''

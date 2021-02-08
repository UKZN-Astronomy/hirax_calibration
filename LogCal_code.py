import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.patches as mpatches
from drift.core import manager
from scipy.sparse import lil_matrix
from random import seed
from random import random

def log(function):
    ''' Take the log of a function with zeros'''
    zer=function==0.
    function[zer]=1.e-10
    return np.log(function)

def index_find(input_arr,ind_1,ind_2):
    '''Find the elements in an input array that have first component of ind_1 and second component of ind_2'''
    arr_find=np.where(input_arr==[ind_1, ind_2])[0]
    full_arr_index=np.array([])
    for i in range(len(arr_find)-1):
        if arr_find[i]==arr_find[i+1]:
            full_arr_index=np.append(full_arr_index,arr_find[i])
    return full_arr_index

def prod_ind(ts_file,ind_1,ind_2):
    '''Find the elements in timestream file that has first component of ind_1 and second component of ind_2'''
    a_loc=np.where(ts_file['input_a']==ind_1)[0]
    b_loc=np.where(ts_file['input_b']==ind_2)[0]
    for i in a_loc:
        for j in b_loc:
            if i==j:
                location=i
    return location

def Bls_counts(manager_config_file):
    '''
    Note that this function uses the NOMINAL dish positions - i.e. you input the prod params file corresponding to the perfectly redundant array.
    Organises the baselines in order of redundancy (from most to least redundant) and gives the number of redundant bls for each of the unique bls. The number
    of unique baselines is equal to the number of redundant blocks, Nblock.

    Parameters
    ----------
    manager_config_file : prod params file
        Config file from driftscan runs using NOMINAL dish positions (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)

    Returns
    -------
    arranged_dish_indices : ndarray
        Dish indices for baselines ordered in decreasing redundancy (Nbls, 2)
    sum_counts : ndarray
        Cumulative number of redundant bls (Nblock)
    arranged_bl_redundancy_counts : ndarray
        Number of redundant bls for each of the unique bls (Nblock)
    bl_abs_mag_length_redundancy_ordered_nominal : ndarray
        Baseline lengths of nominal array (Nbls,)
    bl_descrip_list_nominal : str
        Baseline length and direction
    '''

    t = manager_config_file.telescope
    Nfeeds,_ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    x = t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y = t.feedpositions[:,1]

    unique = np.unique(t.baselines,axis=0)
    #N_unique_bls=len(unique) #can't use this if you set auto_correlations=Yes in prod params file
    a = unique[:,0]
    b = unique[:,1]

    bl_info_matrix = np.array([])
    for k in range(0,len(b)):
        for i in range(0,Ndish):
            for j in range(0,Ndish):
                if y[j]-y[i]==b[k]:
                    if x[j]-x[i]==a[k]:
                        if i!=j:
                            arr_sing = i,j,a[k],b[k]
                            bl_info_matrix = np.append(bl_info_matrix, arr_sing)
    bl_info_matrix = np.reshape(bl_info_matrix,(-1,4)) # 4 columns - first 2 give dish indices of bls, third and fourth give the corresponding baseline length in x and y
    # directions respectively
    _, ind, bl_counts = np.unique(bl_info_matrix[:,2:4], return_counts=True, return_index=True, axis=0)
    arranged_bl_counts = np.flip(np.sort(bl_counts))
    arranged_bl_counts_indices = np.flip(np.argsort(bl_counts))
    arranged_unique_bls_lengths = bl_info_matrix[:,2:4][ind[arranged_bl_counts_indices]]
    full_indices_list_all_bls = np.array([])
    for i in range(arranged_unique_bls_lengths.shape[0]):
        indices_list_one_bl = index_find(bl_info_matrix[:,2:4], arranged_unique_bls_lengths[i,0], arranged_unique_bls_lengths[i,1])
        full_indices_list_all_bls = np.append(full_indices_list_all_bls, indices_list_one_bl)
    full_indices_list_all_bls = full_indices_list_all_bls.astype(int)
    arranged_dish_indices = bl_info_matrix[:,0:2][full_indices_list_all_bls]

    bl_xy_lengths_redundancy_ordered_nominal = bl_info_matrix[:,2:4][full_indices_list_all_bls]
    bl_abs_mag_length_redundancy_ordered_nominal = np.linalg.norm(bl_xy_lengths_redundancy_ordered_nominal, axis=1)
    bl_descrip_list_nominal = []
    for i in bl_xy_lengths_redundancy_ordered_nominal:
        bl_descrip = (str(np.int(i[0])) + ' EW' + ', ' + str(np.int(i[1])) + ' NS')
        bl_descrip_list_nominal = np.append(bl_descrip_list_nominal, bl_descrip)
    sum_counts = np.cumsum(arranged_bl_counts)
    return arranged_dish_indices, sum_counts, arranged_bl_counts, bl_abs_mag_length_redundancy_ordered_nominal, bl_descrip_list_nominal

def Indices_for_reordering(manager_config_file, disordered_input, h5_file=True):
    '''
    Returns indices that that can be used to reorder arrays according to decreasing baseline redundancy

    Parameters
    ----------
    manager_config_file : prod params file
        Config file using NOMINAL dish positions
    disordered_input : ndarray
        Array that is not organised according to baseline redundancy
    h5_file : boolean
        True if it is an h5 file

    Returns
    -------
    Indices_array : ndarray
        The disordered_input array organised according to decreasing baseline redundancy
    '''
    arranged_dish_indices=Bls_counts(manager_config_file)[0]
    Indices_array=np.array([])

    if h5_file==True:
        prods=disordered_input['index_map']['prod'][:]
        for i in arranged_dish_indices:
            Indices_array=np.append(Indices_array,prod_ind(prods,np.int(i[0]),np.int(i[1])))
    else:
        row, col = disordered_input.shape
        for i in range(row):
            if disordered_input[i,0] > disordered_input[i,1]:
                 disordered_input[i,[0,1]] = disordered_input[i,[1,0]]

        for i in arranged_dish_indices:
                Indices_array=np.append(Indices_array,index_find(disordered_input,np.int(i[0]),np.int(i[1])))
    Indices_array = Indices_array.astype(int)
    return Indices_array


def A_matrix(manager_config_file):
    '''
    A matrix that contains details of the array layout, see Liu et al 2010

    Parameters
    ----------
    manager_config_file : prod params file
        Config file from driftscan runs using NOMINAL dish positions (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)

    Returns
    -------
    A : ndarray
        A matrix for amplitude calibration (Nbls+1, Nblock + Ndish)
    A_phase : ndarray
        A matrix for phase calibration (Nbls+1, Nblock + Ndish)
    '''

    t = manager_config_file.telescope
    x = t.feedpositions[:,0]
    y = t.feedpositions[:,1]
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_info_matrix = Bls_counts(manager_config_file)[0]
    Nbls, _= bl_info_matrix.shape
    sum_counts = Bls_counts(manager_config_file)[1]

    N_unique_bls = len(sum_counts)
    N_unknowns = Ndish + N_unique_bls
    A = np.zeros((Nbls, N_unknowns))
    A_phase = np.zeros((Nbls, N_unknowns))
    for n in range(Nbls):
        single_i = np.int(bl_info_matrix[n, 0:2][0])
        single_j = np.int(bl_info_matrix[n, 0:2][1])
        A[n, single_i] = 1
        A[n, single_j] = 1
        A_phase[n, single_i] = 1
        A_phase[n, single_j] = -1

    sum_counts_new = np.append(np.array([0]), sum_counts)
    for i in range(len(sum_counts_new)-1):
        A[sum_counts_new[i]:sum_counts_new[i+1], Ndish+i] = 1
        A_phase[sum_counts_new[i]:sum_counts_new[i+1], Ndish+i] = 1
    constr_sum = np.append(np.ones(Ndish), np.zeros(N_unique_bls))
    constr_x_orient = np.append(x[:Ndish], np.zeros(N_unique_bls))
    constr_y_orient = np.append(y[:Ndish], np.zeros(N_unique_bls))
    A_no_constr = A
    A_no_constr_phase = A_phase
    A = np.vstack((A,constr_sum))
    A_phase = np.vstack((A_phase, constr_sum, constr_x_orient, constr_y_orient))
    return A, A_phase

def Noise_cov_matrix(manager_config_file, measured_vis, sigma):
    '''
    Noise covariance matrix weighted by measured visibilities, see Liu et al 2010

    Parameters
    ----------
    manager_config_file : prod params file
        Config file from driftscan runs using NOMINAL dish positions (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)
    measured vis: ndarray
        Visibilities with noise (Nbls,)
    sigma: float
        Noise calculated from Radiometer equation

    Returns
    -------
    N_amp : ndarray
        N matrix for amplitude calibration (Nbls + 1, Nbls + 1)
    N_phase : ndarray
        N matrix for phase calibration (Nbls + 3, Nbls + 3)
    '''

    t = manager_config_file.telescope
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_info_matrix = Bls_counts(manager_config_file)[0]
    Nbls, _ = bl_info_matrix.shape
    N_cov = np.array([])
    for i in measured_vis:
        N_cov = np.append(N_cov, sigma**2/(np.abs(i))**2)
    N =  np.diag(N_cov)
    N_amp = np.vstack((N, np.zeros(Nbls)))
    N_phase = np.vstack((N, np.zeros((3, Nbls))))
    #print (N.shape)
    zeros = np.zeros((Nbls + 1, 1))
    zeros_phase = np.zeros((Nbls + 3, 3))
    N_amp = np.hstack((N_amp, zeros))
    N_phase = np.hstack((N_phase, zeros_phase))
    N_amp[-1][-1] = 1.
    N_phase[-1][-1] = N_phase[-2][-2] = N_phase[-3][-3]=1.
    return N_amp, N_phase


def lstsq(manager_config_file, A_rec, N_rec, mv):
    '''
    Least squares estimator, see Liu et al 2010
    Computes recovered gains for amplitude or phase, depending on whether the A and N matrices input are for amplitude or phase)

    Parameters
    ----------
    manager_config_file : prod params file
        Config file from driftscan runs using NOMINAL dish positions (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)
    A_rec : ndarray
        A matrix
    N_rec : ndarray
        N matrix
    mv : ndarray
        Log of visibilities with noise (Nbls,)

    Returns
    -------
    x_rec_real : ndarray
        recovered gains (Ndish + Nblock,)
    '''

    t = manager_config_file.telescope
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    x_lstsq_t1 = np.matmul(A_rec.T, np.linalg.pinv(N_rec))
    x_lstsq_t1 = np.linalg.pinv(np.matmul(x_lstsq_t1,A_rec))
    x_lstsq_t2 = np.matmul(A_rec.T, np.linalg.pinv(N_rec))
    x_lstsq_t2 = np.matmul(x_lstsq_t2, mv)
    x_rec_real = np.matmul(x_lstsq_t1, x_lstsq_t2)
    return x_rec_real

def Logcal_solutions(manager_config_file, meas_vis_no_noise, true_vis, gain, meas_vis, sigma):
    '''
    Uses above functions to compute the LogCal solutions for amplitude calibration

    Parameters
    ----------
    manager_config_file : prod params file
        Config file from driftscan runs using NOMINAL dish positions (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)
    meas_vis_no_noise : ndarray
        Noiseless visibilities (Nbls,)
    true_vis : ndarray
        Visibilities with no gains or noise (Nbls,)
    gain : ndarray
        True gains (Ndish,)
    meas_vis : ndarray
        Visibilities with noise and gains (Nbls,)
    sigma : float
        Noise calculated from Radiometer equation

    Returns
    -------
    x_rec_real : ndarray
        Recovered gains (Ndish + Nblock,)
    x_rec_real_no_noise : ndarray
        Recovered gains with no noise (Ndish + Nblock,)
    '''

    t = manager_config_file.telescope
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_info_matrix = Bls_counts(manager_config_file)[0]
    Nbls, _ = bl_info_matrix.shape
    sum_counts = Bls_counts(manager_config_file)[1]
    N_unique_bls = len(sum_counts)
    gain_amp = np.log(gain).real
    constr_amp_sum = np.sum(gain_amp)
    mv_real, mv_real_no_noise = log(np.abs(meas_vis)), log(np.abs(meas_vis_no_noise))
    mv_real, mv_real_no_noise = np.append(mv_real,constr_amp_sum), np.append(mv_real_no_noise, constr_amp_sum)
    tv_real = log(np.abs(true_vis))
    A, A_phase = A_matrix(manager_config_file)
    N, N_phase = Noise_cov_matrix(manager_config_file,meas_vis,sigma)
    x_rec_real = lstsq(manager_config_file, A, N, mv_real)
    x_true_real = np.append(gain_amp,tv_real)
    x_rec_real_no_noise = np.linalg.lstsq(A, mv_real_no_noise, rcond=None)[0]
    mv_real_recovered = np.matmul(A, x_rec_real)
    mv_real_recovered_no_noise = np.matmul(A, x_rec_real_no_noise)
    return x_rec_real, x_rec_real_no_noise

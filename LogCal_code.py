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
    '''
    Take the logarithm of a function, accounting for any zero values in the function
    '''
    zer=function==0.
    function[zer]=1.e-10
    return np.log(function)

def index_find(input_arr,ind_1,ind_2):
    '''
    Finds elements in an input array that contains a specific pair of dish indices, ind_1 and ind_2
    '''
    arr_find=np.where(input_arr==[ind_1, ind_2])[0]
    full_arr_index=np.array([])
    for i in range(len(arr_find)-1):
        if arr_find[i]==arr_find[i+1]:
            full_arr_index=np.append(full_arr_index,arr_find[i])
    return full_arr_index


def Bls_counts(manager_config_file):
    '''
    Organises the baselines in order of redundancy (from most to least redundant) and gives the number of redundant bls for each of the unique bls. The number
    of unique baselines is equal to the number of redundant blocks, Nblock

        Parameters
    ----------
    manager_config_file : config file
        Config file from driftscan runs (using the Radio Cosmology packages - it's not necessary to use this for the LogCal code, but I found it useful)

    Returns
    -------
    arranged_dish_indices : ndarray
        Dish indices for baselines ordered in decreasing redundancy (Nbls, 2)
    arranged_bl_redundancy_counts : ndarray
        Number of redundant bls for each of the unique bls (Nblock)
    '''

    t = manager_config_file.telescope
    Nfeeds,_= t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    Nbls = np.int(Ndish*(Ndish-1)/2)
    x = t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y = t.feedpositions[:,1]

    unique = np.unique(t.baselines,axis=0)
    a = unique[:,0]
    b = unique[:,1]

    bl_matrix_info = np.array([])
    for k in range(0,len(b)):
        for i in range(0,Ndish):
            for j in range(0,Ndish):
                if y[j] - y[i] == b[k]:
                    if x[j] - x[i] == a[k]:
                        if i!=j:
                            arr_sing = i,j,a[k],b[k]
                            bl_matrix_info = np.append(bl_matrix_info, arr_sing)
    bl_matrix_info = np.reshape(bl_matrix_info,(-1, 4)) # 4 columns, first two are dish indices for each bl. Third and fourth columns
    # are the bl lengths in x and y directions, respectively
    bl_length_unique, ind, bl_redundancy_counts = np.unique(bl_matrix_info[:,2:4] ,return_counts = True, return_index = True, axis = 0)
    arranged_bl_redundancy_counts = np.flip(np.sort(bl_redundancy_counts))
    arranged_bl_redundancy_counts_indices = np.flip(np.argsort(bl_redundancy_counts))
    arranged_bl_length_unique_bls = bl_matrix_info[:,0:2][ind[arranged_bl_redundancy_counts_indices]]
    arranged_bl_length_unique_bls_lengths = bl_matrix_info[:,2:4][ind[arranged_bl_redundancy_counts_indices]]
    full_indices_list_all_bls = np.array([])
    for i in range(arranged_bl_length_unique_bls_lengths.shape[0]):
        indices_list_one_bl = index_find(bl_matrix_info[:,2:4], arranged_bl_length_unique_bls_lengths[i,0], arranged_bl_length_unique_bls_lengths[i,1])
        full_indices_list_all_bls = np.append(full_indices_list_all_bls, indices_list_one_bl)
    full_indices_list_all_bls = full_indices_list_all_bls.astype(int)
    arranged_dish_indices = bl_matrix_info[:,0:2][full_indices_list_all_bls]
    return arranged_dish_indices, arranged_bl_redundancy_counts

def A_matrix(manager_config_file):
    t = manager_config_file.telescope
    x = t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y = t.feedpositions[:,1]
    Nfeeds,_ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_matrix_info = Bls_counts(manager_config_file)[0]
    Nbls,_ = bl_matrix_info.shape
    sum_counts = Bls_counts(manager_config_file)[1]

    N_unique_bls = len(sum_counts)
    N_unknowns = Ndish + N_unique_bls
    A = np.zeros((Nbls,N_unknowns))
    A_phase = np.zeros((Nbls,N_unknowns))
    for n in range(Nbls):
        corr_single_i = np.int(bl_matrix_info[n,0:2][0])
        corr_single_j = np.int(bl_matrix_info[n,0:2][1])
        A[n,corr_single_i] = 1
        A[n,corr_single_j] = 1
        A_phase[n,corr_single_i] = 1
        A_phase[n,corr_single_j] = -1

    sum_counts_new = np.append(np.array([0]), sum_counts)
    for i in range(len(sum_counts_new) - 1):
        A[sum_counts_new[i]:sum_counts_new[i+1],Ndish+i] = 1
        A_phase[sum_counts_new[i]:sum_counts_new[i+1],Ndish+i] = 1
    constr_sum = np.append(np.ones(Ndish),np.zeros(N_unique_bls))
    constr_x_orient = np.append(x[:Ndish],np.zeros(N_unique_bls))
    constr_y_orient = np.append(y[:Ndish],np.zeros(N_unique_bls))
    A_no_constr = A
    A_no_constr_phase = A_phase
    A = np.vstack((A, constr_sum))
    A_phase = np.vstack((A_phase, constr_sum, constr_x_orient, constr_y_orient))
    temp = np.linalg.pinv(np.matmul(A_no_constr.T, A_no_constr))
    temp_phase = np.linalg.pinv(np.matmul(A_no_constr_phase.T, A_no_constr_phase))
    error_ID_noise_gain = np.sqrt(np.diag(temp))[:Ndish]
    error_ID_noise_phase = np.sqrt(np.diag(temp_phase))[:Ndish]
    return A_no_constr, A, A_no_constr_phase, A_phase, temp, error_ID_noise_phase, error_ID_noise_gain

def Noise_cov_matrix(manager_config_file,measured_vis,sigma):
    t = manager_config_file.telescope
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_matrix_info = Bls_counts(manager_config_file)[0]
    Nbls, _= bl_matrix_info.shape
    N_cov = np.array([])
    for i in measured_vis:
        N_cov = np.append(N_cov,sigma**2/(np.abs(i))**2)
    N =  np.diag(N_cov)
    N_no_constr = np.diag(N_cov)
    N_amp = np.vstack((N,np.zeros(Nbls)))
    N_phase = np.vstack((N,np.zeros((3,Nbls))))
    #print (N.shape)
    zeros = np.zeros((Nbls+1,1))
    zeros_phase = np.zeros((Nbls+3,3))
    N_amp = np.hstack((N_amp,zeros))
    N_phase = np.hstack((N_phase,zeros_phase))
    N_amp[-1][-1] = 1.
    N_phase[-1][-1] = N_phase[-2][-2] = N_phase[-3][-3]=1.
    return N_no_constr, N_amp, N_phase


def lstsq(manager_config_file, A_err,A_rec,N_err,N_rec,mv):
    t = manager_config_file.telescope
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    x_lstsq_t1 = np.matmul(A_rec.T,np.linalg.pinv(N_rec))
    x_lstsq_t1 = np.linalg.pinv(np.matmul(x_lstsq_t1,A_rec))
    x_lstsq_t2 = np.matmul(A_rec.T, np.linalg.pinv(N_rec))
    x_lstsq_t2 = np.matmul(x_lstsq_t2, mv)
    x_rec_real = np.matmul(x_lstsq_t1, x_lstsq_t2)
    return x_rec_real

def Logcal_solutions(manager_config_file,meas_vis_no_noise,true_vis, gain,meas_vis, sigma):
    t = manager_config_file.telescope
    x = t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y = t.feedpositions[:,1]
    Nfeeds, _ = t.feedpositions.shape
    Ndish = np.int(Nfeeds/2)
    bl_matrix_info = Bls_counts(manager_config_file)[0]
    Nbls, _= bl_matrix_info.shape
    sum_counts = Bls_counts(manager_config_file)[1]
    N_unique_bls = len(sum_counts)
    gain_amp = np.log(gain).real
    constr_amp_sum = np.sum(gain_amp)
    mv_real, mv_real_no_noise = log(np.abs(meas_vis)), log(np.abs(meas_vis_no_noise))
    mv_real, mv_real_no_noise = np.append(mv_real,constr_amp_sum), np.append(mv_real_no_noise,constr_amp_sum)
    tv_real = log(np.abs(true_vis))
    A_no_constr, A, A_no_constr_phase, A_phase, _, _, _ = A_matrix(manager_config_file)
    N_no_constr, N, N_phase = Noise_cov_matrix(manager_config_file,meas_vis,sigma)
    x_rec_real = lstsq(manager_config_file, A_no_constr, A, N_no_constr, N, mv_real)
    x_true_real = np.append(gain_amp, tv_real)
    x_rec_real_no_noise = np.linalg.lstsq(A, mv_real_no_noise, rcond=None)[0]
    mv_real_recovered = np.matmul(A, x_rec_real)
    mv_real_recovered_no_noise = np.matmul(A, x_rec_real_no_noise)
    return x_rec_real, x_rec_real_no_noise, x_true_real

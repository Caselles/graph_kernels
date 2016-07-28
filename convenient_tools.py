# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:45:22 2016

@author: Syzygy
"""

#Convenient tools when working on kernels on graphs

import numpy as np
import math

def kernelm_to_distancem(kernel_matrix):
    """Compute distance matrix associated to kernel matrix in argument"""
    n=len(kernel_matrix)
    distance_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i,j]=kernel_matrix[i,i]+kernel_matrix[j,j]-2*kernel_matrix[i,j]
    return distance_matrix

def distancem_to_affinitym(distance_matrix, beta):
    """Compute affinity matrix associated to distance matrix in argument, using beta as coefficient"""
    affinity_matrix=np.exp(-beta * distance_matrix / distance_matrix.std())
    return affinity_matrix

def normalize_kernel_matrix(K):
    """Compute normalized kernel matrix : K_norm[i,j]=K[i,j]/(sqrt(K[i,i]*K[j,j]))"""
    K_norm=np.zeros((len(K),len(K)))
    for i in range(len(K)):
        for j in range(len(K)):
            K_norm[i,j]=K[i,j]/(math.sqrt(K[i,i]*K[j,j]))
    return K_norm

def save_matrix(filename,matrix):
    """Saves matrix as filename.py in working directory"""
    np.save(filename+".npy", matrix)
    return('Matrix saved into working directory. Cheers mate !')

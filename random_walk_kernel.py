# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:45:22 2016

@author: Syzygy
"""

#Random walk kernel for graphs


import numpy as np
from scipy.sparse import lil_matrix, kron,identity
from scipy.sparse.linalg import lsqr
import networkx as nx
import time 


lmb=0.5
tolerance=1e-8
maxiter=20


def norm(adj_mat):
    """Normalize adjacency matrix"""
    norm = adj_mat.sum(axis=0)
    norm[norm == 0] = 1
    return adj_mat / norm

def random_walk_kernel(g1, g2):
    """Compute random walk kernel for networkx graphs g1 and g2 according to formula (16) of Graph Kernels"""
    start_time = time.time()
    g1=nx.adj_matrix(g1)
    g2=nx.adj_matrix(g2)
    # norm1, norm2 - normalized adjacency matrixes
    norm1 = norm(g1)
    norm2 = norm(g2)
    # if graph is unweighted, W_prod = kron(a_norm(g1)*a_norm(g2))
    #w_prod représente W_x
    w_prod = kron(lil_matrix(norm1), lil_matrix(norm2))
    #starting_prob=stop_prob=q_x=p_x
    starting_prob = np.ones(w_prod.shape[0]) / (w_prod.shape[0])
    stop_prob = starting_prob
    # first solve (I - lambda * W_prod) * x = p_x => x=(I - lambda * W_prod)^(-1)*p_x
    A = identity(w_prod.shape[0]) - (w_prod * lmb)
    x = lsqr(A, starting_prob)
    #then multiply by transpose(q_x) to obtain kernel value
    res = stop_prob.T.dot(x[0])
    print("--- %s seconds (kernel computation time) ---" % (time.time() - start_time))
    # print float(len(data_2)*i + j)/float(N), "%"
    return res

def kernel_matrix_rw_no_memory(kernel, liste):
    """Compute kernel matrix for graphs in liste, using kernel in argument, no memory method"""
    start_time_all = time.time()
    n=len(liste)
    res = np.zeros((n,n))
    i=0
    j=0
    for communei in liste: 
        Gi=nx.read_gexf('you_dir/'+communei+'.gexf',
                   node_type=None, relabel=True, version='1.1draft')
        for communej in liste[:i+1]:
            Gj=nx.read_gexf('your_dir/'+communej+'.gexf',
                   node_type=None, relabel=True, version='1.1draft')
            K_Gi_Gj=random_walk_kernel(Gi, Gj)
            res[i,j]=K_Gi_Gj
            res[j,i]=res[i,j]
            j=j+1
        j=0
        i=i+1
    print("--- %s seconds (entire kernel matrix computation time) ---" % (time.time() - start_time_all))
    return res

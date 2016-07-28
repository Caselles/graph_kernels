# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:45:22 2016

@author: Syzygy
"""

import numpy as np
import scipy
import networkx as nx
import time 
from scipy.sparse import lil_matrix
import math

#Shortest path kernel for graphs

def get_max_path(list_graphs, unweight):
    """Finds the longest shortest path in graphs collection, in order to compute SP kernel next"""
    i=0
    maxi=0
    for commune in list_graphs:
        graph=nx.read_gexf('you_dir/'+commune+'.gexf',
                           node_type=None, relabel=True, version='1.1draft')
        graph=nx.adjacency_matrix(graph)
        floyd=scipy.sparse.csgraph.floyd_warshall(graph, directed=False,
                                                  return_predecessors=False, unweighted=unweight)
        maxi = max(maxi, (floyd[~np.isinf(floyd)]).max())
        i=i+1
        if i%100==0:
            print(i, "graphs done")
    print('Longest path was of length',maxi)
    return maxi


def compute_splom(maxpath, list_graphs, unweight):
    """Returns a matrix (shortest paths length occurence matrix) in which 
    element (i,j) represents the number of shortest paths of length j in graph i"""
    res = lil_matrix(np.zeros((len(list_graphs), maxpath+1)))
    i=0
    maxi=0
    for commune in list_graphs:
        graph=nx.read_gexf('your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        graph=nx.adjacency_matrix(graph)
        floyd=scipy.sparse.csgraph.floyd_warshall(graph, directed=False, 
                                                  return_predecessors=False, unweighted=unweight)
        #print(floyd)
        maxi = max(maxi, (floyd[~np.isinf(floyd)]).max())
        subsetter = np.triu(~(np.isinf(floyd)))
        ind = floyd[subsetter]
        accum = np.zeros(maxpath + 1)
        accum[:ind.max() + 1] += np.bincount(ind.astype(int))
        #normalization : divide by number of shortest path in graph
        #accum=accum/sum(accum)
        res[i] = lil_matrix(accum)
        i=i+1
        if i%100==0:
            print(i, "graphs done")
    print('Longest path was of length',maxi)
    return res

def shortest_path_kernel_matrix(max_path, list_graphs, unweight):
    """Compute shortest path kernel matrix for graphs in data (collection of graphs) 
    defined as scalar product between normalized shortest path length occurence vector of graph i and j.
    User can choose memory method, then specify data or list according to choice."""
    start_time_all=time.time()
    splom = compute_splom(max_path, list_graphs, unweight)
    res=np.asarray(splom.dot(splom.T).todense())
    print("--- %s seconds (entire kernel matrix computation time, no_memory method) ---" 
          % (time.time() - start_time_all))
    return res

def compute_shortest_path_2_categories_plus_predict(list_graphs_train_1, list_graphs_train_2, 
                                                    list_graphs_test_1, list_graphs_test_2, 
                                                    max_path, unweight):
    """For binary classification problems. Allows to use classification algorithms (SVM, kNN, etc)"""
    size_train=len(list_graphs_train_1)+len(list_graphs_train_2)
    size_test=len(list_graphs_test_1)+len(list_graphs_test_2)
    splom_train_1 = compute_splom(max_path, list_graphs_train_1, unweight)
    splom_train_2 = compute_splom(max_path, list_graphs_train_2, unweight)
    splom_train=np.concatenate([splom_train_1.todense(),splom_train_2.todense()])
    splom_train=lil_matrix(splom_train)
    ker=np.asarray(splom_train.dot(splom_train.T).todense())
    #see convenient tools
    ker_norm=normalize_kernel_matrix(ker)

    splom_test_1 = compute_splom(max_path, list_graphs_test_1, unweight)
    splom_test_2 = compute_splom(max_path, list_graphs_test_2, unweight)
    splom_test=np.concatenate([splom_test_1.todense(),splom_test_2.todense()])
    splom_test=lil_matrix(splom_test)
    aux=np.asarray(splom_test.dot(splom_test.T).todense())
    test=np.asarray(splom_test.dot(splom_train.T).todense())
    test_norm=np.zeros((size_test, size_train))
    for i in range(size_test):
        for j in range(size_train):
            test_norm[i,j]=test[i,j]/math.sqrt(aux[i,i]*ker[j,j])
            
    return ker, ker_norm, test, test_norm



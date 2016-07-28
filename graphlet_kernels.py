# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:57:41 2016

@author: Syzygy
"""

#Graphlet kernels (sampled k graphlet, sampled 3&4 graphlets, all  connected 3,4-graphlets, all connected 3,4,5-graphlets (and weights or normalize options))

import numpy as np
import networkx as nx
import time 
import itertools
import random
import math

def number_of_graphlets(size):
    """Number of all undirected graphlets of given size"""
    if size == 2:
        return 2
    if size == 3:
        return 4
    if size == 4:
        return 11
    if size == 5:
        return 34

def generate_graphlets(size):
    """Generates graphlet array from previously stored csv data"""
    if size == 3:
        return np.genfromtxt('/Users/Syzygy/workspace/Stage_Shanghai/3graphlets.csv',delimiter=',').reshape(4, 3, 3)
    elif size == 4:
        return np.genfromtxt('/Users/Syzygy/workspace/Stage_Shanghai/4graphlets.csv',delimiter=',').reshape(11, 4, 4)

def is_3star(adj_mat):
    """Check if a given graphlet of size 4 is a 3-star"""
    return (adj_mat.sum() == 10 and 4 in [a.sum() for a in adj_mat])

def _4_graphlet_contains_3star(adj_mat):
    """Check if a given graphlet of size 4 contains a 3-star"""
    return (4 in [a.sum() for a in adj_mat])

def compare_graphlets(am1, am2):
    """
    Compare two graphlets.
    """
    adj_mat1 = am1
    adj_mat2 = am2
    np.fill_diagonal(adj_mat1, 1)
    np.fill_diagonal(adj_mat2, 1)
    k = np.array(adj_mat1).shape[0]
    if k == 3:
        # the number of edges determines isomorphism of graphs of size 3.
        return np.array(adj_mat1).sum() == np.array(adj_mat2).sum()
    else:
        # (k-1) graphlet count determines graph isomorphism for small graphs
        # return (_count_graphlets(adj_mat1, k-1, graphlet3_array, None) ==
        #         _count_graphlets(adj_mat2, k-1, graphlet3_array, None)).all()
        if not np.array(adj_mat1).sum() == np.array(adj_mat2).sum():
            return False
        if np.array(adj_mat1).sum() in (4, 6, 14, 16):
            # 0, 1, 5 or 6 edges
            return True
        if np.array(adj_mat1).sum() == 8:
            # 2 edges - two pairs or 2-path
            return 3.0 in [adj_mat.sum() for adj_mat in adj_mat1] == 3.0 in [adj_mat.sum() for adj_mat in adj_mat2]
        if np.array(adj_mat1).sum() == 10:
            # 3 edges - 3-star, 3-path or 3-cycle
            sums1 = [adj_mat.sum() for adj_mat in adj_mat1]
            sums2 = [adj_mat.sum() for adj_mat in adj_mat2]
            if (is_3star(adj_mat1) + is_3star(adj_mat2))%2 == 1:
                return False
            if is_3star(adj_mat1) and is_3star(adj_mat2):
                return True
            return (1 in sums1) == (1 in sums2)
        if np.array(adj_mat1).sum() == 12:
            # 4 edges - a simple cycle or something containing 3-star
            return _4_graphlet_contains_3star(adj_mat1) ==  _4_graphlet_contains_3star(adj_mat2)

    return False

def graphlet_index(adj_mat, graphlet_array):
    """Return index to increment."""
    for i, g in enumerate(graphlet_array):
        if compare_graphlets(adj_mat, g):
            return i
    return -1

def count_graphlets(adj_mat, size, graphlet_array):
    adj_mat = adj_mat.todense()
    res = np.zeros((1, number_of_graphlets(size)))
    for subset in itertools.combinations(range(adj_mat.shape[0]), size):
        graphlet = (adj_mat[subset, :])[:, subset]
        res[0][graphlet_index(graphlet, graphlet_array)] += 1
    # print "returning ", res / sum(sum(res))
    return res / res.sum()

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def count_graphlets_sampling(adj_mat, size, graphlet_array, s):
    """Count all graphlets of given size"""
    adj_mat = adj_mat.todense()
    res = np.zeros((1, number_of_graphlets(size)))
    for i in range(s):
        #get random nodes that will form the graphlet
        subset=random_combination(range(adj_mat.shape[0]), size)
        #construct graphlet
        graphlet = (adj_mat[subset, :])[:, subset]
        #increment index that correspond to the graphlet created
        res[0][graphlet_index(graphlet, graphlet_array)] += 1
    return res

def computekgraphlet(k, list_graphs, s):
    """Computes k-graphlets kernel matrix, with s samples"""
    d1 = np.zeros((len(list_graphs), number_of_graphlets(k)))
    graphlet_array=generate_graphlets(k)
    for i, commune in enumerate(list_graphs):
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        graph=nx.adjacency_matrix(graph, weight=None)
        d1[i] = count_graphlets_sampling(graph, k, graphlet_array,s)
        #normalize by the number of graphlets
        d1[i]=d1[i]/sum(d1[i])
        if i%10==0:
            print(i,'graphs done')
    return d1.dot(d1.T)

def compute34graphlet(list_graphs, s):
    """Computes 3,4-graphlets kernel matrix, with s samples"""
    d1 = np.zeros((len(list_graphs), number_of_graphlets(3)+number_of_graphlets(4)))
    graphlet_array3=generate_graphlets(3)
    graphlet_array4=generate_graphlets(4)
    for i, commune in enumerate(list_graphs):
        #print(commune)
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        graph=nx.adjacency_matrix(graph, weight=None)
        d1[i] = np.concatenate((count_graphlets_sampling(graph, 3, graphlet_array3,s)[0],
                                count_graphlets_sampling(graph, 4, graphlet_array4,s)[0]))
        #normalize by the number of graphlets
        d1[i]=d1[i]/sum(d1[i])
        if i%100==0:
            print(i,'graphs done')
    return d1.dot(d1.T)

def findPaths(G,u,n):
    """Finds all the paths of length n starting from node u of graph G"""
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
    return paths

def count_all_connected_3graphlets(graph):
    """Establish distribution of all-connected 3-graphlet in graph"""
    res=[0]*2
    graph=nx.convert_node_labels_to_integers(graph)
    A=nx.adjacency_matrix(graph, weight=None)
    for node in graph.nodes():
        for path in findPaths(graph, node, 2):
            if A[path[0],path[2]]==1:
                res[0]=res[0]+1
                #print(path,'is connected graphlet which is a cycle')
            else:
                res[1]=res[1]+1
                #print(path,'is connected graphlet which is not a cycle')
    res[0]=res[0]/6
    res[1]=res[1]/2
    return res

def count_all_connected_4graphlets(graph):
    """Establish distribution of all-connected 4-graphlet in graph"""
    res=[0]*6
    graph=nx.convert_node_labels_to_integers(graph)
    A=nx.adjacency_matrix(graph, weight=None)
    for node in graph.nodes():
        for path in findPaths(graph, node, 3):
            aux=A[path[0],path[2]]+A[path[0],path[3]]+A[path[1],path[3]]
            if aux==3:
                #6 edges : type 1 connected 4graphlet (complete)
                res[0]=res[0]+1
                #print('aux vaut 3!!!')
            elif aux==2:
                #5 edges : type 2 connected 4graphlet
                res[1]=res[1]+1
                #print('aux vaut 2!')
            elif aux==1:
                #4 edges : either of type 3 or 5 connected 4graphlet
                if A[path[0],path[3]]==1:
                    #then type 5 connected 4graphlet
                    res[4]=res[4]+1
                else:
                    #then type 3 connected 4graphlet
                    res[2]=res[2]+1
            else:
                #3 edges : type 6 connected 4graphlet
                res[5]=res[5]+1
        #now we have to count 3-stars
        if graph.degree(node)>2:
            for subset in itertools.combinations(graph.neighbors(node), 3):
                if (A[subset[0],subset[1]]==0 
                and A[subset[1],subset[2]]==0 
                and A[subset[2],subset[0]]==0):
                    #then type 6 connected 4graphlet (3-star)
                    res[3]=res[3]+1
    w = [1/24, 1/12, 1/4, 1, 1/8, 1/2]
    res=[a*b for a,b in zip(res,w)]
    return res

def count_all_connected_5graphlets(graph):
    """Establish distribution of all-connected 5-graphlet in graph"""
    res=[0]*21
    graph=nx.convert_node_labels_to_integers(graph)
    A=nx.adjacency_matrix(graph, weight=None)
    for node in graph.nodes():
        for path in findPaths(graph, node, 4):
            sub=graph.subgraph([path[0],path[1],path[2],path[3],path[4]])
            aux=A[path[0],
                  path[2]]+A[path[0],
                             path[3]]+A[path[0],
                                        path[4]]+A[path[1],
                                                   path[3]]+A[path[1],
                                                              path[4]]+A[path[2],
                                                                         path[4]]            
            if aux==6:
                #10 edges : type 1 connected 5graphlet (complete)
                res[0]=res[0]+1
            elif aux==5:
                #9 edges : type 2 connected 5graphlet
                res[1]=res[1]+1
            elif aux==4:
                #if it has 8 edges, it can be either graphlet 3 or 4, 
                #which can be distinguished by looking at the minimum degree of the graphlet
                aux2=[sub.degree(path[0]),
                      sub.degree(path[1]),
                      sub.degree(path[2]),
                      sub.degree(path[3]),
                      sub.degree(path[4])]
                if 2 in aux2:
                    #then type 4
                    res[3]=res[3]+1
                else:
                    #then type 3
                    res[2]=res[2]+1
            elif aux==3:
                #if the graphlet has 7 edges, it can be of type 5, 6, 9, or 14
                aux2=sorted([sub.degree(path[0]),
                           sub.degree(path[1]),
                           sub.degree(path[2]),
                           sub.degree(path[3]),
                           sub.degree(path[4])])
                if aux2[0]==1:
                    #then type 9
                    res[8]=res[8]+1
                elif aux2[1]==3:
                    #then type 5
                    res[4]=res[4]+1
                elif aux2[2]==2:
                    #then type 14
                    res[13]=res[13]+1
                else:
                    #then type 6
                    res[5]=res[5]+1
            elif aux==2:
                aux1=[sub.degree(path[0]),
                      sub.degree(path[1]),
                      sub.degree(path[2]),
                      sub.degree(path[3]),
                      sub.degree(path[4])]
                aux2=sorted(aux1)
                if aux2[0]==1:
                    if aux2[2]==2:
                        #then type 16
                        res[15]=res[15]+1
                    else:
                        #then type 10
                        res[9]=res[9]+1
                elif aux2[3]==2:
                    #then type 11
                    res[10]=res[10]+1
                else:
                    aux1=np.array(aux1)
                    ind=np.where(aux1 == 3)[0]
                    if A[path[ind[0]],[path[ind[1]]]]==1:
                        #then type 7
                        res[6]=res[6]+1
                    else:
                        #then type 15
                        res[14]=res[14]+1
            elif aux==1:
                aux1=[sub.degree(path[0]),
                      sub.degree(path[1]),
                      sub.degree(path[2]),
                      sub.degree(path[3]),
                      sub.degree(path[4])]
                aux2=sorted(aux1)
                if aux2[0]==2:
                    #then type 8
                    res[7]=res[7]+1
                elif aux2[1]==1:
                    #then type 18
                    res[17]=res[17]+1
                else:
                    aux1=np.array(aux1)
                    ind1=np.where(aux1 == 1)[0]
                    ind3=np.where(aux1 == 3)[0]
                    if A[path[ind1[0]],[path[ind3[0]]]]==1:
                        #then type 17
                        res[16]=res[16]+1
                    else:
                        #then type 12
                        res[11]=res[11]+1
            else:
                #then type 13
                res[12]=res[12]+1
                
        if graph.degree(node)>3:
            for subset in itertools.combinations(graph.neighbors(node), 4):
                a=[A[subset[0],subset[1]], A[subset[1],subset[2]],A[subset[2],subset[3]],A[subset[3],subset[0]]]
                if sum(a)==0:
                    #then type 21
                    res[20]=res[20]+1
                    
                elif sum(a)==1:
                    #then type 19
                    res[18]=res[18]+1
        #if graph.degree(node)>2:
            #for subset in itertools.combinations(graph.neighbors(node), 3):           
                          
                          
    w = [1/120, 1/72, 1/48, 1/36, 1/28, 1/20, 1/14, 1/10, 1/12, 
         1/8, 1/8, 1/4, 1/2, 1/12, 1/12, 1/4, 1/4, 1/2, 1,1/2,1]         
    res=[a*b for a,b in zip(res,w)]        
    return res

def compute_all_connected_34graphlet(list_graphs):
    """Computes all connected 3,4-graphlets kernel matrix, weight option"""
    start_time_all=time.time()
    d1 = np.zeros((len(list_graphs), 2+6))
    for i, commune in enumerate(list_graphs):
        #print(commune)
        graph=nx.read_gexf('your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        d1[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                count_all_connected_4graphlets(graph)))
        #normalize by the number of graphlets
        d1[i]=d1[i]/sum(d1[i])
        #print(d1[i])
        #w = [100,3,1000, 1000, 100, 10, 50, 2]
        #d1[i]=[a*b for a,b in zip(d1[i],w)]
        if i%100==0:
            print(i,'graphs done')
            print("--- %s seconds of computing, still running... ---" 
          % (time.time() - start_time_all))
    print("--- %s seconds (entire kernel matrix computation time) ---" 
          % (time.time() - start_time_all))    
    return d1.dot(d1.T)



def compute_all_connected_34graphlet_2_categories_plus_predict(list_graphs_train_1, 
                                                               list_graphs_train_2, 
                                                               list_graphs_test_1, 
                                                               list_graphs_test_2):
    """For binary classification"""
    start_time_all=time.time()
    size_train=len(list_graphs_train_1)+len(list_graphs_train_2)
    size_test=len(list_graphs_test_1)+len(list_graphs_test_2)
    d_train_1 = np.zeros((len(list_graphs_train_1), 2+6))
    d_train_2 = np.zeros((len(list_graphs_train_2), 2+6))
    d_test_1 = np.zeros((len(list_graphs_test_1), 2+6))
    d_test_2 = np.zeros((len(list_graphs_test_2), 2+6))
    #w=np.load('inv_freq.npy')
    w=np.ones(8)
    #w[1]=w[7]=0
    
    # for train
    
    for i, commune in enumerate(list_graphs_train_1):
        #print(commune)
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        d_train_1[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                       count_all_connected_4graphlets(graph)))
        #normalize by the number of graphlets
        d_train_1[i]=d_train_1[i]/sum(d_train_1[i])
        #w = [100,3,1000, 1000, 100, 10, 50, 2]
        d_train_1[i]=[a*b for a,b in zip(d_train_1[i],w)]
        if i%100==0:
            print(i,'graphs done')
            print("--- %s seconds of computing (train 1 phase) ---" 
          % (time.time() - start_time_all))
            
    for i, commune in enumerate(list_graphs_train_2):
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                           node_type=None, relabel=True, version='1.1draft')
        d_train_2[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                       count_all_connected_4graphlets(graph)))
        #normalize by the number of graphlets
        d_train_2[i]=d_train_2[i]/sum(d_train_2[i])
        #w = [100,3,1000, 1000, 100, 10, 50, 2]
        d_train_2[i]=[a*b for a,b in zip(d_train_2[i],w)]
        if i%100==0:
            print(i,'graphs done')
            print("--- %s seconds of computing (train 2 phase) ---" 
          % (time.time() - start_time_all))
            
    d_train=np.concatenate([d_train_1, d_train_2])
            
    ker=d_train.dot(d_train.T)
    #see convenient tools
    ker_norm=normalize_kernel_matrix(ker)
    print("--- %s seconds (entire train kernel matrix computation time) ---" 
          % (time.time() - start_time_all)) 
    
    # for test
    
    for i, commune in enumerate(list_graphs_test_1):
        #print(commune)
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        d_test_1[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                      count_all_connected_4graphlets(graph)))
        #normalize by the number of graphlets
        d_test_1[i]=d_test_1[i]/sum(d_test_1[i])
        #w = [100,3,1000, 1000, 100, 10, 50, 2]
        d_test_1[i]=[a*b for a,b in zip(d_test_1[i],w)]
        if i%100==0:
            print(i,'graphs done')
            print("--- %s seconds of computing (test 1 phase) ---" 
          % (time.time() - start_time_all))
            
    for i, commune in enumerate(list_graphs_test_2):
        graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                           node_type=None, relabel=True, version='1.1draft')
        d_test_2[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                      count_all_connected_4graphlets(graph)))
        #normalize by the number of graphlets
        d_test_2[i]=d_test_2[i]/sum(d_test_2[i])
        #w = [100,3,1000, 1000, 100, 10, 50, 2]
        d_test_2[i]=[a*b for a,b in zip(d_test_2[i],w)]
        if i%100==0:
            print(i,'graphs done')
            print("--- %s seconds of computing (test 2 phase) ---" 
          % (time.time() - start_time_all))
            
    d_test=np.concatenate([d_test_1, d_test_2])
            
    test=d_test.dot(d_train.T)
    
    aux=d_test.dot(d_test.T)
    test_norm=np.zeros((size_test,size_train))
    for i in range(size_test):
        for j in range(size_train):
            test_norm[i,j]=test[i,j]/math.sqrt(aux[i,i]*ker[j,j])
            
    return ker, ker_norm, test, test_norm

def compute_all_connected_345graphlet_2_categories_plus_predict(list_graphs_train_1, 
                                                               list_graphs_train_2, 
                                                               list_graphs_test_1, 
                                                               list_graphs_test_2):
                                                                   
    """For binary classification"""
    start_time_all=time.time()
    size_train=len(list_graphs_train_1)+len(list_graphs_train_2)
    size_test=len(list_graphs_test_1)+len(list_graphs_test_2)
    d_train_1 = np.zeros((len(list_graphs_train_1), 2+6+21))
    d_train_2 = np.zeros((len(list_graphs_train_2), 2+6+21))
    d_test_1 = np.zeros((len(list_graphs_test_1), 2+6+21))
    d_test_2 = np.zeros((len(list_graphs_test_2), 2+6+21))
    w=np.ones(29)
    #w=np.load('inv_freq_345.npy')
    #w[1]=w[7]=w[20]=0
    list_delete=[]
    
    # for train
    
    for i, commune in enumerate(list_graphs_train_1):
        
        try:
            
            #print(commune)
            graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                            node_type=None, relabel=True, version='1.1draft')
            d_train_1[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                           count_all_connected_4graphlets(graph),
                                           count_all_connected_5graphlets(graph)))
            #normalize by the number of graphlets
            d_train_1[i]=d_train_1[i]/sum(d_train_1[i])
            d_train_1[i]=[a*b for a,b in zip(d_train_1[i],w)]
            if i%100==0:
                print(i,'graphs done')
                print("--- %s seconds of computing (train 1 phase) ---" 
              % (time.time() - start_time_all))
                
        except IndexError:
            print(commune, 'does not work')
            list_delete.append(i)
            
    for i in list_delete:
        d_train_1 = np.delete(d_train_1, (i), axis=0)
        
    list_delete=[]
            
    for i, commune in enumerate(list_graphs_train_2):
        
        try:
            graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                               node_type=None, relabel=True, version='1.1draft')
            d_train_2[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                           count_all_connected_4graphlets(graph),
                                           count_all_connected_5graphlets(graph)))
            #normalize by the number of graphlets
            d_train_2[i]=d_train_2[i]/sum(d_train_2[i])
            d_train_2[i]=[a*b for a,b in zip(d_train_2[i],w)]
            if i%100==0:
                print(i,'graphs done')
                print("--- %s seconds of computing (train 2 phase) ---" 
              % (time.time() - start_time_all))
                
        except IndexError:
            print(commune, 'does not work')
            list_delete.append(i)
            
    for i in list_delete:
        d_train_2 = np.delete(d_train_2, (i), axis=0)
        
    list_delete=[]
            
    print('number of first label graphs in train :', len(d_train_1))
    print('number of second label graphs in train :', len(d_train_2))
    
    size_train=len(d_train_1)+len(d_train_2)
    
    d_train=np.concatenate([d_train_1, d_train_2])
    
    """The next comment line can be extremely useful !"""
    
    #d_train=(d_train-mean(d_train))/std(d_train)
            
    ker=d_train.dot(d_train.T)
    #see convenient tools
    ker_norm=normalize_kernel_matrix(ker)
    print("--- %s seconds (entire train kernel matrix computation time) ---" 
          % (time.time() - start_time_all)) 
    
    # for test
    
    for i, commune in enumerate(list_graphs_test_1):
        
        try:
            
            #print(commune)
            graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                            node_type=None, relabel=True, version='1.1draft')
            d_test_1[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                          count_all_connected_4graphlets(graph),
                                          count_all_connected_5graphlets(graph)))
            #normalize by the number of graphlets
            d_test_1[i]=d_test_1[i]/sum(d_test_1[i])
            d_test_1[i]=[a*b for a,b in zip(d_test_1[i],w)]
            if i%100==0:
                print(i,'graphs done')
                print("--- %s seconds of computing (test 1 phase) ---" 
              % (time.time() - start_time_all))
                
        except IndexError:
            print(commune, 'does not work')
            list_delete.append(i)
    
    for i in list_delete:
        d_test_1 = np.delete(d_test_1, (i), axis=0)
        
    list_delete=[]
            
    for i, commune in enumerate(list_graphs_test_2):
        
        try:
            
            graph=nx.read_gexf('/your_dir/'+commune+'.gexf', 
                               node_type=None, relabel=True, version='1.1draft')
            d_test_2[i] = np.concatenate((count_all_connected_3graphlets(graph),
                                          count_all_connected_4graphlets(graph),
                                          count_all_connected_5graphlets(graph)))
            #normalize by the number of graphlets
            d_test_2[i]=d_test_2[i]/sum(d_test_2[i])
            d_test_2[i]=[a*b for a,b in zip(d_test_2[i],w)]
            if i%100==0:
                print(i,'graphs done')
                print("--- %s seconds of computing (test 2 phase) ---" 
              % (time.time() - start_time_all))
                
        except IndexError:
            print(commune, 'does not work')
            list_delete.append(i)
            
    for i in list_delete:
        d_test_2 = np.delete(d_test_2, (i), axis=0)
    
    list_delete=[]
            
    print('number of first label graphs in test :', len(d_test_1))
    print('number of second label graphs in test :', len(d_test_2))
    
    size_test=len(d_test_1)+len(d_test_2)
            
    d_test=np.concatenate([d_test_1, d_test_2])
    
    """The next comment line can be extremely useful !"""
    
    #d_test=(d_test-mean(d_test))/std(d_test)
            
    test=d_test.dot(d_train.T)
    
    aux=d_test.dot(d_test.T)
    test_norm=np.zeros((size_test,size_train))
    for i in range(size_test):
        for j in range(size_train):
            test_norm[i,j]=test[i,j]/math.sqrt(aux[i,i]*ker[j,j])
            
    return ker, ker_norm, test, test_norm

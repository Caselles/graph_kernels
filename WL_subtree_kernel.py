# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:25:58 2016

@author: Syzygy
"""

import numpy as np
import networkx as nx
import time
import math
from collections import defaultdict
from copy import deepcopy

#WL subtree kernel for graphs


def compute_mle_wl_kernel(graph_list,h):
    """ Computes original WL kernel for a given height h. """
    
    
    start_time_mle = time.time()
    labels = {}
    label_lookup = {}
    label_counter = 0
    graph_idx = range(len(graph_list))
    num_graphs = len(graph_idx)
    orig_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in graph_idx} for it in range(-1, h)}
    # initial labeling
    for gidx in graph_idx:
        
        G=nx.read_gexf('/Users/Syzygy/workspace/Stage_Shanghai/gexf_simplified_france/'+graph_list[gidx]+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
        G=nx.convert_node_labels_to_integers(G)
        degrees = G.degree()  # this is a dictionary
        nx.set_node_attributes(G, 'label', degrees)
        
        
        labels[gidx] = np.zeros(G.order(), dtype = np.int32)
        #for node in graph_list[gidx].node:
        #for node in range(len(graph_list[gidx])):
        for i in range(len(G.node)):
            label = G.node[i]["label"]
            if not label in label_lookup:
                label_lookup[label] = label_counter
                labels[gidx][i] = label_counter
                label_counter += 1
            else:
                labels[gidx][i] = label_lookup[label]
            orig_graph_map[-1][gidx][label] = orig_graph_map[-1][gidx].get(label, 0) + 1
    compressed_labels = copy.deepcopy(labels)
    
    # WL iterations started
    for it in range(h):
        unique_labels_per_h = set()
        label_lookup = {}
        label_counter = 0
        for gidx in graph_idx:
            
            G=nx.read_gexf('/Users/Syzygy/workspace/Stage_Shanghai/gexf_simplified_france/'+graph_list[gidx]+'.gexf', 
                        node_type=None, relabel=True, version='1.1draft')
            G=nx.convert_node_labels_to_integers(G)
            degrees = G.degree()  # this is a dictionary
            nx.set_node_attributes(G, 'label', degrees)
            
            
            #for node in range(len(graph_list[gidx])):
            #print(gidx,'on en est la')
            for i in range(len(G.node)):
                node_label = tuple([labels[gidx][i]])
                neighbors=G.neighbors(i)
                #neighbors = graph_list[gidx][node]["neighbors"]
                if len(neighbors) > 0:
                    neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                    node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                if not node_label in label_lookup:
                    label_lookup[node_label] = label_counter
                    compressed_labels[gidx][i] = label_counter
                    label_counter += 1
                else:
                    compressed_labels[gidx][i] = label_lookup[node_label]
                orig_graph_map[it][gidx][node_label] = orig_graph_map[it][gidx].get(node_label, 0) + 1
        print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        labels = copy.deepcopy(compressed_labels)
        
    K = np.zeros((num_graphs, num_graphs))

    for it in range(-1, h):
        for i in range(num_graphs):
            for j in range(num_graphs):
                common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
                K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][j].get(k,0) for k in common_keys])
                
    end_time_mle_kernel = time.time()
    print("Total time for MLE computation for WL kernel (with kernel computation) %g"
                %(end_time_mle_kernel - start_time_mle))
    
    return K

def orig_graph_map_WL(graph_list_1, graph_list_2, h):
    """ Computes orig_graph_map for classes 1 and 2, for a given height h. """

    labels = {}
    label_lookup = {}
    label_counter = 0
    graph_idx = range(len(graph_list_1)+len(graph_list_2))
    orig_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in graph_idx} for it in range(-1, h)}
    idx_2=0
    #initial labeling
    for gidx in graph_idx:
        
        if gidx<len(graph_list_1):
        
            G=nx.read_gexf('/your_dir/'
                           +graph_list_1[gidx]+'.gexf', 
                            node_type=None, relabel=True, version='1.1draft')
            G=nx.convert_node_labels_to_integers(G)
            degrees = G.degree()  #this is a dictionary
            nx.set_node_attributes(G, 'label', degrees)
        
        
            labels[gidx] = np.zeros(G.order(), dtype = np.int32)
            #for node in graph_list[gidx].node:
            #for node in range(len(graph_list[gidx])):
            for i in range(len(G.node)):
                label = G.node[i]["label"]
                if not label in label_lookup:
                    label_lookup[label] = label_counter
                    labels[gidx][i] = label_counter
                    label_counter += 1
                else:
                    labels[gidx][i] = label_lookup[label]
                orig_graph_map[-1][gidx][label] = orig_graph_map[-1][gidx].get(label, 0) + 1
                
        else:
            
            
            G=nx.read_gexf('/your_dir/'
                           +graph_list_2[idx_2]+'.gexf', 
                            node_type=None, relabel=True, version='1.1draft')
            G=nx.convert_node_labels_to_integers(G)
            degrees = G.degree()  # this is a dictionary
            nx.set_node_attributes(G, 'label', degrees)
        
        
            labels[gidx] = np.zeros(G.order(), dtype = np.int32)
            #for node in graph_list[gidx].node:
            #for node in range(len(graph_list[gidx])):
            for i in range(len(G.node)):
                label = G.node[i]["label"]
                if not label in label_lookup:
                    label_lookup[label] = label_counter
                    labels[gidx][i] = label_counter
                    label_counter += 1
                else:
                    labels[gidx][i] = label_lookup[label]
                orig_graph_map[-1][gidx][label] = orig_graph_map[-1][gidx].get(label, 0) + 1
                
            idx_2=idx_2+1
            
    compressed_labels = deepcopy(labels)
    
    idx_2=0
    
    # WL iterations started
    for it in range(h):
        unique_labels_per_h = set()
        label_lookup = {}
        label_counter = 0
        idx_2=0
        
        for gidx in graph_idx:
            
            if gidx<len(graph_list_1):
                
            
                G=nx.read_gexf('/your_dir/'
                               +graph_list_1[gidx]+'.gexf', 
                            node_type=None, relabel=True, version='1.1draft')
                G=nx.convert_node_labels_to_integers(G)
                degrees = G.degree()  # this is a dictionary
                nx.set_node_attributes(G, 'label', degrees)
                
            
                #for node in range(len(graph_list[gidx])):
                for i in range(len(G.node)):
                    node_label = tuple([labels[gidx][i]])
                    neighbors=G.neighbors(i)
                    #neighbors = graph_list[gidx][node]["neighbors"]
                    if len(neighbors) > 0:
                        neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                        node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                    if not node_label in label_lookup:
                        label_lookup[node_label] = label_counter
                        compressed_labels[gidx][i] = label_counter
                        label_counter += 1
                    else:
                        compressed_labels[gidx][i] = label_lookup[node_label]
                    orig_graph_map[it][gidx][node_label] = orig_graph_map[it][gidx].get(node_label, 0) + 1
                    
            else:
                    
                    
                G=nx.read_gexf('/your_dir/'
                               +graph_list_2[idx_2]+'.gexf',
                               node_type=None, relabel=True, version='1.1draft')
                G=nx.convert_node_labels_to_integers(G)
                degrees = G.degree()  # this is a dictionary
                nx.set_node_attributes(G, 'label', degrees)
            
            
                #for node in range(len(graph_list[gidx])):

                for i in range(len(G.node)):
                    node_label = tuple([labels[gidx][i]])
                    neighbors=G.neighbors(i)
                    #neighbors = graph_list[gidx][node]["neighbors"]
                    if len(neighbors) > 0:
                        neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                        node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                    if not node_label in label_lookup:
                        label_lookup[node_label] = label_counter
                        compressed_labels[gidx][i] = label_counter
                        label_counter += 1
                    else:
                        compressed_labels[gidx][i] = label_lookup[node_label]
                    orig_graph_map[it][gidx][node_label] = orig_graph_map[it][gidx].get(node_label, 0) + 1
                    
                idx_2=idx_2+1
                    
                    
        print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        labels = deepcopy(compressed_labels)

    
    return orig_graph_map



def compute_WL_kernel_2_cat_plus_predict(list_graphs_train_1,list_graphs_train_2,
                                         list_graphs_test_1, list_graphs_test_2, h):
    """For binary classification"""
    
    size_train=len(list_graphs_train_1)+len(list_graphs_train_2)
    size_test=len(list_graphs_test_1)+len(list_graphs_test_2)
    
    list_1=np.concatenate([list_graphs_train_1, list_graphs_test_1])
    list_2=np.concatenate([list_graphs_train_2, list_graphs_test_2])
    
    start_time_mle = time.time()            
    
    the_map=orig_graph_map_WL(list_1, list_2, h)
    
    end_time_mle = time.time()
    print("Total time for MLE computation for WL kernel %g"
                %(end_time_mle - start_time_mle))
    
    X_train = np.zeros((size_train, size_train))
    X_test = np.zeros((size_test, size_train))
    util_norm = np.zeros((size_test, size_test))
    
    
    for it in range(-1, h):
        for i in range(size_train):
            if i%100==0 and i>0:
                print('iteration', it, 'ligne', i, 'pour X_train')
            for j in range(size_train):
                common_keys = set(the_map[it][i].keys()) & set(the_map[it][j].keys())
                X_train[i][j] += sum([the_map[it][i].get(k,0)*the_map[it][j].get(k,0) for k in common_keys])

    for it in range(-1, h):
        for i in range(size_test):
            if i%100==0 and i>0:
                print('iteration', it, 'ligne', i, 'pour X_test')
            for j in range(size_train):
                common_keys = set(the_map[it][i+size_train].keys()) & set(the_map[it][j].keys())
                X_test[i][j] += sum([the_map[it][i+size_train].get(k,0)*the_map[it][j].get(k,0) 
                                         for k in common_keys])
                
    for it in range(-1, h):
        for i in range(size_test):
            if i%100==0 and i>0:
                print('iteration', it, 'ligne', i, 'pour X_test')
            for j in range(size_test):
                common_keys = set(the_map[it][i+size_train].keys()) & set(the_map[it][j+size_train].keys())
                util_norm[i][j] += sum([the_map[it][i+size_train].get(k,0)*the_map[it][j+size_train].get(k,0) 
                                         for k in common_keys])
                
    end_time = time.time()
    print("Total time %g"%(end_time - start_time_mle))
                
    return X_train, X_test, util_norm

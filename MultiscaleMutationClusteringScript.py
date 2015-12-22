"""
Our Mutliscale Mutation Clustering Algorithm has been made available to the public for research and academic purposes.
For more information please see our manuscript, __________________
To view results from this algorithm on an interactive webpage, visit _________________
If you wish to use this algorithm for commercial uses, please contact _________________

"""

#Run on Python 2.7 using the Scipy and Numpy libraries.

import M2CFunctions as MICO #Multiscale IC Optimizer
from numpy import *
"""
#Multiscale Mutation Clustering Algorithm
#INPUTS:
#Data: raw data (point mutations) of a given gene or sequence
#noise_estimate: initial estimate the percent of the data which is noise. Recomended value: % 0f the total mutations which are synonymous
#OPTIONAL INPUTS AND PARAMETERS:
#interval: The interval over which teh data is defined. Defaults to [min(data), max(data)]
#bandwidths: underlying bandwidths to use for the mixture model. For robust results, these should densely cover the range of feature scales
#            expected in the data. The default setting is bandwidths = 
#            [2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 22, 26, 30, 38, 46, 54, 62, 78, 94, 110, 126, 158, 190, 222, 254, 318, 382, 446]
#            WARNING: Using a sparse set of bandwidths (for example, powers of 2) could result in the algorithm running slowly or using very large amounts of memory.
#min_mut_count: The minimum number of mutations in a cluster. This acts as a filter before clusters from different bandwidths are merged. 
#noise_bias: a probability threshold between 0 and 1 for a data point to be assigned to a gaussian cluster instead of to the noise cluster
#            for example, if noise_bias is .5, the probability of a cluster emitting a data point must be greater than 50% for that point
#            to be assigned to the cluster.
#print_outs: whether or not to printout out additional information as the algorithm runs
#OUTPUTS: 
#Returns a tuple: (final_clusters, final_params, densities)
#final_clusters: a list L of sublists, C. Each sublist C represents a cluster and contains the raw data points assigned to that cluster. Noise cluster not returned
#final_params: a list L of tuples (w, m, s, n) representing gaussian clusters where
#              w is the weight of a cluster, m is the mean, s is the standard deviation, and n is the uniform noise weight
#densities: a list of numpy arrays, where each array is a KDE of one of the bandwidths passed into the M2C function.
"""
def M2C(data, noise_estimate, interval = None, bandwidths = None, min_mut_count = 15, noise_bias = None, print_outs = False):
    #By default calculates the interval from the data    
    if interval == None:
        interval = min(data), max(data)
    DMM = MICO.DMM(data = data, interval = interval, noise_bias = noise_bias)
    #Default Bandwidth settings
    if bandwidths == None:
        h_list = range(2, 6, 1)+range(6, 14, 2)+range(14, 30, 4)+range(30, 62, 8)+range(62, 126, 16)+range(126, 254, 32)+range(254, 510, 64)
    else:
        h_list = bandwidths
    
    
    cluster_list = []
    params_list = []
    density_list = []
    #Kernel Density Estimates and multiple bandwidth mixture model calculate
    for h in h_list:
        print "Evaluating KDE for h=", h
        weights, means, stdevs, CD, density, local_maxima = DMM.AutoRun(h = h, uniform_weight_factor = noise_estimate, extra_info = True, print_outs = print_outs)
        density_list.append(density)
        CL = DMM.GenerateClusterList(ClusterDict = CD)[:-1]
        cluster_list+=CL #Last cluster representing noise is omitted
        PL = [(weights[i], means[i], stdevs[i], weights[-1]) for i in range(len(means))]
        params_list+=PL #weights[-1] is the noise weight

        
    #Merge multiscale clusters by creating a cluster tree
    cluster_set, params_set = MICO.CreateClusterSet(cluster_list, params_list, min_mut_count = min_mut_count)
    if len(cluster_set)>1:
        print "Merging Mixture Models"
        if print_outs:
            print "Creating Cluster Tree"
        Root, node_list, leaf_index_cluster_dict= MICO.CreateHeiarchicalTree(cluster_set, params_set, interval)
        if print_outs:
            print "Flattenning Cluster Tree"
        leaves, IC = MICO.FlattenTree(Root, data, leaf_index_cluster_dict, interval, extra_info = False, print_outs = print_outs)
        all_leaves = [l for l in node_list if l.is_leaf()]
        leaf_intervals = [(min(leaf_index_cluster_dict[l.id][0]),max(leaf_index_cluster_dict[l.id][0])) for l in leaves if len(leaf_index_cluster_dict[l.id][0]) > 0]

        if print_outs:
            print "Checking Cluster Tree"
        all_leaves = [l for l in node_list if l.is_leaf()]
        leaves2, IC2 = MICO.TreeFixer(all_leaves, leaves, leaf_index_cluster_dict, data, interval, extra_info = False, print_outs = print_outs)
        
        if IC2 != None:
            IC = IC2
            
        leaf_intervals.sort()
        final_clusters = [leaf_index_cluster_dict[l.id][0] for l in leaves2]
        final_params = [leaf_index_cluster_dict[l.id][1][:-1] for l in leaves2]
    else:
        print "No Merging Necessary"
        final_clusters = cluster_set
        final_params = [p[:-1] for p in params_set]
    
    #Sort Results
    sort_list = [(min(final_clusters[i]), final_clusters[i], final_params[i]) for i in range(len(final_clusters))]
    sort_list.sort()
    sort_list.reverse()
    final_clusters = [item[1] for item in sort_list]
    final_params = [item[2] for item in sort_list]
    return final_clusters, final_params, density_list

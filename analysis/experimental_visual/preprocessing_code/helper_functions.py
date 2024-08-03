#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:28:25 2019

Collection of functions for preprocessing experimenal visual data.

@author: irisyoon
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
sys.path.append('/usr/local/lib/python3.7/site-packages')

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from itertools import combinations
from scipy import stats

from sklearn.cluster import DBSCAN


# ========== Some common inputs to the functions ==========
# dic: dictionary in directory "data". Should be a variable loaded from one of the following:
#      "AL_st.pkl", "AL_gray.pkl", "V1_st.pkl", or "V1_gray.pkl"
# raster: dictionary in directory "data". Should be a variable loaded from one of the
#       files ending in "___raster.pkl"
# stimulus: string indicating which stimulus we are studying. Should be one of the following:
#       "stimulus_1", ..., "stimulus_3", "gray_1", ... ,"gray_3"
# neuron_list: list of neurons that have been selected.
# =========================================================

# Select neurons
def select_neurons(st_dic, freq_dic, stimulus):
    num_neurons = len(freq_dic.keys())
    # neurons with 0 spike
    # show histogram of neurons with number of trials with 0 spikes
    
    print('1. Select neurons that fire at least once in N or more trials')
    count_0_list = []
    for i in range(num_neurons):
        count_0 = freq_dic[i][freq_dic[i]==0].count()[stimulus]
        count_0_list.append(count_0)
        
    plt.hist(count_0_list,bins=20)
    plt.title('Histogram of number of trials with 0 spike')
    plt.show()
    
    print('Select neurons that have nonzero spike in N or more trials. Select N')
    count_0_threshold = int(input())
    
    count_0 = np.array(count_0_list)
    index_neurons = np.where(count_0 <= count_0_threshold)[0]
    
    print('Number of selected neurons: ', len(index_neurons))
    
    # Remove neurons based on spike_count
    print('2. Select neurons based on total spike count')
    total_spike_count = []
    for i in range(num_neurons):
        total_count = freq_dic[i].sum(axis=0)[stimulus]
        total_spike_count.append(total_count)
       
    spike_count = [total_spike_count[i] for i in index_neurons] 
    plt.hist(spike_count)
    plt.title('Histogram of total number of spike counts') # total: summed over 20 trials
    plt.show()
    
    print('Select neurons whose total spike count is greater than M. Select M ')
    spike_count_threshold = int(input())
    
    neurons_idx = [i for i in index_neurons if total_spike_count[i] > spike_count_threshold]
    print('number of neurons selected: ', len(neurons_idx))
    
    # Remove neurons that fire "uniformly"
    print('3. Remove neurons that fire uniformly')
    
    time_stamps = create_time_stamps(st_dic, stimulus, neurons_idx)
    neurons_list = []
    for neuron in neurons_idx:
        s, p = stats.ks_2samp(time_stamps[neurons_idx.index(neuron)], list(range(426)))
        if p < 0.05:
            neurons_list.append(neuron)
            
    print('number of neurons selected: ', len(neurons_list))
    
    return neurons_list
      

# Create time stamps of selected neurons
def create_time_stamps(dic, stimulus, neuron_list):
    
    time_stamps = []
    for neuron in neuron_list:
        neuron_ts = []
        for item in dic[neuron][stimulus]:
            for time in item[0]:
                neuron_ts.append(time)
        time_stamps.append(neuron_ts)
    return time_stamps

# ==========================================================================
# Updated neuron selection
# ==========================================================================
# Select neurons according to the following
# 1. reliability score
# 2. average standard deviation (and average standard deviation variation)    

def compute_reliability(raster, stimulus, neuron, limit_len = 25):
    # compute how reliability a neuron spikes across trials
    # Given a neuron and it's spike data in trial i and j
    # compute its cross correlogram (for a limited amount of time displacement)
    # compute the average of reliability scores across trials
    
    n_neurons, n_bins, n_trials = raster[stimulus].shape
    
    score_trials = np.zeros((20,20))

    for (i,j) in combinations(range(20),2):
        spikes1 = list(raster[stimulus][neuron,:,i])
        spikes2 = list(raster[stimulus][neuron,:,j])

        if sum(spikes1) == 0 or sum(spikes2) ==0:
            score = 0
        else:
            x_corr = np.correlate(spikes1, spikes2, 'full')
    
            #normalize 
            norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
            corr_norm = [x/norm_factor for x in x_corr]
            score = sum(corr_norm[n_bins-limit_len: n_bins+limit_len])
    
        score_trials[i,j] = score

    return np.average(score_trials)

def compute_avg_std(dic, raster, stimulus, neuron, neurons_idx, eps, min_samples):
    # up to this point, default was eps : 40, min_samples : 20
    # one can use max_std if one wishes
    n_neurons, n_bins, n_trials = raster['stimulus_1'].shape
    
    timestamps = create_time_stamps(dic, stimulus, [neuron])[0]
    neuron_ts = np.array(timestamps).reshape(-1,1)
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(neuron_ts)

    spike_cluster = {}
    label_values = list(set(clustering.labels_))
    for label in label_values:
        spike_cluster[label] = np.zeros((20, n_bins))

    for trial in range(20):
        spike_times = list(dic[neuron].at[trial,stimulus][0])
        for time in spike_times:
            time_bin = int(time)
            time_idx = timestamps.index(time)
            for label in label_values:
                if clustering.labels_[time_idx] == label:
                    spike_cluster[label][trial, time_bin ] += 1
                
# compute standard deviation of spike times in each cluster
# compute average std, while excluding cluster -1 (noise)
# if -1 is the only cluster that exists (that is, all spikes are classified as noice)

    spike_indices = {}
    ts_clustered = {}
    std = {}
    for label in label_values:
        spike_indices[label] = [idx for idx, x in enumerate(list(clustering.labels_)) if x == label]
        ts_clustered[label] = [list(neuron_ts)[x][0] for x in spike_indices[label]] 
        if label != -1:
            std[label] = np.std(ts_clustered[label])
    avg_std = np.average(list(std.values()))
    if len(list(std.values())) == 0:
        max_std = 0
    else:
        max_std = max(list(std.values()))
    
    return spike_cluster, avg_std, max_std



# Note: To visualize the clustering results, run the above "compute_avg_std" 
# Then run the following    
#for label in label_values:
#    plt.figure(figsize = (20, 5))
#    plt.title('spikes in cluster %d' %label)
#    plt.imshow(spike_cluster[label], cmap = 'Greys')



def compute_avg_std_var(dic, raster, stimulus, neuron, neurons_idx, boundary_limit = 50):
    n_neurons, n_bins, n_trials = raster['stimulus_1'].shape
    
    timestamps = create_time_stamps(dic, stimulus, [neuron])[0]
    ts_2 = [x for x in timestamps if x > boundary_limit and x < n_bins - boundary_limit]
    neuron_ts = np.array(ts_2).reshape(-1,1)
    clustering = DBSCAN(eps = 40, min_samples = 20).fit(neuron_ts)

    spike_cluster = {}
    label_values = list(set(clustering.labels_))
    for label in label_values:
        spike_cluster[label] = np.zeros((20, n_bins))

    for trial in range(20):
        spike_times = list(dic[neuron].at[trial,stimulus][0])
        for time in spike_times:
            if time > boundary_limit and  time < n_bins - boundary_limit:
                time_bin = int(time)
                time_idx = ts_2.index(time)
                for label in label_values:
                    if clustering.labels_[time_idx] == label:
                        spike_cluster[label][trial, time_bin ] += 1
                
# compute standard deviation of spike times in each cluster
# compute average std, while excluding cluster -1 (noise)
    spike_indices = {}
    ts_clustered = {}
    std = {}
    for label in label_values:
        spike_indices[label] = [idx for idx, x in enumerate(list(clustering.labels_)) if x == label]
        ts_clustered[label] = [list(neuron_ts)[x][0] for x in spike_indices[label]] 
        if label != -1:
            std[label] = np.std(ts_clustered[label])
    avg_std = np.average(list(std.values()))
    
    return spike_cluster, avg_std



# create raster
def create_raster(st_dic, num_timestamps, stimulus_list, bin_interval = 1):
    # ----- input -----
    # st_dic: dictionary of spike times. Usually AL_st or V1_st
    # stimulus_list: list of stimulus. This will be the keys to the returned dictionary
    # bin_interval: time interval corresponding to one timebin. Defaults to 1
    # one timebin corresponds to interval * 1 timestamp.
    # ex) if interval =0.8, then 1 timebin = 0.8 timestamp
    
    #----- output -----
    # dictionary with key values given by 'stimulus_list'
    # raster[stimulus] is a numpy array 
    #---------------------------------
    num_neurons = len(st_dic.keys())
    num_ts2 = int(num_timestamps/bin_interval) + 1
    num_trials = 20     
    
    raster = {}
    for stimulus in stimulus_list:
        raster_array = np.zeros((num_neurons, num_ts2, num_trials))

        for neuron in range(num_neurons):
            for trial in range(num_trials):
                spike_times = list(st_dic[neuron].at[trial,stimulus][0])
                for time in spike_times: 
                    time_bin = int(time/bin_interval)
                    raster_array[neuron, time_bin, trial] += 1

        raster[stimulus] = raster_array
    return raster

# Visualize raster
def show_raster(raster, stimulus, neuron_list, rotation = 0, save_fig=''):
    # ----- input -----
    # raster: dictionary 
    # stimulus: string
    # neuron_list: list of selected neuron index
    # rotataion: Take the first 'k' number of neurons, add to the end of raster
    #------------------
    
    print('rotation: ', rotation)
    if len(neuron_list) == 1:
        M = np.transpose(raster[stimulus][neuron_list[0],:,:])
        M_rot = np.concatenate((M[:,rotation:], M[:,:rotation]), axis = 1)
        plt.figure(figsize = (20,10))
        plt.imshow(M_rot, cmap = 'Greys')
        plt.title('neuron %d ' %neuron_list[0])
        stimulus_line = 0
        #while stimulus_line <450:
        #    plt.vlines(stimulus_line, 0, 20)
        #    stimulus_line += 53.25
        
        if save_fig != '':
            plt.savefig(save_fig)
        plt.show()
    else:
        fig_size = 2 * len(neuron_list)
        fig, ax = plt.subplots(len(neuron_list), figsize = (20,fig_size))
        for i in range(len(neuron_list)):
            M = np.transpose(raster[stimulus][neuron_list[i],:,:])
            M_rot = np.concatenate((M[:,rotation:], M[:,:rotation]), axis = 1)
            ax[i].imshow(M_rot, cmap = 'Greys')
            ax[i].set_title('neuron %d' %neuron_list[i])
            stimulus_line = 0
            #while stimulus_line < 450:
            #    ax[i].vlines(stimulus_line, 0, 20)
            #    stimulus_line += 53.25
        if save_fig != '':
            plt.savefig(save_fig)
            
        plt.show()
  
# Clustering
        
def k_medoid_cluster(distance_matrix):
    # perform k-medoid clustering with given distance matrix 
    # for number of clusters between 2 <= 10, compute silhouette score to get the optimal number of clusters
        
    num_clusters =range(2,10)
    silhouette_scores = []
    num_neurons = distance_matrix.shape[0]
    for k in num_clusters:
        # select k random points as initial medoids
        initial_medoids = np.random.choice(num_neurons, k, replace=False)    
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()# Create time stamps of selected neurons

        clusters = kmedoids_instance.get_clusters()
        # turn cluster info into label data
        label = [0]* num_neurons
        for i in range(len(clusters)):
            for j in clusters[i]:
                label[j] = i
        # compute silhouette score
        s_score = silhouette_score(distance_matrix, label, metric = 'precomputed')
        silhouette_scores.append(s_score)
    return silhouette_scores

# visualize multidimensional embedding of the clustering result
def MDS_clustering(dist_matrix, clusters, neurons_idx):
    
    # dist_matrix: distance matrix 
    # clusters; list of clustering information 
    
    embedding = MDS(n_components = 2, dissimilarity = 'precomputed')
    transformed = embedding.fit_transform(dist_matrix)
    
    num_clusters = len(clusters)
    colors_list = ['blue', 'red', 'orange', 'black']
    
    for item in range(num_clusters):
        plt.scatter(transformed[clusters[item],0],transformed[clusters[item],1], 
                    color = colors_list[item], label = 'cluster %d' %item)
        for i, txt in enumerate(clusters[item]):
            plt.annotate(neurons_idx[txt], (transformed[clusters[item][i],0],transformed[clusters[item][i],1]))
    
    plt.legend(loc = 'upper right')
    plt.show()
    
# peform k-medoid clustering and visualize using MDS
def clustering_vis(dist_matrix, num_clusters, neurons_idx, raster, stimulus):
    initial_medoids = np.random.choice(len(neurons_idx), num_clusters, replace=False)    
    kmedoids_instance = kmedoids(dist_matrix, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    
    cluster = {}
    for i in range(num_clusters):
        cluster[i] = [neurons_idx[j] for j in clusters[i]]
        
    MDS_clustering(dist_matrix, clusters, neurons_idx)
    
    for i in range(num_clusters):
        print('----------Raster for cluster ', i,'----------')
        show_raster(raster, stimulus, cluster[i])
    
    return clusters

def vis_embedding(dist_matrix, neurons_idx):
    
    # --- input ---
    # dist_matrix: distance matrix as numpy array
    # neurons_idx: list of neurons
    
    # --- output ---
    # plot of MDS embedding
    embedding = MDS(n_components = 2, dissimilarity = 'precomputed')
    transformed = embedding.fit_transform(dist_matrix)
    
    plt.scatter(transformed[:,0],transformed[:,1])
    for idx, neuron in enumerate(neurons_idx):
        plt.annotate(neuron, (transformed[idx,0],transformed[idx,1]))
    plt.show()
    

#=========================================================================
# Similarity and Distances
#=========================================================================

# limited cross correlogram
def windowed_xcorr_neuron(raster1, neuron1, raster2, neuron2, stimulus, limit_len, trial_num = None):
    # compute limited xcorr score among two neurons
    # note: the two neurons can be in the same region or be in different regions
    
    # raster_1: V1 raster or AL raster
    # neuron_1: index of V1 neuron
    # raster 2: V1 raster or AL raster
    # neuron_2: index of AL neuron
    # return a similarity score between two neurons
    
    n_neurons_1, n_bins_1, n_trials_1 = raster1[stimulus].shape 
    n_neurons_2, n_bins_2, n_trials_2 = raster2[stimulus].shape 
    
    n_bins = n_bins_1
    if n_bins_1 != n_bins_2:
        print("Error. Number of bins in two rasters do not match")
        
    if trial_num == None:
        spikes1 = list(raster1[stimulus].sum(axis = 2)[neuron1])
        spikes2 = list(raster2[stimulus].sum(axis = 2)[neuron2])
    else:
        spikes1 = list(raster1[stimulus][:,:,trial_num][neuron1])
        spikes2 = list(raster2[stimulus][:,:,trial_num][neuron2])

    x_corr = np.correlate(spikes1, spikes2, 'full')
    
    #normalize 
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    corr_norm = [x/norm_factor for x in x_corr]
    
    score = sum(corr_norm[n_bins-limit_len: n_bins+limit_len+1])

    return score
def windowed_xcorr(raster, stimulus, neurons_idx, limit_len, trial_num = None):
    # compute limited xcorr similarity and distance for neurons in one region (V1 or AL)
    
    # --- output ---
    # sim_score : array
    # distance : array
    # --------------
    n_neurons = len(neurons_idx)
    xcorr_array = np.zeros((n_neurons, n_neurons))
    
    for (i,j) in combinations(range(n_neurons),2):
        score = windowed_xcorr_neuron(raster, neurons_idx[i], raster, neurons_idx[j], stimulus, limit_len, trial_num = None)
            
        xcorr_array[i,j] = score
        xcorr_array[j,i] = score
    
    xcorr_array_scaled = xcorr_array/np.ceil(np.max(xcorr_array))
    distance = 1-xcorr_array_scaled
    np.fill_diagonal(distance, 0)    
    
    return xcorr_array, distance

def windowed_xcorr_V1_AL(raster_1, neurons_idx_1, raster_2, neurons_idx_2, stimulus, limit_len):
    # compute similarity and distance among neurons in two different regions  
    # --- input ---
    # raster_1 : raster from region 1
    # neurons_idx_1 : selected neurons from region 1
    # raster_2 : raster from region 2
    # neurons_idx_2: selectedn eurons from region 2
    
    
    # --- output ---
    # similarity_scaled: array of similarity scores. Scaled to 0-1
    # distance: array of distance among neurons. distance = 1 - similarity_scaled
    # The first n_neurons_1 rows and columns correspond to neurons in region 1
    # the last rows and columns correspond to neurons in region 2
    
    
    n_neurons_1 = len(neurons_idx_1)
    n_neurons_2 = len(neurons_idx_2)
    n_total = n_neurons_1 + n_neurons_2
    
    V1_AL_similarity = np.zeros((n_neurons_1, n_neurons_2))

    for (index, x) in np.ndenumerate(V1_AL_similarity):
        neuron_1 = index[0]
        neuron_2 = index[1]
    
        V1_AL_similarity[neuron_1, neuron_2] = windowed_xcorr_neuron(raster_1, neurons_idx_1[neuron_1], raster_2, neurons_idx_2[neuron_2],'stimulus_1', 50)

    similarity_1, distance_1 = windowed_xcorr(raster_1, 'stimulus_1', neurons_idx_1, 50)
    similarity_2, distance_2 = windowed_xcorr(raster_2, 'stimulus_1', neurons_idx_2, 50)

    similarity = np.zeros((n_total, n_total))
    similarity[:n_neurons_1, :n_neurons_1] = similarity_1
    similarity[:n_neurons_1, n_neurons_1:] = V1_AL_similarity
    similarity[n_neurons_1:,n_neurons_1:] = similarity_2
    similarity[n_neurons_1:, :n_neurons_1] = np.transpose(V1_AL_similarity)

    # scale similarity to 0-1
    similarity_scaled = similarity / np.ceil(np.max(similarity))
    
    # compute distance
    distance = 1-similarity_scaled
    np.fill_diagonal(distance,0)

    return np.ceil(np.max(similarity)), similarity_scaled, distance

#=========================================================================
# Average STD
#=========================================================================
def avg_std_neuron(timestamps, neuron, neuron_idx):
    
    # compute average standard deviation of a particular neuron
    
    neuron_ts = np.array(timestamps[neuron_idx.index(neuron)]).reshape(-1,1)
    clustering = AgglomerativeClustering(n_clusters = None, distance_threshold = 50, linkage='single').fit(neuron_ts)
    
    num_clusters = clustering.n_clusters_
    indices = {}
    ts_clustered = {}
    list_std = []

    for i in range(num_clusters):
        indices[i] = [j for j, x in enumerate(list(clustering.labels_)) if x == i]
        ts_clustered[i] = [list(neuron_ts)[x] for x in indices[i]]
        list_std.append(np.std(ts_clustered[i]))

    avg_std = np.average(list_std)
    return avg_std


def avg_std(timestamps, neurons_idx):
    # compute the avg std of all neurons in list neurons_idx
    
    avg_std = {}
    for neuron in neurons_idx:
        avg_std[neuron] = avg_std_neuron(timestamps, neuron, neurons_idx)
    
    return avg_std




    

#=========================================================================
# Persistence
#=========================================================================
    
def read_eirene_cycle(cycle_file):
    # From cycle information, turn into index information
    # want list of nodes consisting the cycle
    
    # --- input ---
    # cycle_file: path to text file of Eirene cycle
    # -------------
    
    # --- output ---
    # list of nodes consisting the cycle representative
    # NOTE: In Eirene, the nodes are indexed starting at 1
    # In our analysis, nodes are indexed starting at 0.
    # -------------
    
    cycle = np.loadtxt(cycle_file)
    eirene_index = []

    loc = (0,0)
    while cycle.size != 0:
        node = cycle[loc]
        if node not in eirene_index:
            eirene_index.append(node)

        # move up or down in the column 
        if loc[0] == 0:
            loc = tuple(map(sum, zip(loc, (1,0))))
        else:
            loc = tuple(map(sum, zip(loc, (-1,0))))

        node = cycle[loc]
        if node not in eirene_index:
            eirene_index.append(node)
        # nodes in the column have already been added 
        cycle = np.delete(cycle, loc[1], 1)
    
        # choose next location (different column)
        if cycle.size == 0:
            break
        else:
            loc = [tuple(x) for x in np.argwhere(cycle == node)][0]
    
    e_index = [int(x) for x in eirene_index]
    return e_index

 # visualize persistence cycle
# eirene_index : class representative from Eirene

def vis_persistence_cycle(dist_matrix, eirene_index, clusters, neurons_idx):
    # ---input---
    # dist_matrix: distance matrix
    # eirene_index: index of cycle representatives from Eirene
    # clusters
    # neurons_idx
    
    embedding = MDS(n_components = 2, dissimilarity = 'precomputed')
    transformed = embedding.fit_transform(dist_matrix)
    
    num_clusters = len(clusters)
    colors_list = ['blue', 'red', 'orange', 'black']

    cycle_idx = [x-1 for x in eirene_index]

    for i, item in enumerate(cycle_idx):
        plt.plot((transformed[item,0], transformed[cycle_idx[i-1],0]),( transformed[item,1], transformed[cycle_idx[i-1],1]),'-')

    for item in range(num_clusters):
        plt.scatter(transformed[clusters[item],0],transformed[clusters[item],1], 
                    color = colors_list[item], label = 'cluster %d' %item)
        for i, txt in enumerate(clusters[item]):
            plt.annotate(neurons_idx[txt], (transformed[clusters[item][i],0],transformed[clusters[item][i],1]))

    plt.legend(loc = 'upper right')
    plt.show()  


# visualize persistence cycle and show raster

def vis_persistence_cycle_raster(dist_matrix, clusters, neurons_idx, raster, stimulus, cycle_file):
    eirene_index = read_eirene_cycle(cycle_file)
    cycle_idx = [x-1 for x in eirene_index]
    vis_persistence_cycle(dist_matrix, eirene_index, clusters, neurons_idx)
    
    print('Showing raster of cycle')
    show_raster(raster, stimulus, [neurons_idx[x] for x in cycle_idx])
    

def barcode(file_path, max_filt, save_plot = False):
    # read textfile of persistence intervals. Textfile is the result from clique-top, perseus
    # plot barcode
    
    # ---input---
    # file_path: path to textfile. Each line of textfile contains persistence pair 
    # max_filt: maximum filtration value
    
    with open(file_path) as f:
        pairs = f.read().splitlines()
        
    img_file = file_path[:-4]+'PH.png'
    
    for (index, x) in enumerate(pairs):
        left =int(x.split()[0])
        right = int(x.split()[1])
        if right == -1:
            right = max_filt
        
        plt.hlines(index,left, right)
    plt.yticks([])
    
    if save_plot == True:
        plt.savefig(img_file)
    plt.show()


#=========================================================================
# Visualize V1 and AL 
#=========================================================================
    
def vis_V1_AL(transformed, V1_neurons_idx, AL_neurons_idx, title, cycle_file ='', 
              cycle = [], cycle_loc = 'V1', witness_list = [], annotation = 'off', save_fig = ''):
    # In order to draw a consistent picture, run embedding in notebook before 
    # running this function
    
    # --- input ---
    # transformed : points transformed to 2d via MDS embedding. 
    # ****NOTE****: this transformed must have occurred from distance matrix
    # whose first rows and columns correspond to V1L
    # V1_neurons_idx, AL_neurons_idx: list of neurons in each region
    # title (str): title for image
    # cycle_file: path to cycle file (from Eirene)
    # cycle : list of neurons forming cycle
    # cycle_loc: whether cycle is in V1 or AL
    
    
    
    n_V1 = len(V1_neurons_idx)
    colors_list = ['#AABCDF', '#DFAAAA']
    label = ['landmarks: V1', 'witnesses: AL']
    
    
    # define clusters
    n_V1 = len(V1_neurons_idx)
    n_AL = len(AL_neurons_idx)
    n_total = n_V1 + n_AL

    cluster_V1= list(range(n_V1))
    cluster_AL = list(range(n_V1,n_total))

    clusters = []
    clusters.append(cluster_V1) 
    clusters.append(cluster_AL)
    
    # plot cycle
    if cycle_file !='':
        eirene_index = read_eirene_cycle(cycle_file)
        cycle_idx = [x-1 for x in eirene_index]

    if cycle != []:
        cycle_idx = cycle
    
    # cycle location
    if cycle_loc == 'AL':
            # plot cycle in AL
        cycle_idx = [x + n_V1 for x in cycle_idx]
        for i, item in enumerate(cycle_idx):
            plt.plot((transformed[item,0], transformed[cycle_idx[i-1],0]),( transformed[item,1], transformed[cycle_idx[i-1],1]),'-')
    else:
            # plot cycle in V1
        for i, item in enumerate(cycle_idx):
            plt.plot((transformed[item,0], transformed[cycle_idx[i-1],0]),( transformed[item,1], transformed[cycle_idx[i-1],1]),'-')
    
    # plot the points
    for item in range(2):
        plt.scatter(transformed[clusters[item],0],transformed[clusters[item],1], color = colors_list[item], label = label[item])
     
    # highlight the witnesses
    if witness_list != []:
        witness_list_AL = [x + n_V1 for x in witness_list]
        plt.scatter(transformed[witness_list_AL, 0], transformed[witness_list_AL,1], color ='#FF007F', label = 'witnesses for cycle')
    
    # annotate
    if annotation == 'on':
        for i, txt in enumerate(clusters[item]):
            if item == 0: 
            # annotating neurons in V1
               plt.annotate(V1_neurons_idx[txt], (transformed[clusters[item][i],0],transformed[clusters[item][i],1]))
            if item == 1:
            # annotating neurons in AL
                plt.annotate(AL_neurons_idx[txt-n_V1], (transformed[clusters[item][i],0],transformed[clusters[item][i],1]))
                
    plt.title(title)
    plt.legend()
    if save_fig !='':
        plt.savefig(save_fig)
    plt.show()

def plot_raster(Region_st, neuron, ax = None, show_stimulus_line = True):
    spike_times = [Region_st[neuron]["stimulus_1"][i].tolist()[0] for i in range(20)]
    ax = ax or plt.gca()
    ax.eventplot(spike_times, color = "black", linewidth = 3, linelengths = 1.2)
    stimulus_line = 0
    if show_stimulus_line == True:
        while stimulus_line <425:
            ax.vlines(stimulus_line, -1, 20, linestyles = "dashed", color = "grey")
            stimulus_line += 53.25
    ax.set_xlim((0, 426))
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

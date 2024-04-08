"""
Functions computing similarity & dissimilarity between spike trains

@author Iris Yoon
iris.hr.yoon@gmail.com
"""
import math
import numpy as np
import random
from itertools import combinations
from scipy.signal import correlate

    
# compute distance between neurons from raster 
def pair_similarity(raster, neuron1, neuron2, limit_len):
    n_bins = raster.shape[1]
    
    spikes1 = raster[neuron1,:]
    spikes2 = raster[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    return score

def compute_similarity(raster, limit_len):
    similarity = []
    n_neurons = raster.shape[0]
    for (i,j) in combinations(range(n_neurons),2):
        score = pair_similarity(raster, i, j, limit_len)
        similarity.append(score)
    
    return similarity

def cross_pair_similarity(raster1, neuron1, raster2, neuron2, limit_len):
    # compute similarity between neuron1 from raster1 and neuron2 from raster2
    
    if raster1.shape[1] != raster2.shape[1]:
        raise AssertionError('Rasters must have equal number of timebins')
    
    n_bins = raster1.shape[1]
    spikes1 = raster1[neuron1,:]
    spikes2 = raster2[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    
    return score

def compute_cross_distance(raster1, raster2, limit_len):
    """
    Computes distance matrix between two rasters. 

    Inputs
    ------
    raster1: (np.array) of shape (n_neurons1, n_timebins)
    raster2: (np.array) of shape (n_neurons1, n_timebins)
    limit_len: Int. Maximum displacement value (of timebins) to consider when computing cross correlation

    Outputs
    -------
    distance: (np.array) distance matrix between each pair of neurons in raster1 and raster2
    xcorr: (np.array) similarity matrix
    xcorr_scaled: (np.array) similarity matrix scaled by its maximum entry   
    """
    # check if the two rasters have the same number of time bins 
    if raster1.shape[1] != raster2.shape[1]:
        raise AssertionError('Rasters must have equal number of timebins')
    
    n_neurons1 = raster1.shape[0]
    n_neurons2 = raster2.shape[0]
    
    # compute cross-correlogram among neurons in raster1 and raster2
    xcorr = np.zeros((n_neurons1, n_neurons2))
    for neuron1 in tqdm(range(n_neurons1)):
        for neuron2 in range(n_neurons2):
            score = cross_pair_similarity(raster1, neuron1, raster2, neuron2, limit_len)
            xcorr[neuron1, neuron2] = score
               
    # scale the xcorr matrix
    xcorr_scaled = xcorr/np.ceil(np.max(xcorr))
    distance = 1-xcorr_scaled
    np.fill_diagonal(distance, 0)   
    
    return distance, xcorr, xcorr_scaled

def triu_entries_to_matrix(entries, size):
    # Given entires in the upper triangle of a symmetric matrix, return the matrix
    
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size, 1)] = entries
    return matrix + matrix.T


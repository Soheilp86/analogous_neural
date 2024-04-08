"""helper module
This module contains helper functions for processing spike trains

@author Iris Yoon
irishryoon@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.signal import correlate


# ------------------------------------------------------------------------
# Functions for visualizing spike trains
# ------------------------------------------------------------------------
def plot_spikes(raster, neurons, start = 0, end = 15000, figsize = (20, 10)):
    """Plot spike trains from selected neurons

    Parameters
    ----------
    raster: (array) of spike trains for system. Size: (n,m) where `n` is the number of neurons
    neurons: (list) index of neurons to plot
    start: (idx) beginning time bin
    end: (idx) end of time bin

    Returns
    -------
    None
    """
    n = len(neurons)
    fig, ax = plt.subplots(nrows = n, figsize = figsize)
    
    for i, neuron in enumerate(neurons):
        ax[i].plot(raster[neuron, start:end])
        ax[i].set_title("neuron %i" %neuron)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------
# Functions for computing similarity & distances between spike trains
# ------------------------------------------------------------------------

def pair_similarity(raster, neuron1, neuron2, limit_len):
    """Compute similarity between two neurons in a raster
    
    Parameters
    ----------
    raster: (array) of shape (n,m), where `n` is the number of neurons, and `m` is the number of bins
    neuron1: (int)
    neuron2: (int)
    limit_len: (int): When computing correlation, we allow two spike trains (spikes1, spikes2) to be displaced upto "limit_len"

    Returns
    -------
    score: (float) similarity score between two neurons. 
    """
    n_bins = raster.shape[1]
    
    spikes1 = raster[neuron1,:]
    spikes2 = raster[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    return score

def compute_similarity(raster, limit_len):
    """Compute similarity between every pair of neurons in a raster
    
    Parameters
    ----------
    raster: (array) of shape (n,m), where `n` is the number of neurons, and `m` is the number of bins
    limit_len: (int): When computing correlation, we allow two spike trains (spikes1, spikes2) to be displaced upto "limit_len"

    Returns
    -------
    similarity: (list) of similarity scores between each pair of neurons
    """
    similarity = []
    n_neurons = raster.shape[0]
    for (i,j) in combinations(range(n_neurons),2):
        score = pair_similarity(raster, i, j, limit_len)
        similarity.append(score)
    
    return similarity

def cross_pair_similarity(raster1, neuron1, raster2, neuron2, limit_len):
    """Compute similarity between two neurons in different systems (rasters)
    
    Parameters
    ----------
    raster1: (array) of shape (n1,m), where `n1` is the number of neurons in system 1, and `m` is the number of bins
    neuron1: (int) selected neuron in system 1
    raster2: (array) of shape (n2,m), where `n2` is the number of neurons in system 2, and `m` is the number of bins
    neuron2: (int) selected neuron in system 2
    limit_len: (int): When computing correlation, we allow two spike trains (spikes1, spikes2) to be displaced upto "limit_len"

    Returns
    -------
    score: (float) similarity score between selected neurons
    """
    
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
    """Compute cross-system distance between two systems (rasters)
    
    Parameters
    ----------
    raster1: (array) of shape (n1,m), where `n1` is the number of neurons in system 1, and `m` is the number of bins
    raster2: (array) of shape (n2,m), where `n2` is the number of neurons in system 2, and `m` is the number of bins
    limit_len: (int): When computing correlation, we allow two spike trains (spikes1, spikes2) to be displaced upto "limit_len"

    Returns
    -------
    distance: (array) of cross-system distance between neurons in raster1 and raster2
    xcorr: (array) of cross-system similarity (cross-correlation) between raster1 and raster2
    xcorr_scaled: (array) scaled version of xcorr. distance = 1 - xcorr_scaled
    """
    
    # check if the two rasters have the same number of time bins 
    if raster1.shape[1] != raster2.shape[1]:
        raise AssertionError('Rasters must have equal number of timebins')
    
    n_neurons1 = raster1.shape[0]
    n_neurons2 = raster2.shape[0]
    
    # compute cross-correlogram among neurons in raster1 and raster2
    xcorr = np.zeros((n_neurons1, n_neurons2))
    for neuron1 in range(n_neurons1):
        for neuron2 in range(n_neurons2):
            score = cross_pair_similarity(raster1, neuron1, raster2, neuron2, limit_len)
            xcorr[neuron1, neuron2] = score
               
    # scale the xcorr matrix
    xcorr_scaled = xcorr/np.ceil(np.max(xcorr))
    distance = 1-xcorr_scaled
    np.fill_diagonal(distance, 0)   
    
    return distance, xcorr, xcorr_scaled

def triu_entries_to_matrix(entries, size):
    # Given entires in the upper triangle of a symmetric matrix, return the matrix itself
    
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size, 1)] = entries
    return matrix + matrix.T


def scale_compute_distance(similarity, scale_factor):
    distance =  1- similarity / scale_factor
    return distance
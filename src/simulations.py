#!/usr/bin/env python

# Contains code that helps with simulations of V1 and downstream neurons

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tensorflow as tf
from itertools import combinations
from scipy.signal import correlate
from tick.base import TimeFunction
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson


def angular_distance(angle1, angle2, size = math.pi):
    """Compute distance between two points on S1 obtained by identifying the boundaries of [0, size]
    """
    
    # un-normalized distance between two angles (in radians) in S1 
    s = min(angle1, angle2)
    l = max(angle1, angle2)
    return min(l-s, s + size - l)

def location_distance_on_torus(loc1, loc2, size):
    """Compute distance between two locations on a square torus of shape(size, size) """
    loc1_x = loc1[0]
    loc1_y = loc1[1]
    
    loc2_x = loc2[0]
    loc2_y = loc2[1]
    
    # angular distance on x-coordinates
    distance_x = angular_distance(loc1_x, loc2_x, size = size)
    
    # angular distance on y-coordinates
    distance_y = angular_distance(loc1_y, loc2_y, size = size)

    return math.sqrt(distance_x**2 + distance_y**2)
               
def tuning_curve(theta_preferred, C, R_preferred, sigma, theta, size = math.pi):
    """A Gaussian orientation-runing curve.
    
    --- input ---
    theta_preferred: preferred orientation
    C: baseline rate
    R_preferred: above-baseline rate at preferred orientation
    sigma: tuning width
    theta: some orientation value
    
    --- output ---
    rate at theta
    """  
               
    return C + R_preferred * np.exp(-(angular_distance(theta_preferred,theta, size = size))**2/(2 * sigma**2))
                     
def nonlinear_tanh(g2, L0, r_max, array, r0):
    """Nonlinear function used in the linear-nonlinear model
    
    --- input ---
    g2: how rapidly the firing rate increases
    L0: threshold
    r_max: maximum firing rate
    array: array of dot products between stimulus and Gabor filters
    
    --- output ---
    Array after applying nonlinear_tanh function to every element
    """
    return r0 + r_max*np.maximum(0,np.tanh(g2*(array-L0)))
               
               
def simulate_neuron(rate, bin_size = 5):
    """Simulate the spike train of a single neuron with given rate using inhomogeneous Poisson process.
    Simulates the inhomogeneous Poisson process according to the rate array and return the spike train after binning according to bin_size.
    
   
    Parameters
    ----------
    rate: (arr) of shape (n_time, ). rate[i] is the rate for time interval [i, i+1)
    bin_size: (int) number of bins for one unit of time
                ex) If unit of time in the "rate" array is 1 second,
                then bin_size of 1000 indicates that 1 bin of the 
                resulting raster corresponds go 1/1000 second. There will be 1000 bins for 1 second interval  

    Returns
    -------
    spike_binary: (binary arr) of shape (n_time * bin_size, ) 
            spike_binary[i] = 1 if there exists a spike in bin i
    spike_count: (array of Int) of shape (n_time * bin_size, )
            spike_count[i] is the number of spikes that occurred in bin i
    spike_times: (array) of shape (m,), where m is the number of events. 
    """
    n_time = rate.shape[0]
    bin_unit = 1/bin_size # 1 bin corresponds to 1/bin_size seconds
    
    #Define inhomogeneous poisson process
    T = np.array([x for x in range(n_time)])
    tf = TimeFunction((T, np.ascontiguousarray(rate)))
    in_poi = SimuInhomogeneousPoisson([tf], end_time=n_time, verbose=False)

    # Activate intensity tracking and launch simulation
    in_poi.track_intensity(0.1)
    in_poi.threshold_negative_intensity(True)
    in_poi.simulate()

    # Update raster for neuron
    spike_times = in_poi.timestamps[0]
    spike_count, bin_edges = np.histogram(spike_times, bins = np.arange(0,n_time + bin_unit, bin_unit))
    
    spike_binary = (spike_count > 0).astype(int)

    return spike_binary, spike_count, spike_times

def simulate_raster(rate, bin_size): 
    """Simulate raster of a collection of neurons by running 'simulate_neuron' on every neuron.

    Parameters
    ----------
    rate: (arr) of shape (n_time, n_neurons)
    bin_size: (int) number of bins for one unit of time
                ex) If unit of time in the "rate" array is 1 second,
                then bin_size of 1000 indicates that 1 bin of the 
                resulting raster corresponds go 1/1000 second.    
    
    Returns
    -------
    binary_raster: (binary arr) of shape (n_neurons, n_time * bin_size)
            binary_raster[i][j] = 1 if there exists a spike in bin j for neuron i
    count_raster: (array of integers) of shape (n_neurons, n_time * bin_size)
            count_raster[i][j] is the number of spikes in bin j for neuron i
    """

    n_time, n_neurons = rate.shape
    n_bins = n_time * bin_size 
    bin_unit = 1/bin_size # 1 bin corresponds to 1/bin_size seconds
    
    binary_raster = np.zeros((n_neurons, n_bins))
    count_raster = np.zeros((n_neurons, n_bins))
    for neuron in range(n_neurons):
        spike_binary, spike_count, _  = simulate_neuron(rate[:,neuron], bin_size)
        binary_raster[neuron,:] = spike_binary
        count_raster[neuron,:] = spike_count
        
    return binary_raster, count_raster
           
               
def compute_true_firing_rates(n_neurons, orientations_list, C, R_preferred, sigma):
    # Computes the firing rate of orientation-sensitive neurons in response to a given list of orientations given the parameters of the tuning curve
    """
    --- input---
    n_neurons: (int) number of orientation-sensitive neurons. There will be `n_neurons` whose preferred orientation is equally dispersed around [0, pi] (considered as S1)
    orientations_list: (list of float) orientations of stimuli
    C: (float) parameter for function `tuning_curve`
    R_preferred: (float) parameter for function `tuning_curve`
    sigma: (float) parameter for function `tuning_curve`
    
    --- output ---
    true_rate: (arr) true_rate[i][j]: firing rate of jth orientation-sensitive neuron in response to orientations_list[i]
    """
    
    n_frames = len(orientations_list)
    true_rate = np.zeros((n_frames, n_neurons))
        
    # assume orientation-neurons are equally spaced
    orientations = [x * math.pi / n_neurons for x in range(n_neurons)]
        
    for i in range(n_frames):
        for neuron in range(n_neurons):
            neuron_orientation = orientations[neuron]
            st_orientation = orientations_list[i] 
    
            # compute the rate 
            true_rate[i, neuron] = tuning_curve(neuron_orientation, C, R_preferred, sigma, st_orientation)
    
    return true_rate
"""
def compute_true_firing_rates_with_noise(n_neurons, orientations_list, C, R_preferred, sigma):
    # Computes the firing rate of orientation-sensitive neurons in response to a given list of orientations given the parameters of the tuning curve

    #--- input---
    #n_neurons: (int) number of orientation-sensitive neurons. There will be `n_neurons` whose preferred orientation is equally dispersed around [0, pi] (considered as S1)
    #orientations_list: (list of float) orientations of stimuli
    #C: (float) parameter for function `tuning_curve`
    #R_preferred: (float) parameter for function `tuning_curve`
    #sigma: (float) parameter for function `tuning_curve`
    
    #--- output ---
    #true_rate: (arr) true_rate[i][j]: firing rate of jth orientation-sensitive neuron in response to orientations_list[i]

    
    n_frames = len(orientations_list)
    true_rate = np.zeros((n_frames, n_neurons))
        
    # assume orientation-neurons are equally spaced
    orientations = [x * math.pi / n_neurons for x in range(n_neurons)]
    orientations_perturbed = [i + np.random.normal(0, 0.05) for i in orientations]
    orientations_perturbed = [i if i >0 else i + math.pi for i in orientations_perturbed]
    
    # keep track of the perturbed parameters
    C_perturbed = []
    R_perturbed = []
    sigma_perturbed = []
    
    for neuron in range(n_neurons):
        C_noisy = C + np.random.exponential(1/8)
        R_preferred_noisy = R_preferred + np.random.exponential(1/2)
        sigma_noisy = sigma + np.random.exponential(1/16)
        
        C_perturbed.append(C_noisy)
        R_perturbed.append(R_preferred_noisy)
        sigma_perturbed.append(sigma_noisy)
        
        for i in range(n_frames):
            neuron_orientation = orientations_perturbed[neuron]
            st_orientation = orientations_list[i] 
    
            # compute the rate 
            true_rate[i, neuron] = tuning_curve(neuron_orientation, C_noisy , R_preferred_noisy, sigma_noisy, st_orientation)
    
    return true_rate, orientations_perturbed, C_perturbed, R_perturbed, sigma_perturbed
"""
    
def plot_spike_train(spike_binary, start, end, title ="", ax = None):
    # plot spike train from binary array from 'start' to 'end'
    
    ax = ax or plt.gca()
    spike_train = spike_binary[start:end]
    pos = np.argwhere(spike_train > 0).tolist()

    for spike in pos:
        ax.vlines(spike[0],0,1)
        ax.set_xlim(0, end-start)
    ax.set_title(title)
    return ax 
               
            


def vector_to_symmetric_matrix(vector, n):
    """Given a vector of entries from `compute_similarity` or `scale_compute_distance`, return the symmetric matrix.
    
    --- input ---
    vector: (list) of entries
    n: (int) size of matrix will be n x n.
    
    --- output ---
    D: (array) symmetric matrix whose non-diagonal entries correspond to `vector`
    idx: (list) indices of non-zero entries in the matrix
    """
    
    
    D = np.zeros((n,n))
    idx = []
    for i in range(n):
        for j in range(i+1,n):
            idx.append(j + i* n)
            

    for k, val in enumerate(vector):
        row = idx[k] // n 
        col = idx[k] % n 
        D[row, col] = val
    D = D + D.transpose()

    return D, idx

def predict_rates(X, y, model, scaler):
    """Given a trained model and a target-variable scaler, predict the firing rates.
    """
    
    # scale target variable
    y_scaled = scaler.transform(y)
    
    # predict
    y_pred = model.predict(X)
    y_pred_inv_scaled = scaler.inverse_transform(y_pred)

    # compute score (MSE)
    score = tf.keras.metrics.mean_squared_error(y_pred.flatten(), y_scaled.flatten()).numpy()
    
    return y_pred_inv_scaled, score
    
def add_noise(rate, noise_std):
    """Given a firing rate, add Gaussian noise and rectify negative rates to zero
    """
    
    noise = np.random.normal(0, noise_std, rate.shape)
    return np.maximum(rate + noise, 0) 
    
# need to edit the following -- we use the function for both orientation and direction-sensitive neurons
def compute_firing_rates(stimulus_orientations, neuron_orientations, C, R_preferred, sigma, size = math.pi):
    """Computes the firing rate of orientation (direction)-sensitive neurons in response to a given list of orientations given the parameters of the tuning curve

    --- input---
    stimulus_orientations: (list of float) orientations (directions) of stimuli
    neuron_orientations: (array) of preferred orientations (directions) of neurons
    C: (array) of baseline firing rate of neurons
    R_preferred: (array) of above-baseline parameter for function `tuning_curve` of neurons
    sigma: array of sigma parameter for function `tuning_curve`
    
    --- output ---
    true_rate: (arr) true_rate[i][j]: firing rate of jth orientation-sensitive neuron in response to orientations_list[i]
    """
    
    n_frames = len(stimulus_orientations)
    n_neurons = len(neuron_orientations)
    true_rate = np.zeros((n_frames, n_neurons))

        
    for neuron in range(n_neurons):
        
        orientation_neuron = neuron_orientations[neuron]
        C_neuron = C[neuron]
        R_neuron = R_preferred[neuron]
        sigma_neuron = sigma[neuron]
        
        for i in range(n_frames):

            st_orientation = stimulus_orientations[i] 
    
            # compute the rate 
            true_rate[i, neuron] = tuning_curve(orientation_neuron, C_neuron, R_neuron, sigma_neuron, st_orientation, size = size)
    
    return true_rate

def create_sampled_stimulus_raster(locations_sampled, orientations_sampled, video_locations, video_orientations, max_proximity = 6, image_size = 40):
    """Function that creates a binary "raster" from sampled stimulus images.
    For each stimulus image sampled (for computing stimulus barcode), create a binary array indicating whether the sampled image is present in the stimulus video at a specific frame.
    
    --- input ---
    locations_sampled: (list) of the locations of the sampled stimulus images
    orientations_sampled: (list) of the orientations of the sampled stimulus images
    video_locations: (array) of the locations of the video stimulus
    video_orientations: (array) of the orientations of the video stimulus
    
    --- output ---
    stimulus_raster: (array) of shape (n, f), where n (= len(locations_sampled)) is the number of stimulus images sampled and f ( = len(video_locations)) is the number of frames in stimulus video.
                    stimulus_raster[i,j] = 1 if sampled stimulus "i" is close enough to the stimulus image at frame "j"
    """
    n = len(locations_sampled)
    f = len(video_locations)
    stimulus_raster = np.zeros((n, f))
    
    for i in range(n):

        sampled_loc = locations_sampled[i]
        sampled_orientation = orientations_sampled[i]

        # find all frames that match the sampled orientation
        orientation_match = [idx for idx, ori in enumerate(video_orientations) if ori == sampled_orientation]

        # "allow locations to be an approximate match "
        match = [j for j in orientation_match if location_distance_on_torus(video_locations[j], sampled_loc, size = image_size) <= max_proximity]
        for idx in match:
            stimulus_raster[i, idx] = 1

    return stimulus_raster
    
def add_unreliable_intervals(y_pred, p, n_bins, baseline, reliability_interval = 7):
    """
    For each neuron, add unreliable intervals at which the neurons just fire at baseline. For each unit of "reliability_interval", sample from a Bernoulli distribution to decide whether the neuron should fire according to its predicted firing rate or if it should fire at baseline.  
    
    Recall the design of the stimulus video: given an initial stimulus image S, orientation, and movement direction, we created sample image (of fixed orientation) and movement direction for the next 7 frames. We then repeated this process for neighboring values of orientation and movement direction. Thus, the first 8 frames of the stimulus video have the same orientation. After that, every 7 frame has the same orientation.
    
    --- input ---
    y_pred: (array) of neuron firing rates predicted by the neural network. Has shape (n_timebins, n_neurons)
    p: (float) probability "p" of the Bernoulli distribution
    n_bins: (int) number of bins per 1 second
    baseline: (float) baseline firing rate
    reliability_interval: (int) units (in seconds) for unreliability of neurons 
    
    --- output ---
    y_unreliable: (array) of neuron firing rates in which specific intervals are adjusted to have baseline firing rates 
    """
    
    n_neurons = y_pred.shape[1]
    n_frames = y_pred.shape[0]
    y_unreliable = y_pred.copy()
    
    for i in range(n_neurons):
        binomial = np.random.binomial(1, p, 5714)
        binomial = np.repeat(binomial, 7, axis = 0)
        binomial = np.insert(binomial, 0, 1)
        binomial = np.concatenate((binomial, [1]))
        binomial = np.repeat(binomial, n_bins, axis = 0)
        
        for f in range(n_frames):
            if binomial[f] == 0:
                y_unreliable[f,i] = baseline
    return y_unreliable
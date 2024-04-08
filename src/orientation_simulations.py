# This script generates multiple collections of orientation-sensitive neurons and computes the distance matrices


import sys
sys.path.insert(0, '../')

import numpy as np
import itertools
import h5py
from simulations import *
import pickle
from compute_similarity_multiprocessing import * 

def compute_true_firing_rates2(n_neurons, stimulus_orientations, neuron_orientations, C, R_preferred, sigma):
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
    
    n_frames = len(stimulus_orientations)
    true_rate = np.zeros((n_frames, n_neurons))
        
        
    for neuron in range(n_neurons):
        
        orientation_neuron = neuron_orientations[neuron]
        C_neuron = C[neuron]
        R_neuron = R_preferred[neuron]
        sigma_neuron = sigma[neuron]
        
        for i in range(n_frames):

            st_orientation = stimulus_orientations[i] 
    
            # compute the rate 
            true_rate[i, neuron] = tuning_curve(orientation_neuron, C_neuron, R_neuron, sigma_neuron, st_orientation)
    
    return true_rate



def main():
    ### open sample orientation list###

    file = h5py.File('data_v5/stimulus.h5','r')
    sample_orient = file['simulation_stimulus/video_orientations'][:]
    file.close()
    
    
    ### specify possible default parameters ###
    n_neurons = 64
    R_default = [1,2] #[1, 2, 3]
    sigma_default = [0.1,0.2] # [0.1, 0.2]
    C_default = [0.25]  # [0.25, 0.5]
    
    ### specify the lambda parameter of the exponential distribution for adding noise
    R_lambd = [0, 1]
    sigma_lambd = [0, 1/4, 1/8]
    C_lambd = [0, 1/4, 1/8]
    
    params = [R_default, R_lambd, sigma_default, sigma_lambd, C_default, C_lambd]
    all_params = list(itertools.product(*params))
    
    n = len(all_params)
    
    for i in range(n):
        # sample orientations and parameters
        orientations = np.random.uniform(low=0, high=math.pi, size=n_neurons)
        R_preferred = [all_params[i][0] + np.random.exponential(all_params[i][1]) for x in range(n_neurons)]
        sigma = [all_params[i][2] + np.random.exponential(all_params[i][3]) for x in range(n_neurons)]
        C = [all_params[i][4] + np.random.exponential(all_params[i][5]) for x in range(n_neurons)]
    
        # compute true firing rate
        y = compute_true_firing_rates2(n_neurons, sample_orient, orientations, C, R_preferred, sigma)
    
        # adjust rate bin size
        rate = add_noise(y/25, 0)
    
        # sample spike trains
        bin_size = 1
        ori_raster, _ = simulate_raster(rate, bin_size)
        
        # save raster
        hf = h5py.File('data_v5/orientation_simulations/' + str(i) + '_orientation_raster.h5', 'w')
        hf.create_dataset('raster', data = ori_raster)
        hf.close()
        
        # save parameters used 
        pickle.dump(all_params[i], open("data_v5/orientation_simulations/" + str(i) + '_params', "wb"))  # s

        # compute distance and save
        compute_similarity("data_v5/orientation_simulations/" + str(i) + "_orientation_raster.h5", 100, "data_v5/orientation_simulations/" + str(i) + "_orientation_distance.h5" )

if __name__ == "__main__":
    main()

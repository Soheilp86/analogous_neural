#!/usr/bin/env python

# Generates a stimulus video and saves it as "stimulus.h5"
# Make sure current directory is added in the PATH variable

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from itertools import combinations
from scipy.signal import correlate

var_dict = {}
cwd = os.getcwd()

# ---------------------------------------------------------------------------------
# FUNCTIONS FOR GENERATING STIMULUS
# ---------------------------------------------------------------------------------
def move_img(img, old_x, old_y, change_x, change_y):
    # img: stimulus or Gabor filter
    # move image so that the old center (old_x, old_y) moves 
    # move stimulus center to (new_x, new_y)
    size = img.shape[0]
    old_img = np.tile(img, (3, 3))

    # move image
    new_img = np.roll(old_img, change_x, axis=0)
    new_img = np.roll(new_img, change_y, axis = 1)
    
    # new coordinates
    new_x = (old_x + change_x) % size
    new_y = (old_y + change_y) % size

    return new_img[size: size*2, size: size*2], new_x, new_y


def create_stimulus_image(size, wavelength, phase, radius, theta, x_loc = 0, y_loc = 0):  
    
    # generates a single stimulus image consisting of a wave & circular mask
    """  
    size: size of window
    wavelength: wavelength of wave
    phase: phase offset of wave
    radius: radius of stimuli
    theta: orientation !!! IN RADIANS !!! 
    """ 
    x0 = size // 2
    y0 = size // 2

    # create stimulus without mask
    stimulus = np.zeros((size, size))    
    for x in range(size):
        for y in range(size):
            x_prime = (x-x0) * math.cos(theta) + (y-y0) * math.sin(theta)
            y_prime = -(x-x0) * math.sin(theta) + (y-y0) * math.cos(theta)
        
            real = math.cos( 2 * math.pi * x_prime / (wavelength) + phase)
            stimulus[x][y] = real
        
    # mask stimulus 
    X, Y = np.ogrid[0:size, 0:size]
    mask = (X - size / 2) ** 2 + (Y - size / 2) ** 2 > radius**2   
    wave = stimulus.copy()
    stimulus[mask] = 0
    
    # move stimulus as needed
    change_x = x_loc - x0
    change_y = y_loc - y0
    
    stimulus, _, _ = move_img(stimulus, x0, y0, change_x, change_y)
    
    return -stimulus, wave



def change_orientation(possibilities, current):

    #possibilities: (list) of possible orientation values 
    #current: (float) current orientation. Must be in possibilities
    

    # Choose between increasing or decreasing angle theta for orientation
    move = random.choice([1, -1])
    
    current_idx = possibilities.index(current)
    new_idx = ( current_idx + move ) % len(possibilities)
    new = possibilities[new_idx]
    
    return new

def change_direction(dir_angles, angles_dir, current):
    
    #current: current movement direction. Must be in directions
    
    # get all possibile directions as angles
    possibilities = list(angles_dir.keys())
    possibilities.sort()

    # choose whether to increase / decrease direction 
    idx = possibilities.index(dir_angles[current])
    move = random.choice([1,0,-1])
    new_idx = (idx + move) % len(possibilities)
    new = angles_dir[possibilities[new_idx]]
    
    return new

def create_stimulus_video(
    wavelength,
    phase,
    radius,
    size,
    n_frames,
    video_loc,
    video_orient):
    """Creates the stimulus video as a 3D array.

    Parameters
    ----------
    wavelength: wavelength of stimulus wave
    phase: phase of the stimulus wave
    radius: radius of stimulus mask
    size: size of stimulus window
    n_frames: number of frames of the video
    video_loc: list of locations of the disc
    video_orient: list of orientations (of wave)

    Returns 
    -------
    video: 3D numpy array, where [:,:,i] is the i-th frame of video.
    """
    # generate video
    video = np.zeros((size, size, n_frames))
    for i in range(n_frames):
        img, _ = create_stimulus_image(size, wavelength, phase, radius, video_orient[i], video_loc[i,0], video_loc[i,1])
        video[:,:,i] = img
    return video

def create_stimulus_info(
    n_samples, 
    n_frames,
    location, 
    orientation, 
    direction, 
    orientations, 
    dir_angles, 
    angles_dir, 
    size):
    """Creates stimulus video and saves the relevant information without creating the actual images and videos.
    """

    sample_loc = [location]
    sample_orient = [orientation]
    sample_dir = [direction]
    
    loc_x, loc_y = location
    dir_x, dir_y = direction
    
    for i in range(1, n_samples):

        # every "n_frames", change BOTH orientation AND direction
        if (i > 1) and (i % n_frames == 1):
            # choose new orientation and direction
            orientation = change_orientation(orientations, orientation)
            direction = change_direction(dir_angles, angles_dir, direction)
            
        # get new directions and locations
        dir_x, dir_y = direction
        loc_x = (loc_x + dir_x) % size
        loc_y = (loc_y + dir_y) % size

        #S[:,:,i] = img
        sample_loc.append((loc_x, loc_y))
        sample_dir.append((dir_x, dir_y))
        sample_orient.append(orientation)

    # sample directions as angles
    sample_angle = [dir_angles[item] for item in sample_dir]
    sample_dir[0] = None
    sample_angle[0] = None
        
    return sample_loc, sample_orient, sample_dir, sample_angle


# ---------------------------------------------------------------------------------
# FUNCTIONS FOR CHECKING STIMULUS

# These functions can be used to check that the generated stimulus is good.
# ---------------------------------------------------------------------------------
def compute_stimulus_distance_fixed_orientation(
    size, 
    wavelength, 
    phase, 
    radius, 
    theta):
    """Generate stimulus at every possible location with given parameters and compute pairwise distance matrix.

    Note that the orientation, inparticular, is fixed. Distance between two images A and B are computed by the Frobenius norm of A - B.

    Parameters
    ----------
    size: size of window
    wavelength: wavelength of wave
    phase: phase offset of wave
    radius: radius of stimuli
    theta: Fixed orientation !!! IN RADIANS !!! 
    
    Returns
    -------
    stimuli_dict: Dictionary of stimuli
    stimuli_distance: (np array) distance matrix
    """

    ##### Create stimulus at every location
    stimuli_dict = {}
    for x in range(size):
        for y in range(size):
            stimulus, _ = create_stimulus_image(size, wavelength, phase, radius, theta, x, y)
            stimuli_dict[(x,y)] = stimulus
   
    ##### compute distance
    stimuli_distance = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            pair_distance = np.linalg.norm(stimuli_dict[(i,j)] - stimuli_dict[(j,i)])
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_dict, stimuli_distance


def compute_sampled_stimulus_distance(
    size, 
    wavelength, 
    phase, 
    radius, 
    theta,
    n_sample):
    """ Computes the distance matrix among a random sample of stimulus images with FIXED ORIENTATION. 
    
    Parameters
    ----------
    size: size of window
    wavelength: wavelength of wave
    phase: phase offset of wave
    radius: radius of stimuli
    theta: Fixed orientation !!! IN RADIANS !!! 
    n_sample: number of stimulus to sample
    
    Returns
    -------
    stimuli_matrix: (dict) of stimuli images. stimuli_matrix[i] is the i-th 2d stimulus
    sampled_loc: (list) of sampled stimuli locations
    stimuli_distance: (np array) distance matrix
    """
    n_stimuli = size **2

    # sample (locx, locy) without replacement 
    sampled_idx = random.sample(range(n_stimuli), n_sample)
    sampled_loc = [(item // size, item % size) for item in sampled_idx]

    # create stimuli
    stimuli_matrix = {}
    for idx in range(n_sample):
        locx = sampled_idx[idx] // size
        locy = sampled_idx[idx] % size
        stimulus, _ = create_stimulus_image(size, wavelength, phase, radius, theta, locx, locy)
        stimuli_matrix[idx] = stimulus
        
        
    # compute distance
    stimuli_distance = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            pair_distance = np.linalg.norm(stimuli_matrix[i] - stimuli_matrix[j])
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_distance, stimuli_matrix

def check_locations(orientations, locations, selected_orientation, size):
    """ Visualize the frequency of stimulus center locations.
    
    Parameters
    ----------
    orientations: (list) of stimulus orientations. orientations[i]: orientation of ith frame of stimulus.
    locations: (list) of stimulus locations. locations[i]: location of the ith frame of stimulus.
    selected_orientation: (float) of orientation selected.
    size: (int) size of window. 

    Returns
    -------
    loc_viz: (np array) loc_vis[x][y] is the frequency of stimulus with center at (x, y)
    Also plots a visualization of loc_viz
    """
    loc_vis = np.zeros((size, size))
    for i in range(len(orientations)):
        if orientations[i] == selected_orientation:
            loc_vis[locations[i][0], locations[i][1]] += 1
    
    plt.imshow(loc_vis)
    plt.colorbar()
    plt.show()
    return loc_vis

def check_directions(orientations, directions, selected_orientation, selected_directions):
    """ Check frequency of frames with selected orientation and movement direction pair.

    This function can be used to check that different pairs of orientations and movement directions have similar freuqneices 
    
    Parameters
    ----------
    orientations: (list) of orientations of a stimulus video. orientations[i]: orientation of ith frame of stimulus.
    locations: (list) of locations of a stimulus video. locations[i]: location of the ith frame of stimulus.
    selected_orientation: (float) of selected orientation.
    selected_directions: (list of lists) indicating selected direction. 
        Should include the two directions on the same selected axis. ex: [(3,3),(-3,-3)]

    Returns
    -------
    count: (int) number of frames in the stimulus that have both the selected orientation and selected directions.
    """
    count = 0
    for i in range(len(orientations)):
        if orientations[i] == selected_orientation:
            if directions[i] in selected_directions:
                count += 1
    print('number of frames with orientation %.2f and movements ' %selected_orientation, selected_directions, ' : %d' %count)
    return count

def compute_total_bar_stimulus_distance(
    size, 
    wavelength, 
    phase, 
    radius, 
    sigma,
    gamma,
    orientations,
    n_samples_per_orientation):
    """ Computes the distance matrix among a random sample of stimulus images 
    
    Parameters
    ----------
    size: size of window
    wavelength: wavelength of wave
    phase: phase offset of wave
    radius: radius of stimuli
    n_sample_per_orientation: number of stimulus to sample per orientation
    
    Returns
    -------
    stimuli_matrix: (dict) of stimuli images. stimuli_matrix[i] is the i-th 2d stimulus
    sampled_loc: (list) of sampled stimuli locations
    stimuli_distance: (np array) distance matrix
    """
    n_sample = len(orientations) * n_samples_per_orientation
    n_stimuli = size **2
    
    # create stimuli
    stimuli_matrix = {}
    for idx, theta in enumerate(orientations):
        # sample (locx, locy) without replacement for fixed orientation 
        sampled_idx = random.sample(range(n_stimuli), n_samples_per_orientation)
        #sampled_loc = [(item // size, item % size) for item in sampled_idx]
        
        # create stimulus image and save to dictionary
        for j in range(n_samples_per_orientation):
            locx = sampled_idx[j] // size
            locy = sampled_idx[j] % size
            stimulus = create_bar_stimulus_image(size, wavelength, phase, radius, theta, sigma, gamma, x_loc = locx, y_loc = locy)
            stimuli_matrix[idx * n_samples_per_orientation + j] = stimulus
        
    # compute distance
    stimuli_distance = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            pair_distance = np.linalg.norm(stimuli_matrix[i] - stimuli_matrix[j])
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_distance, stimuli_matrix


def compute_bar_stimulus_angular_distance_fixed_orientation(
    size, 
    wavelength, 
    phase, 
    radius, 
    sigma,
    gamma,
    theta,
    n_sample):
    """ Computes the distance matrix among a random sample of stimulus images with FIXED ORIENTATION. 
    
    Parameters
    ----------
    size: size of window
    wavelength: wavelength of wave
    phase: phase offset of wave
    radius: radius of stimuli
    theta: Fixed orientation !!! IN RADIANS !!! 
    n_sample: number of stimulus to sample
    
    Returns
    -------
    stimuli_matrix: (dict) of stimuli images. stimuli_matrix[i] is the i-th 2d stimulus
    sampled_loc: (list) of sampled stimuli locations
    stimuli_distance: (np array) distance matrix
    """
    n_stimuli = size **2

    # sample (locx, locy) without replacement 
    sampled_idx = random.sample(range(n_stimuli), n_sample)
    sampled_loc = [(item // size, item % size) for item in sampled_idx]

    # create stimuli
    stimuli_matrix = {}
    for idx in range(n_sample):
        locx = sampled_idx[idx] // size
        locy = sampled_idx[idx] % size
        stimulus = create_bar_stimulus_image(size, wavelength, phase, radius, theta, sigma, gamma, x_loc = locx, y_loc = locy)
        stimuli_matrix[idx] = stimulus
        
        
    # compute distance
    stimuli_distance = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            locx_distance = distance_on_S1(sampled_loc[i][0], sampled_loc[j][0], size)
            locy_distance = distance_on_S1(sampled_loc[i][1], sampled_loc[j][1], size)
            angle_distance = 0
            
            pair_distance = math.sqrt(locx_distance**2 + locy_distance**2)
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_distance, stimuli_matrix, sampled_loc

def distance_on_S1(phi, theta, size):
    phi = phi / size * math.pi * 2
    theta = theta / size * math.pi * 2
    
    s = min(phi, theta)
    l = max(phi, theta)
    distance = min(l - s, s + 2 * math.pi - l)
    return distance



def compute_stimulus_angular_distance(
    size, 
    orientations,
    n_samples_per_orientation):
   
    # varaibles keeping track of sampled orientations and locations 
    n_sample = len(orientations) * n_samples_per_orientation
    n_stimuli = size **2
    locations_sampled = []
    orientations_list = [[i] * n_samples_per_orientation for i in orientations]
    orientations_sampled = [item for sublist in orientations_list for item in sublist]
    
    # sample (locx, locy) without replacement for fixed orientation 
    for idx, theta in enumerate(orientations):
        # sample (locx, locy) without replacement for fixed orientation 
        sampled_idx = random.sample(range(n_stimuli), n_samples_per_orientation)
        
        for j in range(n_samples_per_orientation):
            locx = sampled_idx[j] // size
            locy = sampled_idx[j] % size

            locations_sampled.append((locx, locy))
       
    # compute distance
    stimuli_distance = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            locx_distance = distance_on_S1(locations_sampled[i][0], locations_sampled[j][0], size)
            locy_distance = distance_on_S1(locations_sampled[i][1], locations_sampled[j][1], size)
            
            # find the orientations
            theta_i = orientations[i // n_samples_per_orientation]
            theta_j = orientations[j // n_samples_per_orientation]
            angle_distance = distance_on_S1(theta_i, theta_j, math.pi)
            
            pair_distance = math.sqrt(locx_distance**2 + locy_distance**2 + angle_distance**2)
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_distance,locations_sampled, orientations_sampled

"""
def compute_total_bar_stimulus_angular_distance(
    size, 
    wavelength, 
    phase, 
    sigma,
    gamma,
    orientations,
    n_samples_per_orientation):

    n_sample = len(orientations) * n_samples_per_orientation
    n_stimuli = size **2
    locations_sampled = []
    orientations_list = [[i] * n_samples_per_orientation for i in orientations]
    orientations_sampled = [item for sublist in orientations_list for item in sublist]
    
    # create stimuli
    stimuli_matrix = {}
    for idx, theta in enumerate(orientations):
        # sample (locx, locy) without replacement for fixed orientation 
        sampled_idx = random.sample(range(n_stimuli), n_samples_per_orientation)
        
        # create stimulus image and save to dictionary
        for j in range(n_samples_per_orientation):
            locx = sampled_idx[j] // size
            locy = sampled_idx[j] % size

            # This generates the stimulus image: We don't actually need this for computing distance matrix
            stimulus = create_bar_stimulus_image(size = size, 
                                                 wavelength = wavelength, 
                                                 phase = phase, 
                                                 theta = theta, 
                                                 gamma = gamma,
                                                 sigma = sigma,
                                                 x_loc = locx,
                                                 y_loc = locy)
            stimuli_matrix[idx * n_samples_per_orientation + j] = stimulus
            locations_sampled.append((locx, locy))
        
    # compute distance
    stimuli_distance = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            locx_distance = distance_on_S1(locations_sampled[i][0], locations_sampled[j][0], size)
            locy_distance = distance_on_S1(locations_sampled[i][1], locations_sampled[j][1], size)
            
            
            # find the orientations
            theta_i = orientations[i // n_samples_per_orientation]
            theta_j = orientations[j // n_samples_per_orientation]
            angle_distance = distance_on_S1(theta_i, theta_j, math.pi)
            
            pair_distance = math.sqrt(locx_distance**2 + locy_distance**2 + angle_distance**2)
            stimuli_distance[i][j] = pair_distance
            stimuli_distance[j][i] = pair_distance

    return stimuli_distance, stimuli_matrix, locations_sampled, orientations_sampled
"""






def create_stimulus_video_parameters(size = 40, n_samples = 40000, output_file = "stimulus.h5"):
    
    # ------------------------ SET UP PARAMETERS ------------------------
    # create list of orientations 
    n_orientations = 8
    orientations = [x * math.pi / n_orientations for x in range(n_orientations)]
    
    # create movement directions 
    directions = [(3,0), (3,3), (0,3), (-3, 3), (-3,0), (-3,-3), (0,-3), (3, -3)]
    
    # create a dictionary to find (angle, direction)
    dir_angles = {item:np.arctan2(item[0], item[1]) for item in directions}
    angles_dir = {value:key for (key, value) in dir_angles.items()}
    
    # video default parameters
    n_frames = 7 # change movement direction and orientation every "n_frames"
    loc_0 = (0,0)
    orientation_0 =0
    direction_0 = (0,3)
    radius = 0 # This parameter is irrelevant when working with bar sitmulus
    # --------------------------------------------------------------------

    # Generate training stimulus
    train_loc, train_orient, train_direction, train_angle = create_stimulus_info(n_samples, 
                                                n_frames,
                                                loc_0, 
                                                orientation_0, 
                                                direction_0, 
                                                orientations, dir_angles, angles_dir, 
                                                size)

    # Generate simulation stimulus
    simulation_loc, simulation_orient, simulation_direction, simulation_angle = create_stimulus_info(n_samples, 
                                                n_frames,
                                                loc_0, 
                                                orientation_0, 
                                                direction_0, 
                                                orientations, dir_angles, angles_dir, 
                                                size)


    # save video information
    train_direction[0] = (-1,-1)
    simulation_direction[0] = (-1, -1)
    train_angle[0] = -1
    simulation_angle[0] = -1

    f = h5py.File(output_file, 'w')

    ### Save training stimulus info
    # create goup for training stimulus
    grp = f.create_group('train_stimulus')

    # store stimuli video information
    grp.create_dataset('video_locations', data = train_loc)
    grp.create_dataset('video_orientations', data = train_orient)
    grp.create_dataset('video_directions', data = train_direction)
    grp.create_dataset('orientations', data = orientations)
    grp.create_dataset('directions', data = directions)

    # save stimuli parameters as attribute
    grp.attrs['frames'] = n_samples
    grp.attrs['change_frames'] = n_frames

    ### Save simulation stimulus info
    # create group for simulation stimulus
    grp = f.create_group('simulation_stimulus')
    # store stimuli video information
    grp.create_dataset('video_locations', data = simulation_loc)
    grp.create_dataset('video_orientations', data = simulation_orient)
    grp.create_dataset('video_directions', data = simulation_direction)
    grp.create_dataset('orientations', data = orientations)
    grp.create_dataset('directions', data = directions)

    # save stimuli parameters as attribute
    grp.attrs['frames'] = n_samples
    grp.attrs['change_frames'] = n_frames

    f.close()
    print('Stimulus generation complete. Stimulus info saved to ' + output_file)

    
    
# ---------------------------------------------------------------------------------
# Functions computing similarity & dissimilarity between spike trains
# These have been copy-pasted to "spiketrain_similarity.py" file (July, 2022)
# ---------------------------------------------------------------------------------    
    
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
    # Given entires in the upper triangle of a symmetric matrix, return the matrix itself
    
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size, 1)] = entries
    return matrix + matrix.T


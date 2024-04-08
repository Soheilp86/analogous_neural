#!/usr/bin/env python

# As a module: contains functions for generating and checking Gabor filters
# When this script is run as the main file, it generates Gabor filters and saves them
# Make sure current directory is added in the PATH variable
import h5py
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import random
from tqdm import tqdm

# ---------------------------------------------------------------------------------
# GENERATING GABOR FILTERS
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

def create_gabor_filter(x0, y0, wavelength, theta, phase, sigma, gamma, size):
    # creates a single Gabor filter
    """    
    x0, y0: location of Gabor filter
    wavelength: wavelength of carrier
    theta: orientation of carrier
    phase: phase offset of carrier
    sigma: variance of Gaussian envelope
    gamma: spatial aspect ratio of the Gaussian envelope
    size: size of window 
    """
    
    real_matrix = np.zeros((size, size))
    imaginary_matrix = np.zeros((size, size))
    filter_matrix = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):

            x_prime = (x-x0) * math.cos(theta) + (y-y0) * math.sin(theta)
            y_prime = -(x-x0) * math.sin(theta) + (y-y0) * math.cos(theta)

            real = math.exp(-(x_prime**2 + gamma**2 * y_prime**2 )/(2 * sigma**2)) * math.cos( 2 * math.pi * x_prime / (wavelength) + phase)
            imaginary = math.exp(-(x_prime**2 + gamma**2 * y_prime**2 )/(2 * sigma**2)) * math.sin( 2 * math.pi * x_prime / (wavelength) + phase)
            
            real_matrix[x,y] = real
            imaginary_matrix[x,y] = imaginary
            filter_matrix[x,y] = real + imaginary
            
    return real_matrix, imaginary_matrix, filter_matrix


def create_Gabor_filters(size, wavelength, sigma, gamma, orientation_list, phase_list, Gabor_locations):
    # Creates a collection of Gabor filters given the possible orientations, phases, and locations
    # size, wavelength, sigma, gamma: parameters of Gabor filters
    # orientations_list, phase_list, Gabor_locations: (list) Create Gabor filters for every combination of these lists
    n_filters = len(orientation_list) * len(phase_list) * len(Gabor_locations)

    df_Gabor = pd.DataFrame(columns = ['orientation', 'location_x', 'location_y', 'phase','arr'],  
                       index = list(range(n_filters))) 
    df_Gabor['arr'] = df_Gabor['arr'].astype(object)
    idx = 0
    for ori in orientation_list:
        for p in phase_list:
            for loc in Gabor_locations:
                x, y = loc 

                # create Gabor filter
                real, _, _ = create_gabor_filter(size//2, size//2, wavelength, ori, p, sigma, gamma, size)
    
                change_x = x - size//2
                change_y = y - size//2

                new_img, _, _ = move_img(real, size//2, size//2, change_x, change_y)

                df_Gabor.loc[idx,'orientation'] = ori
                df_Gabor.loc[idx, 'location_x'] = x
                df_Gabor.loc[idx, 'location_y'] = y
                df_Gabor.loc[idx, 'phase'] = p
                df_Gabor.loc[idx, 'arr'] = new_img
                #df_Gabor.loc[idx] = [ori, x, y, p, new_img]
                idx += 1
        
    return df_Gabor

def create_Gabor_filters_info(size, wavelength, sigma, gamma, orientation_list, Gabor_locations):
    # Creates a collection of Gabor filters given the possible orientations, phases, and locations
    # size, wavelength, sigma, gamma: parameters of Gabor filters
    # orientations_list, phase_list, Gabor_locations: (list) Create Gabor filters for every combination of these lists
    n_filters = len(orientation_list) * len(Gabor_locations)

    df_Gabor = pd.DataFrame(columns = ['orientation', 'location_x', 'location_y'],  
                            index = list(range(n_filters))) 
    
    idx = 0
    for ori in orientation_list:
        for loc in Gabor_locations:
            x, y = loc 

            # create Gabor filter
            #real, _, _ = create_gabor_filter(size//2, size//2, wavelength, ori, p, sigma, gamma, size)

            #change_x = x - size//2
            #change_y = y - size//2

            #new_img, _, _ = move_img(real, size//2, size//2, change_x, change_y)

            df_Gabor.loc[idx,'orientation'] = ori
            df_Gabor.loc[idx, 'location_x'] = x
            df_Gabor.loc[idx, 'location_y'] = y
            #df_Gabor.loc[idx, 'arr'] = new_img
            #df_Gabor.loc[idx] = [ori, x, y, p, new_img]
            idx += 1
        
    return df_Gabor


########## Functions for creating Gabor filters at various locations ##########
# Returns a list of list of the format [[point0_x, point0_y], [point1_x, point1_y], ..., ]

def Gabor_filters_random_locations(size, n_filters):
    Gabor_locations = []
    while len(Gabor_locations) < n_filters:
        # sample point
        x = random.choice(range(size))
        y = random.choice(range(size))

        # check that sampled point is new
        if [x, y] not in Gabor_locations:
            Gabor_locations.append([x,y])
    return Gabor_locations

def Gabor_filters_sample_accept(size, n_filters):
    """Sample locations for Gabor filters, then accept or reject depending on existing density
    
    """

    # neighbors of a point: 24 points in the immediate  vicinity
    relative_neighbors =[[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1],
                         [2,0], [-2,0], [0,2], [0,-2], [2,2], [2,-2], [-2,2], [-2,-2],
                         [2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [-1, 2], [1, -2],[-1, -2] ]

    Gabor_locations = []
    while len(Gabor_locations) < n_filters:
        # sample point
        sampled_x = random.choice(range(size))
        sampled_y = random.choice(range(size))

        # check that sampled point is new
        if [sampled_x, sampled_y] not in Gabor_locations:

            # generate neighbors
            neighbors = [list(map(operator.add, [sampled_x, sampled_y], item)) for item in relative_neighbors]
            neighbors = [[item[0] % size, item[1] % size ] for item in neighbors]

            # count number of neighbors that are already sampled
            count = 0
            for point in neighbors:
                if point in Gabor_locations:
                    count +=1
            # if less than 2 neighbors have been sampled, add new point to list of locations
            if count < 2:
                Gabor_locations.append([sampled_x, sampled_y])
    
    return Gabor_locations

"""
def Gabor_filters_on_cross(size):
    # size: size of window. must be divisible by 2

    # cross center
    center_x = size // 2
    center_y = size // 2 

    Gabor_locations = [[center_x, y] for y in range(size)] +[[x, center_y] for x in range(size)]
    return Gabor_locations


def Gabor_filters_on_grid(size, grid_size):
    # size: size of window.
    # grid_size: Gabor filters will be on every 'grid_size' locations. 'grid_size' must divide 'size'

    # define grid
    locx = list(range(0, size, grid_size))
    locy = list(range(0, size, grid_size))

    locations = itertools.product(locx, locy)
    Gabor_locations = [list(item) for item in locations]

    return Gabor_locations

def Gabor_filters_on_grid_perturb(size, grid_size):
    # size: size of window
    # grid_size: gabor filters will be on every 'grid_size' location before being perturbed.
    grid_locations = Gabor_filters_on_grid(size, grid_size)

    # perturb
    move_possibilities = {0 : [0,0], 
                      1 : [1,0], 
                      2 : [-1,0], 
                      3 : [0,1], 
                      4 : [0,-1], 
                      5 : [1,1], 
                      6 : [1,-1], 
                      7 : [-1,1], 
                      8 : [-1,-1]}

    Gabor_locations = []
    for loc in grid_locations:
        # sample movement direction
        move = np.random.choice(range(9))

        # update location
        new_loc = list(map(operator.add, loc, move_possibilities[move]))

        # take care of locations modulo size
        new_loc[0] = new_loc[0] % size
        new_loc[1] = new_loc[1] % size
        Gabor_locations.append(new_loc)
    
    return Gabor_locations
"""
def plot_Gabor_locations(Gabor_locations, size):
    """ Visualize the locations of the Gabor filters
    """
    Gabor_loc = np.zeros((size, size))
    for item in Gabor_locations:
        Gabor_loc[item[0], item[1]] = 1

    plt.figure(figsize=(8,8))
    plt.pcolor(Gabor_loc[::-1],edgecolors='k', linewidths=1)
    plt.show()

    
def compute_video_V1_overlap(video, V1):
    """compute the overlap between each frame of a video and simulated V1 neurons (Gabor filters)
    
    --- input ---
    video: array of size (s, s, n). Video of "n" frames, where each frame consists of an s x s image
    V1: array of size (s, s, m). Consists of "m" Gabor filters, where each Gabor filter is an image of size s x s
    
    --- output ---
    overlap: array of size (n, m), where overlap[i,j] is the overlap between video frame "i" and V1 neuron "j"

    """
    
    n_videos = video.shape[2]
    n_neurons = V1.shape[2]
    overlap = np.zeros((n_videos, n_neurons))

    for i in tqdm(range(n_videos)):
        img = video[:,:,i]
        for j in range(n_neurons):
            neuron = V1[:,:,j]
            overlap[i,j] = np.sum(np.multiply(img, neuron))
    return overlap
    
    
def compute_A_minus_B_on_circle(A, B, size):
    """computes x - x0 on a circle of given size
    
    """
    
    orig = abs(A - B)
    option1 = abs(A + size - B)
    option2 = abs(A - size - B)
    
    min_val = min(orig, option1, option2)
    if min_val == orig:
        return A-B
    elif min_val == option1:
        return A + size - B
    else:
        return A - size - B
        
def create_gabor_filter_torus(x0, y0, wavelength, theta, phase, sigma, gamma, size):
    # creates a single Gabor filter
    """    
    x0, y0: location of Gabor filter
    wavelength: wavelength of carrier
    theta: orientation of carrier
    phase: phase offset of carrier
    sigma: variance of Gaussian envelope
    gamma: spatial aspect ratio of the Gaussian envelope
    size: size of window 
    """
    
    real_matrix = np.zeros((size, size))
    imaginary_matrix = np.zeros((size, size))
    filter_matrix = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):
            
            x_diff = compute_A_minus_B_on_circle(x, x0, size)

            y_diff = compute_A_minus_B_on_circle(y, y0, size)

            
            x_prime = (x_diff) * math.cos(theta) + (y_diff) * math.sin(theta)
            y_prime = -(x_diff) * math.sin(theta) + (y_diff) * math.cos(theta)

            real = math.exp(-(x_prime**2 + gamma**2 * y_prime**2 )/(2 * sigma**2)) * math.cos( 2 * math.pi * x_prime / (wavelength) + phase)
            imaginary = math.exp(-(x_prime**2 + gamma**2 * y_prime**2 )/(2 * sigma**2)) * math.sin( 2 * math.pi * x_prime / (wavelength) + phase)
            
            real_matrix[x,y] = real
            imaginary_matrix[x,y] = imaginary
            filter_matrix[x,y] = real + imaginary
            
    return -real_matrix, imaginary_matrix, filter_matrix
        
        
# ---------------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------------

def main():
    """Generate Gabor filters and save."""
    ### Default parameters of Gabor filters ----------------------------------
    # Parameters for single Gabor filter
    size = 80 # same as stimuli
    wavelength = 16 # wavelength of carrier 
    phase = math.pi # phase offset of carrier
    sigma = 10 # control variance Gaussian envelope
    gamma = 1 # spatial aspect ratio of the Gaussian envelope
    
    # Parameters for the collection of Gabor filters
    n_orientations = 8
    orientations = [x * math.pi / n_orientations for x in range(n_orientations)]
    #------------------------------------------------------------------------------

    # User inputs number of Gabor filters per orientation
    print('*** This script creates Gabor filters and saves them in neurons.h5 in CWD. ***')
    n_filters = int(input('Enter the number of Gabor filters per orientation: '))
    print('Total number of Gabor filters: ', n_orientations * n_filters)

    df_Gabor = pd.DataFrame(columns = ['orientation', 'location_x', 'location_y'])

    for ori in orientations:
        # sample Gabor filter locations
        Gabor_locations = Gabor_filters_sample_accept(size, n_filters)
        
        df_Gabor_sub = create_Gabor_filters_info(size, wavelength, sigma, gamma, [ori], Gabor_locations)
        df_Gabor = pd.concat([df_Gabor, df_Gabor_sub])

    df_Gabor.reset_index(inplace = True)

    # SAVE
    cwd = os.getcwd()
    f = h5py.File(cwd + '/neurons.h5','a')
    # create group for Gabor filters
    grp = f.create_group('V1')
    grp.create_dataset('orientations', data = df_Gabor.orientation.tolist())
    grp.create_dataset('location_x', data = df_Gabor.location_x.tolist())
    grp.create_dataset('location_y', data = df_Gabor.location_y.tolist())
    grp.attrs['size'] = size
    grp.attrs['wavelength'] = wavelength
    grp.attrs['phase'] = phase
    grp.attrs['sigma'] = sigma
    grp.attrs['gamma'] = gamma
    grp.attrs['orientations'] = orientations
    f.close()

if __name__ == '__main__':
    main()
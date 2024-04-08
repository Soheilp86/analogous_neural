#!/usr/bin/env python

# This script computes the Gabor filter response in a simulation
# Uses multiprocessing 
# Make sure current directory is added in the PATH variable
import h5py
import math
import multiprocessing
from multiprocessing import Pool, RawArray
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import time
from tick.base import TimeFunction
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson

var_dict = {}
cwd = os.getcwd()

def init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

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
    # generates a single stimulus image
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
    stimulus[mask] = 0
    
    # move stimulus as needed
    change_x = x_loc - x0
    change_y = y_loc - y0
    
    stimulus, _, _ = move_img(stimulus, x0, y0, change_x, change_y)
    
    return stimulus


def create_stimulus(stimuli):
    # create the stimulus image.

    # get stimulus video info
    f = h5py.File(cwd+'/stimulus.h5', 'r')
    
    if stimuli == 'train':
        group = 'train_stimulus'
    elif stimuli == 'simulation':
        group = 'simulation_stimulus'
    else:
        raise KeyError("Selected group doesn't exist in data.h5")

    video_loc = np.array(f[group + '/video_locations'])
    video_orient = list(f[group + '/video_orientations'])
    n_frames = f[group].attrs['frames']
    wavelength = f[group].attrs['wavelength']
    phase = f[group].attrs['phase']
    radius = f[group].attrs['radius']
    size = f[group].attrs['size']
    f.close()

    # generate images 
    S = np.zeros((size, size, n_frames))
    for i in range(n_frames):
        img = create_stimulus_image(size, wavelength, phase, radius, video_orient[i], video_loc[i,0], video_loc[i,1])
        S[:,:,i] = img

    return S

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

def create_Gabor_filter_images():
    # create the Gabor filter images 
    # get Gabor filter info
    f = h5py.File(cwd + '/neurons.h5','r')
    
    orientations = list(f['V1/orientations'])
    locx = list(f['V1/location_x'])
    locy = list(f['V1/location_y'])
    size = f['V1'].attrs['size']
    wavelength = f['V1'].attrs['wavelength']
    phase = f['V1'].attrs['phase']
    sigma = f['V1'].attrs['sigma']
    gamma = f['V1'].attrs['gamma']  
    f.close()

    # generate Gabor filter images
    # create Gabor filter
    n_filters = len(orientations)
    G = np.zeros((size, size, n_filters))
    for i in range(n_filters):

        real, _, _ = create_gabor_filter(size//2, size//2, wavelength, orientations[i], phase, sigma, gamma, size)

        change_x = locx[i] - size//2
        change_y = locy[i] - size//2

        new_img, _, _ = move_img(real, size//2, size//2, change_x, change_y)
        G[:,:,i] = new_img
    return G

def compute_Gabor_response(Gabor_filter):
    X_video = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    n_frames = X_video.shape[2]
    Gabor_response = [0]* n_frames 
    for frame in range(n_frames):
        img = X_video[:,:,frame]
        Gabor_response[frame] = np.sum(np.multiply(img, Gabor_filter))
    return Gabor_response

def nonlinear_tanh(g2, L0, r_max, array, r0):
    return r0 + r_max*np.maximum(0,np.tanh(g2*(array-L0)))

def simulate_neuron(rate, bin_size = 5):
    """Simulate the spike train of a single neuron with given rate using inhomogeneous Poisson process.
    
    Parameters
    ----------
    rate: (arr) of shape (n_time, )
    bin_size: (int) number of bins for one unit of time
                ex) If unit of time in the "rate" array is 1 second,
                then bin_size of 1000 indicates that 1 bin of the 
                resulting raster corresponds go 1/1000 second.   

    Returns
    -------
    spike_binary: (binary arr) of shape (n_time * bin_size, ) 
            spike_binary[i] = 1 if there exists a spike in bin i
    spike_count: (array of Int) of shape (n_time * bin_size, )
            spike_count[i] is the number of spikes that occurred in bin i
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

    return spike_binary, spike_count

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
        spike_binary, spike_count = simulate_neuron(rate[:,neuron], bin_size)
        binary_raster[neuron,:] = spike_binary
        count_raster[neuron,:] = spike_count
        
    return binary_raster, count_raster
"""
def main():
    print("*** This script computes the V1 neuron's response to a given stimulus. ***")
    print("*** Both the stimulus and V1 neurons must be stored in data.h5 in CWD. *** ")
    stimuli = input("Select stimuli 'train' or 'simulation' : ")   
    
    ##### Prepare stimuli video for child process #####
    print('Generating stimulus images')
    train_video = create_stimulus(stimuli)
    video_shape = train_video.shape
    n_frames = train_video.shape[2]

    # use RawArray, since we only need read access
    X = RawArray('d', video_shape[0] * video_shape[1] * video_shape[2])

    # wrap X as an numpy array
    X_np = np.frombuffer(X, dtype = np.float64).reshape(video_shape)

    # copy data to shared array X
    np.copyto(X_np, train_video)

    ##### Prepare Gabor filters #####
    print('Generating Gabor filter images')
    Gabor = create_Gabor_filter_images()
    n_filters = Gabor.shape[2]

    ##### using multiprocessor #####
    print('Computing Gabor filter repsonse to stimulus...')
    n_processes = 50
    pool = multiprocessing.Pool(processes = n_processes, initializer=init_worker, initargs=(X, video_shape))
    
    data_list = [Gabor[:,:,i] for i in range(n_filters)]
    tik = time.time()
    output = pool.map(compute_Gabor_response, data_list)
    pool.close()
    tok = time.time()
    duration = tok-tik
    print('Computation time for %d filters with %d processors: %.2f' %(n_filters, n_processes, duration))

    ##### convert response to firing rate #####
    print('Computing firing rate and simulating raster...')
    response = np.zeros((n_frames, n_filters))
    for i in range(n_filters):
        response[:,i] = output[i]

    # parameters for tanh function
    r0  = 0.25 # baseline firing rate
    L0 = 10
    g2 = 0.05
    r_max = 5
    rate_tanh = r0 + nonlinear_tanh(g2, L0, r_max, response)

    # simulate raster
    bin_size = 1
    binary_raster, _ = simulate_raster(rate_tanh, bin_size)

    # save output
    if stimuli == 'train':
        hf = h5py.File(cwd + '/V1_training_rate.h5','w')
        hf.create_dataset('rate', data = rate_tanh)
        hf.close()
    else:
        # save rate
        hf = h5py.File(cwd + '/V1_simulation_rate.h5', 'w')
        hf.create_dataset('rate', data = rate_tanh)
        hf.close()
        # save raster
        hf = h5py.File(cwd + '/V1_simulation_raster.h5', 'w')
        hf.create_dataset('raster', data = binary_raster)
        hf.attrs['bin_size'] = bin_size
        hf.close()

    print('Finished.')

if __name__ == '__main__':
    main()
"""
# serial version
"""
tik = time.time()
for item in range(n_filters):
    result = compute_Gabor_response(item)
tok = time.time()
duration = tok-tik
print('Serial computation time for %d neurons: %.2f seconds' %(n_filters,duration ))
"""
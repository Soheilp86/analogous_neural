#!/usr/bin/env python

# Functions for plotting in python

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import Gabor_filter_functions as gf
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from stimulus_helper import *


def plot_PD(barcode, 
             lw = 10,
             color = "slategrey",
             ax = None,
             highlight = None,
             title = "",
             titlefontsize = 12,
             alpha = 0.5,
             pd_min = None,
             pd_max = None,
             tickfontsize = 5,
             threshold = None,
             threshold_lw = 3,
             diagonal_lw = 3,
             *args,
             **kwargs):
    """plots PD"""
    
    m = max(barcode[:,1])
    ax = ax or plt.gca()
            
    ax.set_aspect('equal', 'box')
    if pd_max == None:
        ax.set_xlim((0, m * 1.1))
        ax.set_ylim((0, m * 1.1))
        ax.plot([0, m * 1.1], [0, m * 1.1], c = "black", linewidth = diagonal_lw, zorder = 1)
        
        # if "threshold" is provided, plot dotted line
        if threshold != None:
            x = np.linspace(0, m * 1.1)
            y = x + threshold
            ax.plot(x, y, linestyle = "dashed", c = "black", linewidth = threshold_lw, zorder = 1)
    
        
    else:
        ax.set_xlim((pd_min, pd_max))
        ax.set_ylim((pd_min, pd_max))
        ax.plot([0, pd_max], [0, pd_max], c = "black", linewidth = diagonal_lw, zorder = 1)
        
        # if "threshold" is provided, plot dotted line
        if threshold != None:
            x = np.linspace(pd_min, pd_max)
            y = x + threshold
            ax.plot(x, y, linestyle = "dashed", c = "black", linewidth = threshold_lw, zorder = 1)
    
        
        
    ax.scatter(barcode[:,0], barcode[:,1], c = color, alpha = alpha, *args, **kwargs, zorder = 2)
        # if "highlight" is provided, color the selected points in specific colors
    if highlight != None:
        for (p_color, point) in highlight.items():
            ax.scatter(barcode[point, 0], barcode[point, 1], c = p_color, *args, **kwargs, zorder = 2)
    ax.tick_params(labelsize=tickfontsize)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize = titlefontsize)
   
    return ax

def plot_barcode(barcode, 
                 lw = 10,
                 color = "slategrey",
                 xleft = None,
                 xright = None,
                 ybottom = None,
                 ytop = None,
                 title = None,
                 titlefontsize = 16,
                 labelsize = None,
                 ax = None):
    """plots barcode"""
    
    ax = ax or plt.gca()
    n_bars = barcode.shape[0]
    
    # find maximum death time (excluding Inf, if it exists)
    death_times = barcode[:,1]
    death_max = np.nanmax(death_times[death_times != np.inf])

    # plot bars
    for i in range(n_bars):
        left = barcode[i,0]
        right = barcode[i,1]
        
        # adjust the right end point (if inf)
        if right == np.inf:
            right = death_max * 1.2
        
        # specific kwargs
        ax.hlines(i, left, right, color = color, lw = lw)
        
    # set xlim  
    if xleft != None:
        ax.set_xlim(left = xleft)
    if xright != None:
        ax.set_xlim(right = xright)    
    if labelsize != None:
        ax.tick_params(axis='x', labelsize=labelsize)
        
    # set ylim
    if ytop != None:
        ax.set_ylim(top = ytop)
    if ytop != None:
        ax.set_ylim(bottom = ybottom)
    if (ytop == None) & (ybottom == None):    
        ax.set_ylim(bottom = -0.5, top = n_bars - 0.5)
        
    ax.set_yticks([])
    
    if title != None:
        ax.set_title(title, fontsize = titlefontsize)
  
    return ax

def color_barcode(ax, barcode, bar_idx, color, lw = 10, epsilon = None):
    """given a barcode plot, color specific bars. 
    This function should be called after calling `plot_barcode()`
    
    Parameters
    ----------
    ax: (matplotlib axes object)
    barcode: (array)
    bar_idx: (list) of bars to color
    color: (str) color
    epsilon: (float) if provided, only color the portion of the bar on the right side of epsilon. 
    
    Returns
    -------
    ax
    """
    # given a plot of a barcode, color specific bars
    
    # find maximum death time (excluding Inf, if it exists)
    death_times = barcode[:,1]
    death_max = np.nanmax(death_times[death_times != np.inf])
    
    for i in bar_idx:
        left = barcode[i,0]
        right = barcode[i,1]
        
        # adjust the left end point (if epsilon is provided)
        if epsilon != None:
            left = max(left, epsilon)
        
        # adjust the right end point (if inf)
        if right == np.inf:
            right = death_max * 1.2
        
        
        ax.hlines(i, left, right, color = color, lw = lw)
    return ax

def plot_points(P, 
                Q, 
                ax = None, 
                P_color = "#2a9d8f", # dark green 
                Q_color =  "#fb8500", # dark yellow
                P_size = 100,
                Q_size = 100, 
                P_alpha = 1,
                Q_alpha = 1,
                P_label = "P",
                Q_label = "Q",
                P_marker = "o",
                Q_marker = "X",
                legend_size = 18,
                legend = True,
                legend_loc = "lower left"):
    """Plots the points on a 2D-square 
    
    key parameters
    --------------
    P: (array) of size (n, 2) or (n,3), where n is the number of points.
    Q: (array) of size (n, 2) or (n,3)
    
    Returns
    -------
    axes object for plotting
    """
    
    # create axes
    if ax is None:
        ax = plt.gca()
    
    # plot points
    ax.scatter(P[:,0], P[:,1], s = P_size, c = P_color, alpha = P_alpha, label = P_label, marker = P_marker, edgecolors = "black")
    ax.scatter(Q[:,0], Q[:,1], s = Q_size, c = Q_color, alpha = Q_alpha, label = Q_label, marker= Q_marker, edgecolors = "black")

    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if legend == True:
        ax.legend(loc = legend_loc, fontsize = legend_size)
    return ax


def plot_cycle(cycle, ax, points, cycle_color = "#ff1493", linewidth = 4):
    """plots cycle on a 2D square
    This function should be called after calling the `plot_points()` function.
    
    Parameters
    ----------
    cycle: (array) of shape (m, 2). cycle[i] = [x,y] indicates the presence of a 1-simplex [x,y]
    ax: (Axes object) output of the function `plot_square_torus()`
    points: (array) of shape (n, 2). Coordinates of the points on which the cycle lives.
    """
    # get theta, phi values
    x = points[:,0]
    y = points[:,1]
    
    # plot cycle 
    for simplex in cycle:
        v1, v2 = simplex
        v1_x, v1_y = x[v1-1], y[v1-1] # -1 is necessary because Julia indexing starts at 1
        v2_x, v2_y = x[v2-1], y[v2-1]

        ax.plot([v1_x, v2_x], [v1_y, v2_y], c = cycle_color, lw = linewidth)
    return ax

def plot_cycle_on_square_torus(cycle, ax, points, cycle_color = "#ff1493", linewidth = 4):
    """plots cycle on a square torus.
    This function should be called after calling the `plot_points()` function.
    
    Parameters
    ----------
    cycle: (array) of shape (m, 2). cycle[i] = [x,y] indicates the presence of a 1-simplex [x,y]
    ax: (Axes object) output of the function `plot_points()`
    points: (array) of shape (n, 2). Coordinates of the points on which the cycle lives.
    """
    # get theta, phi values
    theta = points[:,0]
    phi = points[:,1]
    
    # plot cycle 
    for simplex in cycle:
        v1, v2 = simplex
        v1_theta, v1_phi = theta[v1-1], phi[v1-1] # -1 is necessary because Julia indexing starts at 1
        v2_theta, v2_phi = theta[v2-1], phi[v2-1]

        # if the simplex doesn't cross the square boundary, plot it:
        if (abs(v1_theta - v2_theta) <= 2 ) and (abs(v1_phi - v2_phi) <= 2):
            ax.plot([v1_theta, v2_theta], [v1_phi, v2_phi], c = cycle_color, lw = linewidth)
    return ax

def get_classrep_locations(classrep, locations_sampled):
    """Given a classrep (saved from Eirene), get the locations of the classrep stimulus image"""
    return [(locations_sampled[i-1][0], locations_sampled[i-1][1]) for i in classrep]

def plot_locations(locations, size, cmap  = ListedColormap(["lightgrey", "salmon"])):
    """ Visualize the locations of stimulus or Gabor filters
    """
    loc = np.zeros((size, size))
    for item in locations:
        loc[item[0], item[1]] = 1

    plt.figure(figsize=(8,8))
    plt.pcolor(loc[::-1],edgecolors='k', linewidths=1, cmap = cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

    
def plot_stimulus_cyclerep_images(classrep, orientations_sampled, locations_sampled, wavelength, size, radius, phase):
    n = classrep.shape[0]
    nrows = math.ceil(n / 10)
    
    fig, ax = plt.subplots(nrows = nrows, ncols = 10, figsize = (20, nrows*2))
    
    for idx, i in enumerate(classrep):
        row = idx //10
        col = idx % 10
        img, _ = create_stimulus_image(size, wavelength, phase, radius, orientations_sampled[i-1], 
                                    x_loc = locations_sampled[i-1][0],
                                    y_loc = locations_sampled[i-1][1])
        ax[row][col].imshow(img, cmap = "Greys")
   
    plt.setp(ax, xticks=[], yticks=[])
    plt.show() 
    
    
    
def plot_V1_cyclerep_images(classrep, V1_locx, V1_locy, V1_size, V1_wavelength, V1_orientations, V1_phase, V1_sigma, V1_gamma):
    n = classrep.shape[0]
    nrows = math.ceil(n / 10)
    
    fig, ax = plt.subplots(nrows = nrows, ncols = 10, figsize = (20, nrows*2))
    
    for idx, i in enumerate(classrep):
        row = idx //10
        col = idx % 10
        real, _, _ = gf.create_gabor_filter(V1_size//2, V1_size//2, V1_wavelength, V1_orientations[i-1], V1_phase, V1_sigma, V1_gamma, V1_size)
        
        change_x = V1_locx[i-1] - V1_size//2
        change_y = V1_locy[i-1] - V1_size//2
        new_img, _, _ = gf.move_img(real, V1_size//2, V1_size//2, change_x, change_y)

        ax[row][col].imshow(new_img, cmap = "Greys")
   
    plt.setp(ax, xticks=[], yticks=[])
    plt.show() 
    
    
def plot_locations_gradation(locations, size):
    """ Visualize the locations of stimulus or Gabor filters
    """
    loc = np.zeros((size, size))
    for idx, item in enumerate(locations):
        loc[item[0], item[1]] = idx + 1

    plt.figure(figsize=(8,8))
    plt.pcolor(loc[::-1],edgecolors='k', linewidths=1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
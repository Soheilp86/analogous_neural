"""
Code for selecting significant points on a persistence diagram. 

@author Iris Yoon
irishryoon@gmail.com
"""
module PD_significance

include("../src/eirene_helper.jl")
using Eirene
using CSV
using DataFrames
using JLD
using HDF5
using MultivariateStats
using Plots
using Statistics
using StatsBase
using Random
using .helper

export get_lifetimes,
        get_ratios,
        get_relative_lifetimes,
        get_barlength_distribution,
        get_relative_barlength_distribution,
        get_empirical_relative_barlengths,
        select_persistent_intervals,
        select_persistent_intervals_with_sqrt,
        select_persistent_intervals_sub,
        select_intervals,
        plot_length_distributions,
        select_from_IQR,
        select_from_shuffled_IQR,
        select_from_shuffled_dissimilarity
        
###########################################################################
# get features from persistence diagrams
########################################################################### 

get_lifetimes(barcode_region) = barcode_region[:,2] - barcode_region[:,1]

get_ratios(barcode_region) = barcode_region[:,2] ./ barcode_region[:,1]

function get_relative_lifetimes(barcode_region; return_median = false)
    lifetimes = get_lifetimes(barcode_region)
    lifetime_med = median(lifetimes)
    
    if return_median == false
        return lifetimes ./ lifetime_med    
    else
        return lifetimes ./ lifetime_med, lifetime_med
    end
end




function select_persistent_intervals(data_barcode, D; dim = 1, n = 10, k = 3, use_min_length = false)
    
    # get relative lengths from given matrix
    interval_length = data_barcode[:,2] - data_barcode[:,1]
    if use_min_length == true
        min_length = minimum(interval_length)
        rel_length = interval_length ./ min_length
    else
        median_length = median(interval_length)
        rel_length = interval_length ./ median_length
    end
    
    # get empirical distribution of relative lengths from shuffled distances
    empirical_rel_lengths = get_empirical_relative_barlengths(D, n = n)
   
    # combine true and empirical relative lengths
    combined = vcat(rel_length, empirical_rel_lengths)
    
    # square-root transform
    empirical = empirical_rel_lengths
    data = rel_length
    
    # compute the points above the IQR range with given "k"
    _, Q1, Q2, Q3, _ = nquantile(empirical, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above the cutoff 
    selected_idx = [idx for (idx, v) in enumerate(data) if v > cutoff]
    
    return selected_idx, empirical, data, cutoff
    
end



function select_persistent_intervals_with_sqrt(data_barcode, D; dim = 1, n = 10, k = 3, use_min_length = false)
    
    # get relative lengths from given matrix
    interval_length = data_barcode[:,2] - data_barcode[:,1]
    if use_min_length == true
        min_length = minimum(interval_length)
        rel_length = interval_length ./ min_length
    else
        median_length = median(interval_length)
        rel_length = interval_length ./ median_length
    end
    
    # get empirical distribution of relative lengths from shuffled distances
    empirical_rel_lengths = get_empirical_relative_barlengths(D, n = n)
   
    # combine true and empirical relative lengths
    combined = vcat(rel_length, empirical_rel_lengths)
    
    # square-root transform
    sqrt_empirical = [sqrt(x) for x in empirical_rel_lengths]
    sqrt_data = [sqrt(x) for x in rel_length]
    
    # compute the points above the IQR range with given "k"
    _, Q1, Q2, Q3, _ = nquantile(sqrt_empirical, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above the cutoff 
    selected_idx = [idx for (idx, v) in enumerate(sqrt_data) if v > cutoff]
    
    return selected_idx, sqrt_empirical, sqrt_data, cutoff
    
end


function select_persistent_intervals_sub(data_barcode, D; dim = 1, n = 10, use_min_length = false)
    
    # get relative lengths from given matrix
    interval_length = data_barcode[:,2] - data_barcode[:,1]
    if use_min_length == true
        min_length = minimum(interval_length)
        rel_length = interval_length ./ min_length
    else
        median_length = median(interval_length)
        rel_length = interval_length ./ median_length
    end
    
    # get empirical distribution of relative lengths from shuffled distances
    empirical_rel_lengths = get_empirical_relative_barlengths(D, n = n)
   
    # combine true and empirical relative lengths
    combined = vcat(rel_length, empirical_rel_lengths)
    
    # square-root transform
    sqrt_combined = [sqrt(x) for x in combined]
    
    # compute the points above the IQR range with given "k"
    #_, Q1, Q2, Q3, _ = nquantile(sqrt_combined, 4)
    #cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above the cutoff 
    #selected_idx = [idx for (idx, v) in enumerate(sqrt_combined) if v > cutoff]
    
    return empirical_rel_lengths, combined#, cutoff
    
end




function plot_length_distributions(data_barcode, sampling_distribution, min_length; title1 = "data", title2= "shuffled", rel = false)
    if rel == "median"
        data_lengths = data_barcode[:,2] - data_barcode[:,1]
        data_med = median(data_lengths)
        data_lengths = data_lengths ./ data_med
    elseif rel == "min"
        data_lengths = data_barcode[:,2] - data_barcode[:,1]
        data_min = minimum(data_lengths)
        data_lengths = data_lengths ./ data_min
    else
        data_lengths = data_barcode[:,2] - data_barcode[:,1]     
    end      
    p1 = histogram(data_lengths, label = "", title =title1)
    p2 = histogram(sampling_distribution, label = "", title = title2)
    vline!(p1, [min_length, min_length], label = "")
    vline!(p2, [min_length, min_length], label = "")
    p = plot(p1, p2, layout = grid(1,2), figsize = (500, 500))
    return p
end


###########################################################################
# Selection from IQR 
###########################################################################

function select_from_IQR(barcode_region; k = 1.5, mode = "lifetimes")
    
    if mode == "lifetimes"
        features = get_lifetimes(barcode_region)
    elseif mode == "relative lifetimes"
        features = get_relative_lifetimes(barcode_region)
    elseif mode == "ratios"
        features = get_ratios(barcode_region)
    else
       print("mode should be one of the following: lifetimes, relative lifetimes, ratios") 
    end
             
    _, Q1, Q2, Q3, _ = nquantile(features, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above cutoff
    selected_idx = [idx for (idx, v) in enumerate(features) if v > cutoff]
    return selected_idx, cutoff    
end

function select_from_shuffled_IQR(D; k = 1.5, dim = 1, n = 10, alpha = 0.05, mode = "lifetimes")
    # compute barcode
    C = eirene(D, maxdim = dim)
    barc = barcode(C, dim = dim)
    
    # get data features
    if mode == "lifetimes"
        features = get_lifetimes(barc)
    elseif mode == "relative lifetimes"
        features = get_relative_lifetimes(barc)
    elseif mode == "ratios"
        features = get_ratios(barc)
    else
       print("mode should be one of the following: lifetimes, relative lifetimes, ratios") 
    end
    
    # get features from shuffled dissimilarity matrices
    shuffled_features = get_shuffled_features(D, n = n, mode = mode)
      
    # compute IQR
    _, Q1, Q2, Q3, _ = nquantile(shuffled_features, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above cutoff
    selected_idx = [idx for (idx, v) in enumerate(features) if v > cutoff]
    return selected_idx, cutoff, shuffled_features    
end


###########################################################################
# Selection from shuffled dissimilarity matrix
###########################################################################

function select_from_shuffled_dissimilarity(D; dim = 1, n = 10, alpha = 0.001, mode = "lifetimes")
    # compute barcode
    C = eirene(D, maxdim = dim)
    barc = barcode(C, dim = dim)
    
    if mode == "lifetimes"
        features = get_lifetimes(barc)
        cutoff, sample_distribution = get_barlength_distribution(D, alpha = alpha, n = n) 
        selected = [i for (i,v) in enumerate(features) if v >= cutoff]
        return selected, cutoff, sample_distribution
        
    elseif mode == "relative lifetimes"
        features = get_relative_lifetimes(barc)
        cutoff, sample_distribution = get_relative_barlength_distribution(D, alpha = alpha, n = n) 
        selected = [i for (i, v) in enumerate(features) if v >= cutoff]
        return selected, cutoff, sample_distribution
        
    elseif mode == "ratios"
        features = get_ratios(barc)
        cutoff, sample_distribution = get_ratios_distribution(D, alpha = alpha, n = n) 
        selected = [i for (i, v) in enumerate(features) if v >= cutoff]
        return selected, cutoff, sample_distribution
    else
       print("mode should be one of the following: lifetimes, relative lifetimes, ratios") 
    end  
end

function permute_symmetric_array(D)
    # Permute a symmetric (dissimilarity) array whose diagonals are 0. 
    # "D": distance array
    
    # Columns of relevant entries
    #row 1: 2, ... , n
    #row 2: 3, ... , n  
    #row 3: 4, ... , n
    #    ...
    # row n-1: n

    # which have the following indices:
    # 2, ... , n
    # n+3, ... , 2n
    # 2n + 4, ... , 3n, 
    # ...
    # (n-1)n

    n = size(D, 1)
    
    # extract above-diagonal elements in order
    elements = []
    for i=1:n
        row = D[i,i+1:end]
        append!(elements, row)
    end

    # Indices of above-diagonal elements
    indices = []
    for i =1:n-1
        row_indices = collect(n*(i-1)+(i+1):i*n)
        append!(indices, row_indices)
    end

    # permute indices 
    perm_indices = shuffle(indices)

    # array with permuted indices
    D_perm = zeros(n,n)
    setindex!(D_perm, elements, perm_indices)

    # turn D_perm into a symmetric matrix 
    D_perm = D_perm + transpose(D_perm);
    
    return D_perm
end

function get_shuffled_features(D; n = 10, mode = "lifetimes")
    features = []
    for i = 1:n
        D_perm = permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        
        if mode == "lifetimes"
            features_i = get_lifetimes(barc)

        elseif mode == "relative lifetimes"
            features_i = get_relative_lifetimes(barc)

        elseif mode == "ratios"
            features_i = get_ratios(barc)
        else
           print("mode should be one of the following: lifetimes, relative lifetimes, ratios") 
        end  
        append!(features, features_i)
    end
    return features
end


function get_barlength_distribution(D; alpha = 0.05, n = 10)
    # find empirical distribution of bar lengths 
    # (permute matrix D and compute persistençce) x 10   
    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1]  
        append!(length_sample, bar_lengths)
    end

    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    length_param = quantile(length_sample, 1-alpha)

    return length_param, length_sample

    # To plot histogram of sampled bar_lengths:
    #histogram(length_sample, label="", title="Histogram of bar lengths of shuffled distance", color = :grey)
    #vline!([length_param], label = "alpha = 0.05", linewidth = 5)
end


function get_relative_barlength_distribution(D; alpha = 0.05, n = 10)
    # find empirical distribution of bar lengths relative to the median bar length 
    # (permute matrix D and compute persistençce) x 10   
    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1] 
        bar_median = median(bar_lengths)
        append!(length_sample, bar_lengths ./ bar_median)
    end

    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    length_param = quantile(length_sample, 1-alpha)

    return length_param, length_sample

    # To plot histogram of sampled bar_lengths:
    #histogram(length_sample, label="", title="Histogram of bar lengths of shuffled distance", color = :grey)
    #vline!([length_param], label = "alpha = 0.05", linewidth = 5)

end



function get_ratios_distribution(D; alpha = 0.05, n = 10)
    # find empirical distribution of bar lengths relative to the median bar length 
    # (permute matrix D and compute persistençce) x 10   
    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        ratios = get_ratios(barc)
        append!(length_sample, ratios)
    end

    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    length_param = quantile(length_sample, 1-alpha)

    return length_param, length_sample

    # To plot histogram of sampled bar_lengths:
    #histogram(length_sample, label="", title="Histogram of bar lengths of shuffled distance", color = :grey)
    #vline!([length_param], label = "alpha = 0.05", linewidth = 5)

end


function get_empirical_relative_barlengths(D; n = 10)
    # find empirical distribution of bar lengths relative to the median bar length 
    # (permute matrix D and compute persistençce) x 10   
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = ext_var.permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1] 
        bar_median = median(bar_lengths)
        append!(length_sample, bar_lengths ./ bar_median)
    end

    return length_sample

end



"""
    select_intervals(D; <keyword arguments>)
Select intervals of a barcode with long enough length. Performs the following. \n
1. Get empirical distribution of interval lengths by permuting the distance matrix `D`. \n
2. Select `min_length` at which ``P(x \\geq min\\_length) \\leq \\alpha`` for some value of ``\\alpha`` (defaults to 0.05). \n
3. Select intervals of barcode(Y) whose length is greater than or equal to `min_length` \n

### Arguments
- `D::Array` : Distance matrix
- `dim::Int64=1`: Dimension
- `n::Int64=10`: number of times to shuffle the distance matrix
- `relative::bool = true`: whether to consider relative bar lengths or absolute bar lengths
- `use_min_length::bool = false`: When `relative = true` and `use_min_length = true`, consider bar lengths relative to median. If `relative = true` and `use_min_length = false`, consider bar lengths relative to the minimum bar length. Set this parameter to true when your barcode has few bars. 

### Outputs
- `selected::Array` : List of indices corresponding to selected intervals of the barcode. 
"""
function select_intervals(D; dim = 1, n = 10, alpha = 0.001, relative = true, use_min_length = false)
    # get barcode
    C = eirene(D, maxdim = dim)
    barc = barcode(C, dim = dim)
    interval_length = barc[:,2] - barc[:,1]
    
    # empirical distribution of relative length
    if relative == true
        min_length, sample_distribution = get_relative_barlength_distribution(D, alpha = alpha, n = n) 
        

        if use_min_length == true
            median_length = minimum(interval_length)
        else
            median_length = median(interval_length)
        end
        
        rel_length = interval_length ./ median_length
        selected = [i for (i, v) in enumerate(rel_length) if v >= min_length]

        return selected, min_length, sample_distribution
    # get empirical distribution of absolute length
    else
        min_length, sample_distribution = ext_var.get_barlength_distribution(D, alpha = alpha, n = n) 
        selected = [i for (i,v) in enumerate(interval_length) if v >= min_length]
        return selected, min_length, sample_distribution
    end
    
end



end
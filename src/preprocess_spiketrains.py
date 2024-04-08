"""
Functions used for preprocessing spike trains

@author: Iris Yoon (irishryoon@gmail.com)
"""

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


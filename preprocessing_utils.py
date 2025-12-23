import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy

def estimate_speed_2d(x_pos, y_pos, T):
    speeds = np.zeros(x_pos.shape)
    
    # Compute speed using central differences for the middle points
    for i in range(1, len(x_pos) - 1):
        dx = x_pos[i+1] - x_pos[i-1]
        dy = y_pos[i+1] - y_pos[i-1]
        dt = T[i+1] - T[i-1]
        speeds[i] = np.sqrt(dx**2 + dy**2) / dt
    
    # For the first point, use forward difference
    dx_forward = x_pos[1] - x_pos[0]
    dy_forward = y_pos[1] - y_pos[0]
    dt_forward = T[1] - T[0]
    speeds[0] = np.sqrt(dx_forward**2 + dy_forward**2) / dt_forward
    
    # For the last point, use backward difference
    dx_backward = x_pos[-1] - x_pos[-2]
    dy_backward = y_pos[-1] - y_pos[-2]
    dt_backward = T[-1] - T[-2]
    speeds[-1] = np.sqrt(dx_backward**2 + dy_backward**2) / dt_backward
    
    return speeds

def create_spike_raster(spike_time, spike_neu_id, num_neurons, time_bin_edges):
    spike_raster = np.zeros((len(time_bin_edges) - 1, num_neurons))
    
    # Filter spikes that are within the period of interest
    valid_spike_indices = np.where((spike_time >= time_bin_edges[0]) & (spike_time <= time_bin_edges[-1]))[0]
    filtered_spike_time = spike_time[valid_spike_indices]
    filtered_spike_neu_id = spike_neu_id[valid_spike_indices]

    # Find the bin index for each filtered spike time
    bin_indices = np.digitize(filtered_spike_time, time_bin_edges) - 1 # -1 to correct the bin index
    assert np.all(bin_indices >= 0) and np.all(bin_indices < len(time_bin_edges) - 1)

    np.add.at(spike_raster, (bin_indices.astype(np.int64), filtered_spike_neu_id.astype(np.int64)), 1)
    return spike_raster

def calc_place_fields_1d(
        spike_raster, pos, speed,
        pos_bin_edges, pos_bin_size, time_bin_size,
        speed_threshold=5, kernel_wid=4):
    assert spike_raster.shape[0] == pos.shape[0] == speed.shape[0]
    num_timesteps = spike_raster.shape[0]
    num_neurons = spike_raster.shape[1]
    spike_counts = np.zeros((len(pos_bin_edges) - 1, num_neurons))
    pos_counts = np.zeros((len(pos_bin_edges) - 1))

    for timestep in range(num_timesteps):
        if speed[timestep] >= speed_threshold: # only consider spikes during moving
            pos_bin_index = np.digitize(pos[timestep], pos_bin_edges) - 1 # -1 to correct the bin index
            assert pos_bin_index >= 0 and pos_bin_index < len(pos_bin_edges) - 1

            pos_counts[pos_bin_index] += 1
            spike_counts[pos_bin_index, :] += spike_raster[timestep, :]
    
    # Normalize place fields with position counts (spatial occupancy)
    pos_counts += 1e-15 # counts can be zero
    place_fields = spike_counts / (time_bin_size * np.expand_dims(pos_counts, axis=1))

    # Gaussian smoothing
    sigma_in_bins = kernel_wid / pos_bin_size
    for neuron in range(num_neurons):
        place_fields[:, neuron] = gaussian_filter1d(place_fields[:, neuron], sigma=sigma_in_bins)
    
    return place_fields

def calc_place_fields_1d_dir(
        spike_raster, pos, speed,
        pos_bin_edges, pos_bin_size, time_bin_size,
        speed_threshold=5, kernel_wid=4):
    assert spike_raster.shape[0] == pos.shape[0] == speed.shape[0]
    T, N = spike_raster.shape
    n_bins = len(pos_bin_edges) - 1

    # instantaneous signed velocity
    vel = np.zeros(T)
    vel[0] = (pos[1] - pos[0]) / time_bin_size
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * time_bin_size)
    vel[-1] = (pos[-1] - pos[-2]) / time_bin_size
    # vel = gaussian_filter1d(vel, sigma=0.1)
    # masks
    is_move = speed >= speed_threshold
    is_LtoR = is_move & (vel > 0)
    is_RtoL = is_move & (vel < 0)
    # rates
    spikes_LtoR = np.zeros((n_bins, N))
    spikes_RtoL = np.zeros((n_bins, N))
    occ_LtoR = np.zeros(n_bins)
    occ_RtoL = np.zeros(n_bins)
    for t in range(T):
        b = np.digitize(pos[t], pos_bin_edges) - 1
        assert b >= 0 and b < n_bins
        if is_LtoR[t]:
            occ_LtoR[b] += 1
            spikes_LtoR[b, :] += spike_raster[t, :]
        elif is_RtoL[t]:
            occ_RtoL[b] += 1
            spikes_RtoL[b, :] += spike_raster[t, :]
    eps = 1e-15
    rates_LtoR = spikes_LtoR / ((occ_LtoR + eps)[:, None] * time_bin_size)
    rates_RtoL = spikes_RtoL / ((occ_RtoL + eps)[:, None] * time_bin_size)
    # smooth
    sigma_in_bins = kernel_wid / pos_bin_size
    for i in range(N):
        rates_LtoR[:, i] = gaussian_filter1d(rates_LtoR[:, i], sigma_in_bins)
        rates_RtoL[:, i] = gaussian_filter1d(rates_RtoL[:, i], sigma_in_bins)

    return rates_LtoR, rates_RtoL
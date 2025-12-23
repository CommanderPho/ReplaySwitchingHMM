import os
import logging
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from ssm.ssm import SSM
from ssm.optimization import perform_inference, neg_loglik
from ssm.utils import (
    ic_uniform,
    dt_diagonal,
    ct_fragmented, ct_stationary, ct_driftdiffusion)

###-----Setup-----###
np.random.seed(0)

######
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat1/Linear1'
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
# MULTIRUN = False
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear2')
# MULTIRUN = False
###
# data_folder =  os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear3')
# MULTIRUN = True
###
data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear4')
MULTIRUN = True
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear1')
# MULTIRUN = False
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear2')
# MULTIRUN = False
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat3/Linear3')
# MULTIRUN = True
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear3')
# MULTIRUN = False
###
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear4')
# MULTIRUN = False
######

save_folder = './temp'

###-----Retrieve Data-----###
# Preprocessed place fields
loaded_data = np.load(os.path.join(data_folder, 'processed/processed_data.npz'), allow_pickle=True)
place_fields = loaded_data['place_fields']
pos_bin_size = loaded_data['pos_bin_size']
pos_bin_edges = loaded_data['pos_bin_edges']
pos_bin_centers = loaded_data['pos_bin_centers']
time_bin_edges_run = loaded_data['time_bin_edges']
loaded_data.close()

# Preprocessed sharp-wave ripples
loaded_data = np.load(os.path.join(data_folder, 'processed/10msBin_10msAdvance/ripples.npz'), allow_pickle=True)
time_bin_centers_replay = loaded_data['time_bin_centers_replay']
start_ids = loaded_data['start_ids']
end_ids = loaded_data['end_ids']
start_times = loaded_data['start_times']
end_times = loaded_data['end_times']
ripples = loaded_data['ripples']
loaded_data.close()

# Retrieve spike rasters for SWRs
# 1. Position bin size: change unit cm -> meter
pos_bin_size = pos_bin_size/100

# 2. Time bin size
time_bin_size_replay = 0.01
    # np.mean(time_bin_centers_replay[1:] - time_bin_centers_replay[0:-1])
assert (time_bin_centers_replay[1] - time_bin_centers_replay[0]) - time_bin_size_replay < 1e-5

# 3. Place fields
place_fields = place_fields + 1e-10
    # shape (n_pos_bins, n_neurons)

# 4. Sharp-wave ripples
#    - remove leading and trailing zero spike timesteps
#    - remove leading and trailing low-firing timesteps
#    - at least 50 ms
TIME_LEN_THRE = 0.05 # sec
FIRE_THRE = 2 # spikes per second per neuron
ripple_spike_trains = []
start_end_times = []
ripple_realtime = []
###### Remove flanking no-spike timesteps
# for i in range(len(ripples)):
#     ripple = ripples[i]
#     spike_sum = np.sum(ripple, axis=1)
#     non_zero = (spike_sum > 0)
#     if np.sum(non_zero) > 0:
#         first_non_zero = np.argmax(non_zero)
#         last_non_zero = len(non_zero) - 1 - np.argmax(non_zero[::-1])
#         if time_bin_size_replay * (last_non_zero + 1 - first_non_zero) >= TIME_LEN_THRE:
#             pop_burst = ripple[first_non_zero:last_non_zero+1, :]
#                 # population burst, shape (n_time, n_neurons)
#             ripple_spike_trains.append(pop_burst)
###### Remove flanking low-firing timesteps
for i in range(len(ripples)):
    ripple = ripples[i]
    spike_sum = np.sum(ripple, axis=1)
    high_fire = (spike_sum / (ripple.shape[1] * time_bin_size_replay)) > FIRE_THRE
    if np.sum(high_fire) >= 5: # at least 5 high firing timesteps
        first_high_fire = np.argmax(high_fire)
        last_high_fire = len(high_fire) - 1 - np.argmax(high_fire[::-1])
        if time_bin_size_replay * (last_high_fire + 1 - first_high_fire) >= TIME_LEN_THRE:
            pop_burst = ripple[first_high_fire:last_high_fire+1, :]
                # population burst, shape (n_time, n_neurons)
            ripple_spike_trains.append(pop_burst)
            start_end_times.append((start_times[i], end_times[i]))

            timeid = np.arange(start_ids[i], end_ids[i]+1)
            timeid = timeid[first_high_fire:last_high_fire+1]
            ripple_realtime.append(time_bin_centers_replay[timeid])
# object_array = np.empty(len(ripple_realtime), dtype=object)
# object_array[:] = ripple_realtime
# ripple_realtime = object_array
# np.save('./temp/ripple_realtime.npy', ripple_realtime)
# raise EOFError
######

# 5. Create arrays of replay spike raster
#    numpy attempts to infer a more specific array structure than dtype=object
object_array = np.empty(len(ripple_spike_trains), dtype=object)
object_array[:] = ripple_spike_trains
ripple_spike_trains = object_array

# Count preplay events
if MULTIRUN:
    run_times = \
        np.array([[tber[0], tber[-1]] for tber in time_bin_edges_run]).ravel()
    start_end_times = np.array(start_end_times)
    edge_event_indices = [
        np.argmin(np.abs(start_end_times[:, 1] - rt))
        for rt in run_times]
    edge_event_indices = sorted(set(edge_event_indices))
    print(edge_event_indices)
else:
    run_start_time, run_end_time = \
        time_bin_edges_run[0], time_bin_edges_run[-1]
    start_end_times = np.array(start_end_times)
    print('# Preplay: ', np.sum(start_end_times[:, 1] < run_start_time))
    print('# Awake replay: ', np.sum((start_end_times[:, 1] >= run_start_time) & (start_end_times[:, 1] < run_end_time)))
    print('# Replay: ', np.sum(start_end_times[:, 1] >= run_end_time))
# raise EOFError

###-----Optimization-----###
diag_lower, diag_higher = 0.50, 0.99        # 0.1, 0.99
lamb1_lower, lamb1_higher = -750., 750.     # 0., 1500. lambda = 750 -> 750 * spatial bin * time bin = 15 m/s * time bin
lamb2_lower, lamb2_higher = -750., 750.     # not used
sig_lower, sig_higher = 5., 50.             # sigma = 10 -> standard deviation 10 * spatial bin * sqrt(time bin)
initial_diag, initial_lamb1, initial_lamb2, initial_sig = 0.98, 0., 0., 20.
bounds = \
    [(diag_lower, diag_higher),
     (lamb1_lower, lamb1_higher),
     (lamb2_lower, lamb2_higher),
     (sig_lower, sig_higher)]
initial_guess = [initial_diag, initial_lamb1, initial_lamb2, initial_sig]
model_type = 'hybrid_drift_diffusion'
# model_type = 'drift_diffusion'
# model_type = 'diffusion'
# model_type = 'hybrid_diffusion'
# model_type = 'frag_stat'
# model_type = 'frag'
# model_type = 'stat'

optim_results = np.zeros((len(ripple_spike_trains), 5))
for ripple_i in tqdm(range(len(ripple_spike_trains))):
    ripple_spike_train = ripple_spike_trains[ripple_i]
    result = minimize(
        neg_loglik, initial_guess, bounds=bounds,
        args=(place_fields, ripple_spike_train, time_bin_size_replay, model_type),
        method='L-BFGS-B')
    optim_params = result.x
    optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
        perform_inference(
            optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
    optim_results[ripple_i, 0] = data_log_likelihood
    optim_results[ripple_i, 1:5] = optim_params # diag, lambda1, lambda2, sigma
    
###-----Save Results-----###
np.savez_compressed(
    os.path.join('./temp', 'optim_results.npz'),
    place_fields=place_fields,
    ripple_spike_trains=ripple_spike_trains,
    time_window=time_bin_size_replay,
    position_window=pos_bin_size,
    optim_results=optim_results)
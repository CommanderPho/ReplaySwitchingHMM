import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import logging
import itertools
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

np.random.seed(2024)

###-----Simulation settings-----###
###### Parameter identifiability (default)
n_pos_bins = 200        # 4.0 m track; 2 cm position bin size
n_neurons = 300
peak_firing_rate = 20   # Hz
std_dev = 5             # position bin
noise_std_dev = 1e-5    # Hz
dt = 0.01               # sec
###
LAMB_PLACEHOLDER, SIG_PLACEHOLDER = 0, 10
diag_vals = [0.9999999]
x0_to_grids = {
    30:  ([0, 100, 200, 300, 400], [10, 20]),
    100: ([0, 100], [5, 10, 15, 20, 25])}
dyn_vals = [0, 1, 2] # 0 = drift-diffusion, 1 = uniform, 2 = stationary
n_time = 60
###
rows = []
for diag in diag_vals:
    for x0, (lamb_grid, sig_grid) in x0_to_grids.items():
        for I1 in dyn_vals:
            for I2 in dyn_vals:
                if I1 == I2:
                    # keep I1 != I2
                    continue
                if (I1 != 0) and (I2 != 0):
                    # neither segment is drift–diffusion -> don't factorize (lambda, sigma)
                    rows.append((diag, LAMB_PLACEHOLDER, SIG_PLACEHOLDER, I1, I2, x0, n_time))
                else:
                    # at least one segment is drift–diffusion -> factorize over (lambda, sigma)
                    for lamb in lamb_grid:
                        for sig in sig_grid:
                            rows.append((diag, lamb, sig, I1, I2, x0, n_time))
gt_settings = np.array(rows, dtype=float)
    # diag_gt, lamb_gt, sig_gt, I1, I2, x0, n_time
###
n_ripples = 100
    # number of repeats
gt_settings = np.repeat(gt_settings, n_ripples, axis=0)
###### More strict settings
# n_pos_bins = 100        # 2.0 m track; 2 cm position bin size
# n_neurons = 120
# peak_firing_rate = 20   # Hz
# std_dev = 5             # position bin
# noise_std_dev = 1e-5    # Hz
# dt = 0.01               # sec
# ###
# LAMB_PH, SIG_PH = 0, 10 # place holder
# diag_vals = [0.9999999]
# x0_to_grids = {
#     15: ([0, 100, 200, 300, 400], [10, 20]),
#     50: ([0, 100], [5, 10, 15, 20, 25])}
# dyn_vals = [0, 1, 2] # 0 = drift-diffusion, 1 = uniform, 2 = stationary
# n_time = 30
# ###
# rows = []
# for diag in diag_vals:
#     for x0, (lamb_grid, sig_grid) in x0_to_grids.items():
#         for I1 in dyn_vals:
#             for I2 in dyn_vals:
#                 if I1 == I2:
#                     # keep I1 != I2
#                     continue
#                 if (I1 != 0) and (I2 != 0):
#                     # neither segment is drift–diffusion -> don't factorize (lambda, sigma)
#                     rows.append((diag, LAMB_PH, SIG_PH, I1, I2, x0, n_time))
#                 else:
#                     # at least one segment is drift–diffusion -> factorize over (lambda, sigma)
#                     for lamb in lamb_grid:
#                         for sig in sig_grid:
#                             rows.append((diag, lamb, sig, I1, I2, x0, n_time))
# gt_settings = np.array(rows, dtype=float)
#     # diag_gt, lamb_gt, sig_gt, I1, I2, x0, n_time
# ###
# n_ripples = 100
#     # number of repeats
# gt_settings = np.repeat(gt_settings, n_ripples, axis=0)
######

###-----Specify environment and place fields-----###
place_fields_sim = np.zeros((n_pos_bins, n_neurons))
peak_locations = []
for neuron in range(n_neurons):
    center = np.random.randint(n_pos_bins)
    peak_locations.append(center)
    positions = np.arange(n_pos_bins)
    place_fields_sim[:, neuron] = \
        peak_firing_rate * np.exp(-((positions - center) ** 2) / (2 * std_dev ** 2))

# Sort neurons
sorted_indices = np.argsort(peak_locations)
place_fields_sim = place_fields_sim[:, sorted_indices]

# Add noise
noise = np.random.normal(0, noise_std_dev, place_fields_sim.shape)
place_fields_sim += noise
place_fields_sim = np.clip(place_fields_sim, 0, None)

###-----Simulate trajectory and spike rasters-----###
sim_spike_trains = []
for gt_set in gt_settings:
    diag_gt, lamb_gt, sig_gt, I1, I2, x0, n_time = gt_set
    I1, I2, x0, n_time = int(I1), int(I2), int(x0), int(n_time)

    ###-----Specify ground-truth model-----###
    # diag_gt, lamb_gt, sig_gt = 0.9999999, 0, 30
    gamma = 1.0
    n_states = 3
    ic_type = ic_uniform()
    dt_type = dt_diagonal(diagonal_value=diag_gt)
    ct_row = np.array(
        [ct_driftdiffusion(lamb=lamb_gt, sig=sig_gt),
        ct_fragmented(),
        ct_stationary()], dtype=object)
    ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    model_gt = SSM(place_fields_sim, dt, gamma, ic_type, dt_type, ct_types)

    ###-----Simulate trajectories and spikes-----###
    ########## use sampled locations
    # Simulate trajectory
    discrete_states = [I1]
    locations = [x0]
    for t in range(1, n_time):
        I_prev = discrete_states[-1]
        if t < n_time/2:
            I_next = I1
        else:
            I_next = I2
        if (t == n_time/2) and (I1 == 1) and (I2 == 0):
            locations[-1] = x0
        discrete_states.append(I_next)

        x_prev = locations[-1]
        location_probs = model_gt.continuous_state_transition[I_prev, I_next, x_prev, :]
        location_probs /= location_probs.sum()
        x_next = np.random.choice(range(n_pos_bins), p=location_probs)
        locations.append(x_next)
    discrete_states = np.array(discrete_states)
    locations = np.array(locations)

    # Simulate replay spike raster
    spike_raster = np.zeros((n_time, n_neurons))
    for t in range(n_time):
        x_t = locations[t]
        firing_rates = place_fields_sim[x_t, :]
        spike_raster[t, :] = np.random.poisson(firing_rates * dt)
    ########## use distribution of locations
    # deprecated
    ##########

    sim_spike_trains.append(spike_raster)
# detour because numpy attempts to
# infer a more specific array structure than dtype=object
object_array = np.empty(len(sim_spike_trains), dtype=object)
object_array[:] = sim_spike_trains
sim_spike_trains = object_array

###-----Optimization-----###
diag_lower, diag_higher = 0.50, 0.99
lamb1_lower, lamb1_higher = -750., 750. # 0., 1500.
lamb2_lower, lamb2_higher = -750., 750. # not used
sig_lower, sig_higher = 5., 50.
initial_diag, initial_lamb1, initial_lamb2, initial_sig = 0.98, 0., 0., 20.
bounds = \
    [(diag_lower, diag_higher),
     (lamb1_lower, lamb1_higher),
     (lamb2_lower, lamb2_higher),
     (sig_lower, sig_higher)]
initial_guess = [initial_diag, initial_lamb1, initial_lamb2, initial_sig]
model_type = 'hybrid_drift_diffusion'

time_bin_size_replay = dt
ripple_spike_trains = sim_spike_trains
place_fields = place_fields_sim

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
    os.path.join('./temp2', 'optim_results.npz'),
    gt_settings=gt_settings,
    place_fields=place_fields,
    ripple_spike_trains=ripple_spike_trains,
    time_window=time_bin_size_replay,
    optim_results=optim_results)
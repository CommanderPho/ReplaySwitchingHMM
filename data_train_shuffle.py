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

###-----Settings-----###
np.random.seed(2024)

######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
# result_folder = os.path.join('results/dynamics_models/hybrid_drift_diffusion_nonorm/optimize_lambda')
# result_folder = os.path.join('results2/dynamics_models/hybrid_drift_diffusion')
# result_folder = os.path.join('results0505/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 420, 232, 1852
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear2')
# result_folder = os.path.join('results0505/rat1_linear2/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 776, 157, 1271
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear3')
# result_folder = os.path.join('results0505/rat1_linear3/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 812, 10, 10 # 812, 870, 1537, 1732, 2681, 2794
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear4')
# result_folder = os.path.join('results0505/rat1_linear4/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 575, 10, 10 # 575, 652, 1836, 2035, 3600, 3676
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear1')
# result_folder = os.path.join('results0505/rat2_linear1/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 631, 41, 553
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear2')
# result_folder = os.path.join('results0505/rat2_linear2/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 734, 45, 1013
######
data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat3/Linear3')
result_folder = os.path.join('results0505/rat3_linear3/dynamics_models/hybrid_drift_diffusion')
num_preplay, num_awake_replay, num_post_replay = 0, 10, 10 # 0, 15, 16, 66, 67, 89
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear3')
# result_folder = os.path.join('results0505/rat5_linear3/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 1428, 47, 1435
######
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear4')
# result_folder = os.path.join('results0505/rat5_linear4/dynamics_models/hybrid_drift_diffusion')
# num_preplay, num_awake_replay, num_post_replay = 1373, 83, 1541
######

save_folder = './temp'

###-----Retrieve results-----###
loaded_data = np.load(os.path.join(result_folder, 'optim_results.npz'), allow_pickle=True)
optim_results_hdd = loaded_data['optim_results']
place_fields = loaded_data['place_fields']
ripple_spike_trains = loaded_data['ripple_spike_trains']
time_bin_size_replay = loaded_data['time_window']
pos_bin_size = loaded_data['position_window']
loaded_data.close()
    # see train_Pfeiffer1D.py for how to get these numbers

###-----Shuffle-----###
def shuffle_timebin(raster, num_shuffles=10):   
    shuffled_replay = np.empty((num_shuffles, len(raster)), dtype=object)
    for i, spike_raster in enumerate(raster):
        for j in range(num_shuffles):
            shuffled_raster = \
                spike_raster[np.random.permutation(spike_raster.shape[0]), :]
            shuffled_replay[j, i] = shuffled_raster
    return shuffled_replay.ravel()

def shuffle_neuron(raster, num_shuffles=10):   
    shuffled_replay = np.empty((num_shuffles, len(raster)), dtype=object)
    for i, spike_raster in enumerate(raster):
        for j in range(num_shuffles):
            shuffled_raster = \
                spike_raster[:, np.random.permutation(spike_raster.shape[1])]
            shuffled_replay[j, i] = shuffled_raster
    return shuffled_replay.ravel()

##########
# shuffle_raster_array = shuffle_timebin
shuffle_raster_array = shuffle_neuron
##########

preplay_raster = ripple_spike_trains[0:num_preplay]
replay_raster = ripple_spike_trains[num_preplay:]
preplay_shuffled = \
    shuffle_raster_array(preplay_raster, num_shuffles=120)
    # shuffle_raster_array(preplay_raster, num_shuffles=100)
replay_shuffled = \
    shuffle_raster_array(replay_raster, num_shuffles=120)
    # shuffle_raster_array(replay_raster, num_shuffles=100)
# print(preplay_shuffled.shape)
# print(replay_shuffled.shape)

###-----Optimization-----###
diag_lower, diag_higher = 0.50, 0.99        # 0.1, 0.99
lamb1_lower, lamb1_higher = -750., 750.     # 0., 1500.
lamb2_lower, lamb2_higher = -750., 750.     # not used
sig_lower, sig_higher = 5., 50.
initial_diag, initial_lamb1, initial_lamb2, initial_sig = 0.98, 0., 0., 20.
bounds = \
    [(diag_lower, diag_higher),
     (lamb1_lower, lamb1_higher),
     (lamb2_lower, lamb2_higher),
     (sig_lower, sig_higher)]
initial_guess = [initial_diag, initial_lamb1, initial_lamb2, initial_sig]
model_type = 'hybrid_drift_diffusion'
# model_type = 'drift_diffusion'
# model_type = 'hybrid_diffusion'
# model_type = 'frag_stat'
# model_type = 'frag'
# model_type = 'stat'

# 1. Preplay shuffled
optim_results_pre = np.zeros((len(preplay_shuffled), 5))
for ripple_i in tqdm(range(len(preplay_shuffled))):
    ripple_spike_train = preplay_shuffled[ripple_i]
    result = minimize(
        neg_loglik, initial_guess, bounds=bounds,
        args=(place_fields, ripple_spike_train, time_bin_size_replay, model_type),
        method='L-BFGS-B')
    optim_params = result.x
    optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
        perform_inference(
            optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
    optim_results_pre[ripple_i, 0] = data_log_likelihood
    optim_results_pre[ripple_i, 1:5] = optim_params # diag, lambda1, lambda2, sigma

# 2. Replay shuffled
optim_results_post = np.zeros((len(replay_shuffled), 5))
for ripple_i in tqdm(range(len(replay_shuffled))):
    ripple_spike_train = replay_shuffled[ripple_i]
    result = minimize(
        neg_loglik, initial_guess, bounds=bounds,
        args=(place_fields, ripple_spike_train, time_bin_size_replay, model_type),
        method='L-BFGS-B')
    optim_params = result.x
    optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
        perform_inference(
            optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
    optim_results_post[ripple_i, 0] = data_log_likelihood
    optim_results_post[ripple_i, 1:5] = optim_params # diag, lambda1, lambda2, sigma

###-----Save Results-----###
np.savez_compressed(
    os.path.join('./temp3', 'optim_results_shuffled.npz'),
    place_fields=place_fields,
    preplay_shuffled=preplay_shuffled,
    replay_shuffled=replay_shuffled,
    time_window=time_bin_size_replay,
    position_window=pos_bin_size,
    optim_results_pre=optim_results_pre,
    optim_results_post=optim_results_post)
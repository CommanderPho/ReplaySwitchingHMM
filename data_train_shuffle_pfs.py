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
# num_preplay, num_awake_replay, num_post_replay = 812, 10, 10 # 812, 870, 1537, 1732, 2681, 279
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

###-----Shuffle-----###
def shuffle_place_fields(place_fields, num_shuffles=10):
    n_bins, n_neurons = place_fields.shape
    null_place_fields = np.empty((num_shuffles, n_bins, n_neurons), dtype=place_fields.dtype)
    for i in range(num_shuffles):
        shifts = np.random.randint(0, n_bins, size=n_neurons)
        shuffled = np.empty_like(place_fields)
        for j in range(n_neurons):
            shuffled[:, j] = np.roll(place_fields[:, j], shifts[j])
        null_place_fields[i] = shuffled
    return null_place_fields

num_shuffles = 120
null_place_fields = shuffle_place_fields(place_fields, num_shuffles)

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

num_events = len(ripple_spike_trains)
optim_results = np.zeros((num_shuffles, num_events, 5))
for j in range(num_shuffles):
    print('Shuffle index:', j+1, '/', num_shuffles)
    pfs_null = null_place_fields[j, :, :]
    for ripple_i in tqdm(range(num_events)):
        ripple_spike_train = ripple_spike_trains[ripple_i]
        result = minimize(
            neg_loglik, initial_guess, bounds=bounds,
            args=(pfs_null, ripple_spike_train, time_bin_size_replay, model_type),
            method='L-BFGS-B')
        optim_params = result.x
        optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
            perform_inference(
                optim_params, pfs_null, ripple_spike_train, time_bin_size_replay, model_type)
        optim_results[j, ripple_i, 0] = data_log_likelihood
        optim_results[j, ripple_i, 1:5] = optim_params # diag, lambda1, lambda2, sigma

###-----Save Results-----###
np.savez_compressed(
    os.path.join('./temp', 'optim_results_shuffled.npz'),
    place_fields=place_fields,
    null_place_fields=null_place_fields,
    time_window=time_bin_size_replay,
    position_window=pos_bin_size,
    optim_results=optim_results)
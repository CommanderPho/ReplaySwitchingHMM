import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

import pickle
from collections import Counter
from tqdm import tqdm
import torch
import numpy as np
from scipy import special
from scipy.io import loadmat
from scipy.integrate import cumtrapz
from scipy.stats import norm, levy_stable, gaussian_kde
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from ssm.optimization import perform_inference
from ssm.baseline import (
    bayesian_decoding,
    bayesian_decoding_smooth,
    bayesian_decoding_acausal)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Liberation Sans'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.titlesize'] = 20 # 'large'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.labelpad'] = 7.0 # 4.0
mpl.rcParams['axes.linewidth'] = 1.6 # 0.8
mpl.rcParams['axes.labelsize'] = 15 # 'medium'
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titlesize'] = 20 # 'large'
mpl.rcParams['axes.titleweight'] = 'bold' # 'normal'
mpl.rcParams['xtick.labelsize'] = 12 # 'medium'
mpl.rcParams['ytick.labelsize'] = 12 # 'medium'
mpl.rcParams['xtick.major.width'] = 1.6 # 0.8
mpl.rcParams['xtick.minor.width'] = 1.2 # 0.6
mpl.rcParams['ytick.major.width'] = 1.6 # 0.8
mpl.rcParams['ytick.minor.width'] = 1.2 # 0.6
mpl.rcParams['xtick.major.size'] = 5.0 # 3.5
mpl.rcParams['xtick.minor.size'] = 4.0 # 2.0
mpl.rcParams['ytick.major.size'] = 5.0 # 3.5
mpl.rcParams['ytick.minor.size'] = 4.0 # 2.0
mpl.rcParams['legend.frameon'] = False # True
mpl.rcParams['legend.fontsize'] = 12 # 'medium'

###-----Replay position of stationary events-----###
if 0:
    # Retrieve awake ripples
    epochs = [
        [420, 232, 1852],
        [776, 157, 1271],
        [812, 870, 1537, 1732, 2681, 2794],
        [575, 652, 1836, 2035, 3600, 3676],
        [631, 41, 553],
        [734, 45, 1013],
        [0, 15, 16, 66, 67, 89],
        [1428, 47, 1435],
        [1373, 83, 1541]]
    awake_ripples = []
    for epoch in epochs:
        if len(epoch) == 3:
            start, length, _ = epoch
            awake_ripples.append(np.arange(start, start + length + 1, dtype=int))
        elif len(epoch) == 6:
            parts = []
            for s, e in zip(epoch[0::2], epoch[1::2]):
                parts.append(np.arange(s, e + 1, dtype=int))
            awake_ripples.append(np.concatenate(parts))

    # Retrieve inference results of awake ripples
    data_folders = [
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear2'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear3'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear4'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear1'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear2'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat3/Linear3'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear3'),
        os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear4')]
    result_folders = [
        'results0505/rat1_linear1',
        'results0505/rat1_linear2',
        'results0505/rat1_linear3',
        'results0505/rat1_linear4',
        'results0505/rat2_linear1',
        'results0505/rat2_linear2',
        'results0505/rat3_linear3',
        'results0505/rat5_linear3',
        'results0505/rat5_linear4']
    realtime_files = [
        'submission/Figure10/ripple_realtime/ripple_realtime_r1l1.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r1l2.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r1l3.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r1l4.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r2l1.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r2l2.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r3l3.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r5l3.npy',
        'submission/Figure10/ripple_realtime/ripple_realtime_r5l4.npy']

    replay_pos, real_pos = [], []
    dd_ripples, st_ripples, fr_ripples = [], [], []
    for awaker, df, rf, rtf in zip(awake_ripples, data_folders, result_folders, realtime_files):
        f1 = np.load(os.path.join(df, 'processed/processed_data.npz'), allow_pickle=True)
        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        f3 = np.load(rtf, allow_pickle=True)

        # Position tracking
        x_pos_intp = f1['x_pos_intp']
        time_bin_centers = f1['time_bin_centers']
        if isinstance(x_pos_intp, np.ndarray) and x_pos_intp.dtype == object:
            # multi-run
            timestamp = np.concatenate([np.asarray(tbc) for tbc in time_bin_centers])
            xpos = np.concatenate([np.asarray(xpi) for xpi in x_pos_intp])
        else:
            timestamp = time_bin_centers
            xpos = np.asarray(x_pos_intp)
        f1.close()

        # Inference
        optim_results_hdd = f2['optim_results']
        place_fields = f2['place_fields']
        ripple_spike_trains = f2['ripple_spike_trains']
        time_bin_size_replay = f2['time_window']
        f2.close()

        dd_ripples_this_sess, st_ripples_this_sess, fr_ripples_this_sess = [], [], []
        for ripple_i in tqdm(range(len(ripple_spike_trains))):
        # for ripple_i in tqdm(awaker):
            ######
            model_type = 'hybrid_drift_diffusion'
            # state_names = \
            #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
            state_names = \
                ['driftdiffusion1', 'fragmented', 'stationary']
            optim_results_vis = optim_results_hdd
            ######
            # model_type = 'drift_diffusion'
            # state_names = \
            #     ['driftdiffusion1', 'driftdiffusion2']
            # optim_results_vis = optim_results_dd
            ######
            # model_type = 'diffusion'
            # state_names = \
            #     ['diffusion1', 'diffusion2']
            # optim_results_vis = None # optim_results_d
            ######
            ripple_spike_train = ripple_spike_trains[ripple_i]
            optim_params = optim_results_vis[ripple_i, 1:5] # diag, lambda1, lambda2, sigma
            optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
                perform_inference(
                    optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
            state_marginal = np.sum(acausal_posterior, axis=2)
            position_marginal = np.sum(acausal_posterior, axis=1)

            # Dynamic duration and category
            threshold = 0.7
            p = state_marginal.squeeze()
            T = p.shape[0]
            labels = np.full(T, 'un', dtype=object)

            # 1) Pure states
            labels[p[:, 0] >= threshold] = 'dd' # 'drift_diffusion'
            labels[p[:, 1] >= threshold] = 'fr' # 'fragmented'
            labels[p[:, 2] >= threshold] = 'st' # 'stationary'

            # 2) Mixtures (only where still unclassified)
            unclass = (labels == 'un')
            sc_mix = (p[:, 2] + p[:, 0] >= threshold) & unclass
            fc_mix = (p[:, 1] + p[:, 0] >= threshold) & unclass
            labels[sc_mix] = 'st'
                # 'st_dd', 'stationary_driftdiffusion'
                # equivalent to stationary
            labels[fc_mix] ='un'
                # 'fr_dd', 'fragmented_driftdiffusion'
                # equivalent to unclassified

            # Consider replay position of stationary timesteps
            stat = (labels == 'st')
            posterior = position_marginal.squeeze()
            position_estimate = np.argmax(posterior, axis=1)
            replay_pos.append(position_estimate[stat])
            
            # Consider real position of stationary timesteps
            realtime = f3[ripple_i][stat]
            real_pos.append(np.interp(realtime, timestamp, xpos))

            # Classify ripples
            drdi = (labels == 'dd')
            stat = (labels == 'st')
            frag = (labels == 'fr')
            if np.sum(drdi) >= 5:
                dd_ripples_this_sess.append(ripple_i)
                    # this is different from how I classified in analysis_Pfeiffer1D.py
                    # analysis_Pfeiffer1D.py was used for analysis
            # if np.sum(stat) >= 5:
            if np.sum(stat) == T:
                st_ripples_this_sess.append(ripple_i)
            # if np.sum(frag) >= 5:
            if np.sum(frag) == T:
                fr_ripples_this_sess.append(ripple_i)
        dd_ripples.append(np.array(dd_ripples_this_sess, dtype=int))
        st_ripples.append(np.array(st_ripples_this_sess, dtype=int))
        fr_ripples.append(np.array(fr_ripples_this_sess, dtype=int))
    
    replay_pos = np.concatenate(replay_pos)
    real_pos = np.concatenate(real_pos)
    np.save('./temp/replay_pos.npy', replay_pos)
    np.save('./temp/real_pos.npy', real_pos)

    np.savez_compressed(
        "./temp/ripples_class.npz",
        dd=np.array(dd_ripples, dtype=object),
        st=np.array(st_ripples, dtype=object),
        fr=np.array(fr_ripples, dtype=object))

if 0:
    replay_pos = np.load('submission/Figure10/replay_pos.npy')
    real_pos = np.load('submission/Figure10/real_pos.npy')

    # Unit conversion
    replay_pos = replay_pos * 0.02 + 0.01 # bin -> m
    real_pos = real_pos/100 # cm -> m
    diff_pos = np.abs(replay_pos - real_pos)

    print(np.mean(diff_pos <= 0.2))
    raise EOFError

    n_bins = 20
    bins_1 = np.linspace(0, np.max(real_pos), n_bins + 1)
    bins_2 = np.linspace(0, np.max(replay_pos), n_bins + 1)
    bins_3 = np.linspace(0, np.max(diff_pos), n_bins + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
    axes[0].hist(
        real_pos, bins=bins_1,
        color='grey', edgecolor='white', linewidth=1.0, density=True)
    axes[1].hist(
        replay_pos, bins=bins_2,
        color='grey', edgecolor='white', linewidth=1.0, density=True)
    axes[2].hist(
        diff_pos, bins=bins_3,
        color='grey', edgecolor='white', linewidth=1.0, density=True)
    axes[2].set_yticks([0, 2, 4, 6, 8])
    for ax in axes:
        ax.set_xlim([0, 1.8])
        ax.set_xticks([0, 0.6, 1.2, 1.8])
    plt.savefig('./temp/stationary_position.svg')
    plt.close()

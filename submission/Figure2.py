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


###-----Combination of Dynamics-----###
if 0:
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
    num_preplays = [420, 776, 812, 575, 631, 734, 0, 1428, 1373]
    combo_dynamics_sess = []
    dynamics_dur_sess = []
    for rf, num_pre in zip(result_folders, num_preplays):
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        ripple_spike_trains = f1['ripple_spike_trains']
        place_fields = f1['place_fields']
        optim_results = f1['optim_results']
        time_bin_size_replay = f1['time_window']
        pos_bin_size = f1['position_window']
        f1.close()

        # Perform inference
        state_marginals = []
        for ripple_i in tqdm(range(len(ripple_spike_trains))):
            ######
            model_type = 'hybrid_drift_diffusion'
            # state_names = \
            #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
            state_names = \
                ['driftdiffusion1', 'fragmented', 'stationary']
            ######

            ripple_spike_train = ripple_spike_trains[ripple_i]
            optim_params = optim_results[ripple_i, 1:5] # diag, lambda1, lambda2, sigma
            optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
                perform_inference(
                    optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
            
            state_marginal = np.sum(acausal_posterior, axis=2).squeeze()
            position_marginal = np.sum(acausal_posterior, axis=1)
            state_marginals.append(state_marginal)

        # Dynamic duration and category
        threshold = 0.7
        dynamics_dur = np.zeros((len(state_marginals), 4), dtype=int)
        dynamic_categories = []
        for ii, p in enumerate(state_marginals):
            T = p.shape[0]
            # labels = np.full(T, 'unclassified', dtype=object)
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
            
            # 3) Count each category
            dynamics_dur[ii, 0] = np.count_nonzero(labels == 'dd')
            dynamics_dur[ii, 1] = np.count_nonzero(labels == 'fr')
            dynamics_dur[ii, 2] = np.count_nonzero(labels == 'st')
            dynamics_dur[ii, 3] = np.count_nonzero(labels == 'un')

            dynamic_categories.append(labels)
        dynamics_dur_sess.append(dynamics_dur)

        # Combination of dymnamic categories
        min_bins = 2 # time bin size: 10 ms
        combos = []
        for labels in dynamic_categories:
            event_set = set()
            # iterate through runs
            start = 0
            for idx in range(1, len(labels) + 1):
                # either end of array or label change
                if idx == len(labels) or labels[idx] != labels[start]:
                    run_label = labels[start]
                    run_length = idx - start
                    if run_length >= min_bins:
                        event_set.add(run_label)
                    start = idx
            combos.append(event_set)
        
        combo_dynamics_sess.append(combos)

    with open('./temp/combo_dynamics_sess.pkl', 'wb') as f:
        pickle.dump(combo_dynamics_sess, f)
    with open('./temp/dynamics_dur_sess.pkl', 'wb') as f:
        pickle.dump(dynamics_dur_sess, f)

if 0:
    REPLAY_MODE = True
    num_preplays = [420, 776, 812, 575, 631, 734, 0, 1428, 1373]
    ###
    with open('submission/Figure2/dynamics_dur_sess.pkl', 'rb') as f:
        dynamics_dur_sess = pickle.load(f)
    dynamics_durs = []
    for num_pre, dur in zip(num_preplays, dynamics_dur_sess):
        if REPLAY_MODE:
            dynamics_durs.append(dur[num_pre:, :])
        else:
            dynamics_durs.append(dur[:num_pre, :])
    dynamics_durs = np.concatenate(dynamics_durs, axis=0)
    ###
    with open('submission/Figure2/combo_dynamics_sess.pkl', 'rb') as f:
        combo_dynamics_sess = pickle.load(f)
    combos = []
    for num_pre, co in zip(num_preplays, combo_dynamics_sess):
        if REPLAY_MODE:
            combos.extend(co[num_pre:])
        else:
            combos.extend(co[:num_pre])
    ###

    # fraction of time for each dynamics
    dynamics_frac = np.sum(dynamics_durs, axis=0)/np.sum(dynamics_durs)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax.bar(np.arange(dynamics_frac.shape[0]), dynamics_frac, color='black', alpha=0.65)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['dd', 'fr', 'st', 'un'])
    ax.set_ylabel("% of time")
    plt.savefig('./temp/dynamic_duration.png', dpi=300)
    plt.savefig('./temp/dynamic_duration.svg')
    plt.close()

    # see analysis_Pfeiffer1D.py
    # count each unique combination
    combo_strings = ["+".join(sorted(ev)) if ev else "none" for ev in combos]
    n_events = len(combo_strings)
    counts = Counter(combo_strings)
    filtered = [(lbl, cnt) for lbl, cnt in counts.items() if cnt/n_events > 0.01]
    filtered.sort(key=lambda x: x[1], reverse=True)
    combo_labels, combo_counts = zip(*filtered)
    combo_percent = np.array(combo_counts) / n_events * 100

    print(combo_percent)
    raise EOFError

    # build a boolean presence matrix
    # cat_labels = [
    #     'St.',
    #     'DD.',
    #     'Uni-DD.',
    #     'Uni.',
    #     'Unc.']
    cat_labels = [
        'St.',
        'DD.',
        'Uni.',
        'Unc.']
    raw_to_cat = {
        'st':   'St.',
        'dd':   'DD.',
        'fr_dd':'Uni-DD.',
        'fr':   'Uni.',
        'un':   'Unc.'}
    presence = np.zeros((len(combo_labels), len(cat_labels)), dtype=bool)
    for i, combo in enumerate(combo_labels):
        raw_labels = combo.split('+')
        for raw in raw_labels:
            mapped = raw_to_cat.get(raw)
            if mapped is not None:
                j = cat_labels.index(mapped)
                presence[i, j] = True

    # marginal counts of each category
    marginal_counts = {cat: 0 for cat in cat_labels}
    for ev in combos:
        if ev:
            for raw in ev:
                cat = raw_to_cat.get(raw)
                marginal_counts[cat] += 1
    n_ripples = len(combos)
    total_marginal = sum(marginal_counts.values())
    # marginal_percent = np.array(
    #     [marginal_counts[cat] / total_marginal * 100 for cat in cat_labels])
    #     # if you pick a segment, what is the prob of each category
    marginal_percent  = np.array(
        [marginal_counts[cat] / n_ripples * 100 for cat in cat_labels])
        # if you pick a ripple, what is the prob of each category

    # Plotting
    x = np.arange(len(combo_percent))
    y = np.arange(len(cat_labels))

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 3], wspace=0.02, hspace=0.05)
    
    # 1. Left marginal histogram
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.barh(y, marginal_percent, color='black', alpha=0.65)
    ax_left.invert_yaxis()    # top category at top
    ax_left.invert_xaxis()    # bars extend leftward
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(cat_labels)
    ax_left.set_xlabel("Marginal % of Category")
    ax_left.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in ['top', 'right', 'left']:
        ax_left.spines[spine].set_visible(False)
    
    # 2. Dot + line matrix (no combo labels on x-axis)
    ax_matrix = fig.add_subplot(gs[1, 1], sharey=ax_left)
    for i in x:
        # grey dots
        ax_matrix.scatter(
            [i]*len(y), y,
            marker='o', facecolors='none', edgecolors='lightgrey', s=100, zorder=1)
        # black dots
        ys = np.where(presence[i])[0]
        ax_matrix.scatter([i]*len(ys), ys, marker='o', color='gray', s=100, zorder=3)
        # connect if >1
        if len(ys) > 1:
            ax_matrix.vlines(i, ys.min(), ys.max(), linewidth=3, color='gray', zorder=2)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels(cat_labels)
    ax_matrix.yaxis.tick_right() # move ticks to right
    ax_matrix.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=True)
    for spine in ['top', 'bottom']:
        ax_matrix.spines[spine].set_visible(False)
    
    # 3. Top histogram
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_matrix)
    ax_top.bar(x, combo_percent, color='black', alpha=0.65)
    ax_top.set_xticks([])
    ax_top.set_ylabel("% of Ripples")
    ax_top.set_ylim(0, combo_percent.max() * 1.1)
    
    plt.savefig('./temp/dynamic_category.png', dpi=300)
    plt.savefig('./temp/dynamic_category.svg')
    plt.close()

###-----Example Replay Events-----###
if 0:
    ######
    data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
    save_folder = './temp'
    result_folder = 'results0505/rat1_linear1'
    num_preplay, num_awake_replay, num_post_replay = 420, 232, 1852 # 544, 278, 2399
        # see train_Pfeiffer1D.py for how to count preplay events
    ######
    # data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear3')
    # save_folder = './temp'
    # result_folder = 'results0505/rat1_linear3'
    # num_preplay, num_awake_replay, num_post_replay = 812, 10, 10 # 812, 870, 1537, 1732, 2681, 2794
    ######

    ###
    # Example events in rat 1 linear 1
    # Drift-diffusion Uniform Mix
    #   2441
    # Drift-dffusion
    #   2217, 2397
    # Drift-diffusion + Stationary
    #   475, 2288, 2311
    # Drift-diffusion + Uniform
    #   2244, 2331
    # Stationary + Uniform
    #   2320, 2321
    # Stationary
    #   2317
    # Extremely long ripple
    #   2230, 2378
    ###
    # Example events in rat 1 linear 3
    # 3215 4.6 m/s; Drift-diffusion
    # 3608 9.7 m/s; Drift-diffusion + Stationary
    # 3650 1.7 m/s; Drift-diffusion + Stationary
    # 3767 2.3 m/s; Drift-diffusion + Uniform
    # 3780 1.1 m/s; Drift-diffusion
    # 3855 6.0 m/s; Drift-diffusion
    # 3885 6.2 m/s; Drift-diffusion
    # 3886 1.5 m/s; Drift-diffusion + Stationary
    # 3892 4.5 m/s; Drift-diffusion + Stationary
    ######

    loaded_data = np.load(os.path.join(result_folder, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
    optim_results_hdd = loaded_data['optim_results']
    place_fields = loaded_data['place_fields']
    ripple_spike_trains = loaded_data['ripple_spike_trains']
    time_bin_size_replay = loaded_data['time_window']
    pos_bin_size = loaded_data['position_window']
    loaded_data.close()

    r1l1 = [1958] # [1958, 2441, 2217, 2397, 475, 2288, 2311, 2244, 2331, 2320, 2321, 2317, 2230, 2378]
    # r1l3 = [3215, 3608, 3650, 3767, 3780, 3855, 3885, 3886, 3892]
    for ripple_i in r1l1:
    # for ripple_i in r1l3:
        # ripple_i = 2217

        ######
        model_type = 'hybrid_drift_diffusion'
        # state_names = \
        #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
        state_names = \
            ['driftdiffusion1', 'fragmented', 'stationary']
        optim_results_vis = optim_results_hdd
        ######
        # model_type = 'hybrid_diffusion'
        # state_names = \
        #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
        # optim_results_vis = optim_results_hd
        ######
        # model_type = 'drift_diffusion'
        # state_names = \
        #     ['driftdiffusion1', 'driftdiffusion2']
        # optim_results_vis = optim_results_dd
        ######
        # model_type = 'diffusion'
        # # state_names = \
        # #     ['diffusion1', 'diffusion2']
        # state_names = \
        #     ['driftdiffusion1', 'driftdiffusion2']
        # optim_results_vis = optim_results_d
        ######

        ripple_spike_train = ripple_spike_trains[ripple_i]
        optim_params = optim_results_vis[ripple_i, 1:5] # diag, lambda1, lambda2, sigma
        optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
            perform_inference(
                optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)

        n_time = ripple_spike_train.shape[0]
        n_pos_bins = place_fields.shape[0]
        spike_time_ind, neuron_ind = np.nonzero(ripple_spike_train)
        state_marginal = np.sum(acausal_posterior, axis=2)
        position_marginal = np.sum(acausal_posterior, axis=1)

        threshold = 0.7
        p = state_marginal.squeeze()
        T = p.shape[0]
        labels = np.full(T, 'unclass', dtype=object)
        # 1) Pure states
        labels[p[:, 0] >= threshold] = 'driftdiffusion1'
        labels[p[:, 1] >= threshold] = 'fragmented'
        labels[p[:, 2] >= threshold] = 'stationary'
        # 2) Mixtures (only where still unclassified)
        unclass = (labels == 'unclass')
        sc_mix = (p[:, 2] + p[:, 0] >= threshold) & unclass
        fc_mix = (p[:, 1] + p[:, 0] >= threshold) & unclass
        labels[sc_mix] = 'stationary' # equivalent to stationary
        labels[fc_mix] = 'fragmented_driftdiffusion'

        STATE_COLORS = {
            'driftdiffusion1': '#00CD9E',
            'driftdiffusion2': None,
            'fragmented': '#666666',
            'stationary': '#FFAB3F',
            'fragmented_driftdiffusion': 'white', # '#105500'
            'unclass': 'white'}
        STATE_LABEL = {
            'driftdiffusion1': 'DD.',
            'driftdiffusion2': None,
            'fragmented': 'Uni.',
            'stationary': 'St.'}
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True, sharex=True)
        ###
        # axes[0].scatter(
        #     (np.arange(n_time)[spike_time_ind]), neuron_ind,
        #     color='black', zorder=1, marker='|', s=160, linewidth=4)
        # axes[0].set_yticks((0, ripple_spike_train.shape[1]))
        # axes[0].set_ylabel('Neuron Index')
        ###
        T = p.shape[0]
        start, current = 0, labels[0]
        for t in range(1, T+1):
            # whenever the label changes (or we hit the end)
            if t == T or labels[t] != current:
                # span from `start` up to `t` in x-coordinates
                axes[0].axvspan(
                    start, t,
                    color=STATE_COLORS[current], alpha=0.3, linewidth=0,
                    zorder=0)
                # reset for the next run
                if t < T:
                    current = labels[t]
                start = t
        axes[0].plot(
            np.arange(n_time), state_marginal[:, 0, 0],
            label=STATE_LABEL['driftdiffusion1'], color=STATE_COLORS['driftdiffusion1'],
            linewidth=4, zorder=1)
        axes[0].plot(
            np.arange(n_time), state_marginal[:, 2, 0],
            label=STATE_LABEL['stationary'], color=STATE_COLORS['stationary'],
            linewidth=4, zorder=1)
        axes[0].plot(
            np.arange(n_time), state_marginal[:, 1, 0],
            label=STATE_LABEL['fragmented'], color=STATE_COLORS['fragmented'],
            linewidth=4, zorder=1)
        axes[0].set_ylim((-0.01, 1.01))
        axes[0].set_xticks([])
        axes[0].set_yticks([0, 1])
        ###
        vmin = 0.   # np.percentile(position_marginal, 2)
        vmax = 0.2  # np.percentile(position_marginal, 98)
        im = axes[1].imshow(
            position_marginal[:, :, 0].T,
            aspect='auto', cmap='hot', origin='lower',
            extent=[np.arange(n_time)[0], np.arange(n_time)[-1], 0, n_pos_bins],
            vmin=vmin,
            vmax=vmax)
        axes[1].set_xticks([0, n_time-1])
        axes[1].set_xticklabels(['0', str(n_time*10)+' ms'])
        axes[1].set_yticks([0, n_pos_bins])
        axes[1].set_yticklabels(['0', '180 cm'])
        cbar = plt.colorbar(im, ax=axes[1], label='')
        cbar.set_ticks([0., vmax])
        sns.despine(ax=axes[1], top=False, right=False)
        ###
        # plt.savefig(f'./temp/marginal_posterior_{ripple_i}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./temp/marginal_posterior_{ripple_i}.svg', bbox_inches='tight')
        plt.close()

# Classic decoding
if 0:
    ######
    # data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
    # save_folder = './temp'
    # result_folder = 'results0505/rat1_linear1'
    # num_preplay, num_awake_replay, num_post_replay = 420, 232, 1852 # 544, 278, 2399
        # see train_Pfeiffer1D.py for how to count preplay events
    ######
    data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear3')
    save_folder = './temp'
    result_folder = 'results0505/rat1_linear3'
    num_preplay, num_awake_replay, num_post_replay = 812, 10, 10 # 812, 870, 1537, 1732, 2681, 2794
    ######

    loaded_data = np.load(os.path.join(result_folder, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
    optim_results_hdd = loaded_data['optim_results']
    place_fields = loaded_data['place_fields']
    ripple_spike_trains = loaded_data['ripple_spike_trains']
    time_bin_size_replay = loaded_data['time_window']
    pos_bin_size = loaded_data['position_window']
    loaded_data.close()

    n_pos_bins = place_fields.shape[0]
    r1l1 = [1958, 2441, 2217, 2397, 475, 2288, 2311, 2244, 2331, 2320, 2321, 2317, 2230, 2378]
    r1l3 = [3215, 3608, 3650, 3767, 3780, 3855, 3885, 3886, 3892]
    # for ripple_i in r1l1:
    for ripple_i in r1l3:
        # ripple_i = 1976
        ripple_spike_train = ripple_spike_trains[ripple_i]
        n_time = ripple_spike_train.shape[0]
        spike_time_ind, neuron_ind = np.nonzero(ripple_spike_train)
        posterior, _ = bayesian_decoding(
            n=ripple_spike_train.T,
            f_x=place_fields.T,
            dx=pos_bin_size,
            dt=time_bin_size_replay,
            pos_bin_centers=np.arange(n_pos_bins),
            epsilon=1e-10)
    
        fig, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True, sharex=False)
        axes[0].scatter(
            (np.arange(n_time)[spike_time_ind]), neuron_ind,
            color='black', zorder=1, marker='|', s=160, linewidth=4)
        axes[0].set_xlim(-0.5, n_time-0.5)
        axes[0].set_xticks([0, n_time-1])
        axes[0].set_xticklabels(['0', str(n_time*10)+' ms'])
        axes[0].set_yticks((0, ripple_spike_train.shape[1]))
        vmin = 0.   # np.percentile(position_marginal, 2)
        vmax = 0.2  # np.percentile(position_marginal, 98)
        im = axes[1].imshow(
            posterior.T,
            aspect='auto', cmap='hot', origin='lower',
            extent=[np.arange(n_time)[0], np.arange(n_time)[-1], 0, n_pos_bins],
            vmin=vmin, vmax=vmax)
        axes[1].set_xticks([0, n_time-1])
        axes[1].set_xticklabels(['0', str(n_time*10)+' ms'])
        axes[1].set_yticks([0, n_pos_bins])
        axes[1].set_yticklabels(['0', '180 cm'])
        cbar = plt.colorbar(im, ax=axes[1], label='')
        cbar.set_ticks([0., vmax])
        sns.despine(ax=axes[1], top=False, right=False)
        plt.savefig(f'./temp/classic_posterior_{ripple_i}.svg', bbox_inches='tight')
        plt.close()

# Draw legends
if 0:
    line_colors = {
        'Drift–Diffusion':  '#9467BD',
        'Stationary':       '#FF7F0E',
        'Uniform':          '#7F7F7F'}
    shade_colors = {
        'Drift–Diffusion':  '#9467BD',
        'Stationary':       '#FF7F0E',
        'Uniform':          '#7F7F7F',
        'DD + Uniform':     '#C5B0D5'}

    line_handles = [
        Line2D([0], [0], color=col, linewidth=3)
        for col in line_colors.values()]
    line_labels = list(line_colors.keys())

    shade_handles = [
        Patch(facecolor=col, edgecolor='none', alpha=0.8)
        for col in shade_colors.values()]
    shade_labels = list(shade_colors.keys())

    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_axis_off()
    leg1 = ax.legend(
        handles = line_handles,
        labels  = line_labels,
        title   = "Dynamics",
        loc     = "upper left",
        frameon = False)
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles = shade_handles,
        labels  = shade_labels,
        title   = "Combinations",
        loc     = "upper right",
        frameon = False
    )
    plt.savefig(f'./temp/legends.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./temp/legends.pdf', bbox_inches='tight')
    plt.close()
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
from ssm.baseline import bayesian_decoding, bayesian_decoding_smooth, weighted_correlation, simple_linear_regression

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


###-----Illustation of Bayesian Decoding-----###
if 0:
    ###-----Retrieve Data-----###
    data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
    loaded_data = np.load(os.path.join(data_folder, 'processed/processed_data.npz'))
    pos_bin_centers = loaded_data['pos_bin_centers']
    loaded_data.close()

    result_folder = os.path.join(data_folder, 'processed/10msBin_10msAdvance/hmms_decode_2/drift_diffusion')
    loaded_data = np.load(os.path.join(result_folder, 'optim_results.npz'), allow_pickle=True)
    place_fields = loaded_data['place_fields']
    ripple_spike_trains = loaded_data['ripple_spike_trains']
    time_window = loaded_data['time_window']
    position_window = loaded_data['position_window']
    optim_results = loaded_data['optim_results']
    loaded_data.close()

    ripple_i = 475 # 2603
    post, x_hat = bayesian_decoding_smooth(
        n=ripple_spike_trains[ripple_i],
        f_x=place_fields,
        position_bin_size=position_window,
        tau=time_window,
        epsilon=1e-10,
        sigma_continuity=0.4)

    raster_vis = ripple_spike_trains[ripple_i][:, 16:].T
    post = post[16:, :]
    x_hat = x_hat[16:]


    n_time = raster_vis.shape[0]
    n_pos_bins = place_fields.shape[0]
    n_neuron = raster_vis.shape[1]
    spike_time_ind, neuron_ind = np.nonzero(raster_vis)

    fig, axes = plt.subplots(3, 1, figsize=(6, 18), constrained_layout=True, sharex=False)
    axes[0].scatter(
        (np.arange(n_time)[spike_time_ind]), neuron_ind,
        color='black', zorder=1, marker='|', s=220, linewidth=6)
    axes[0].set_yticks((0, n_neuron))
    axes[0].set_ylabel('Neuron Index')

    vmin = 0.   # np.percentile(position_marginal, 2)
    vmax = np.percentile(post, 98) # 0.2  # np.percentile(position_marginal, 98)
    im = axes[1].imshow(
        post.T,
        aspect='auto', cmap='hot', origin='lower',
        extent=[np.arange(n_time)[0], np.arange(n_time)[-1], 0, n_pos_bins],
        vmin=vmin,
        vmax=vmax)
    axes[1].set_xticks([0, n_time-1])
    axes[1].set_xticklabels(['0', str(n_time*10)+' ms'])
    axes[1].set_yticks([])
    # axes[1].set_yticklabels(['0', '180 cm'])
    # cbar = plt.colorbar(im, ax=axes[1], label='')
    # cbar.set_ticks([0., vmax])
    sns.despine(ax=axes[1], top=False, right=False)

    times_ms = np.arange(n_time) * 10
    axes[2].plot(times_ms, x_hat, '-o', color='k', linewidth=3, markersize=10, markerfacecolor='gray')
    axes[2].set_ylim(-0.05, 1.85)
    axes[2].set_xticks([])
    axes[2].set_yticks([0, 1.8])
    axes[2].set_yticklabels(['0', '180 cm'])

    plt.savefig(f'./temp/bayes_{ripple_i}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./temp/bayes_{ripple_i}.pdf', bbox_inches='tight')
    plt.close()

###-----Relationship to Replay Metrics-----###
if 0:
    result_folder = os.path.expanduser('~/Projects/RNN_navigation_2/results/recovery_analysis/fix_gamma')
        # './results/recovery_analysis/fix_gamma'
        # './results/recovery_analysis/optimize_gamma'
        # './results/recovery_analysis/fix_gamma'
    save_folder = './temp'

    ###-----Retrieve Results-----###
    loaded_data = np.load(os.path.join(result_folder, 'recovery_analysis.npz'), allow_pickle=True)
    place_fields = loaded_data['place_fields']
    lambda_gts = loaded_data['lambda_gts']
    sigma_gts = loaded_data['sigma_gts']
    sim_trajs = loaded_data['sim_trajs']
    sim_ripples = loaded_data['sim_ripples']
    time_window = loaded_data['time_window']
    position_window = loaded_data['position_window']
    optim_results = loaded_data['optim_results']
    loaded_data.close()

    if 0:
        decoded_corr = np.zeros(sim_ripples.shape[0:3])
        decoded_velocity = np.zeros(sim_ripples.shape[0:3])
        max_jump = np.zeros(sim_ripples.shape[0:3])
        for lam_i in tqdm((range(len(lambda_gts)))):
            for sig_i in range(len(sigma_gts)):
                for ripple_i in range(sim_ripples.shape[2]):
                    post, x_hat = bayesian_decoding(
                        n=sim_ripples[lam_i, sig_i, ripple_i],
                        f_x=place_fields,
                        dx=position_window,
                        dt=time_window,
                        pos_bin_centers=position_window * np.arange(place_fields.shape[1]),
                        epsilon=1e-10)
                    corr = weighted_correlation(
                        posterior=post,
                        position_bin_size=position_window,
                        tau=time_window)
                    velo = simple_linear_regression(
                        posterior=post,
                        position_bin_size=position_window,
                        tau=time_window)
                    jump = position_window * np.max(np.abs(
                        sim_trajs[lam_i, sig_i, ripple_i, 1:] - sim_trajs[lam_i, sig_i, ripple_i, :-1]))
                    decoded_corr[lam_i, sig_i, ripple_i] = corr
                    decoded_velocity[lam_i, sig_i, ripple_i] = velo
                    max_jump[lam_i, sig_i, ripple_i] = jump
        np.savez_compressed(
            os.path.join('./temp', 'comparison_bayesian.npz'),
            decoded_corr=decoded_corr,
            decoded_velocity=decoded_velocity,
            max_jump=max_jump)
    if 1:
        loaded_data = np.load('submission/Figure3/comparison_bayesian.npz', allow_pickle=True)
        decoded_corr = loaded_data['decoded_corr']
        decoded_velocity = loaded_data['decoded_velocity']
        max_jump = loaded_data['max_jump']
        loaded_data.close()

    abs_corr = np.abs(decoded_corr)
    speed = np.abs(decoded_velocity)
    data00 = [abs_corr[i, 0:2, :].ravel() for i in range(6)]
    data01 = [abs_corr[2:, i, :].ravel() for i in range(6)]
    data10 = [speed[i, 0:2, :].ravel() for i in range(6)]
    data11 = [speed[2:, i, :].ravel() for i in range(6)]
    data20 = [max_jump[i, 0:2, :].ravel() for i in range(6)]
    data21 = [max_jump[2:, i, :].ravel() for i in range(6)]


    sig_ticks = np.array([10, 50, 100, 200, 300, 400])
    boxprops = dict(linewidth=3.5, edgecolor='black', facecolor='#0083DF', alpha=1.0, zorder=3)
    whiskerprops = dict(linewidth=3.5, color='black', alpha=1.0, zorder=2)
    capprops = dict(linewidth=3.5, color='black', alpha=1.0, zorder=2)
    medianprops = dict(color='#FFA90D', linewidth=6.0, zorder=4)
    flierprops = dict(marker='o', markersize=5, linestyle='none', linewidth=1, alpha=1.0)
    data_vis = [
        [data00, data01],
        [data10, data11],
        [data20, data21]]

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for i in range(3):
        for j in range(2):
            ax = axes[i, j]
            if j == 0:
                bp = ax.boxplot(
                    data_vis[i][j], patch_artist=True,
                    boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, medianprops=medianprops, flierprops=flierprops)
            else:
                bp = ax.boxplot(
                    data_vis[i][j], positions=sig_ticks * position_window, patch_artist=True,
                    boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops, medianprops=medianprops, flierprops=flierprops)
            if j == 0:
                ax.set_xticks(np.arange(1, 7))
                ax.set_xticklabels([0, 2, 4, 6, 8, 10])
                ax.set_xlabel('$\lambda$ (m/s)')
            else:
                ax.set_xticks([0.2, 1.0, 2.0, 4.0, 6.0, 8.0])
                ax.set_xticklabels([0.2, 1.0, 2.0, 4.0, 6.0, 8.0])
                ax.set_xlabel('$\sigma$ (m/$\sqrt{\mathrm{s}}$ )')
            if i == 0:
                ax.set_ylabel('Absolute Correlation')
            elif i == 1:
                ax.set_ylabel('Estimated Speed (m/s)')
            else:
                ax.set_ylabel('Maximun Jump Distance (m)')
            # ax.set_ylim(0, None)
    axes[1, 1].set_ylim([None, 50])
    # plt.tight_layout()
    plt.savefig('./temp/comparison.png', dpi=300)
    plt.savefig('./temp/comparison.svg')
    plt.close()

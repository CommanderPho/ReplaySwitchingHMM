import os
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


###-----Running Position-----###
if 0:
    data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')

    # Position data
    # (Time, x-position (in cm), y-position (in cm), and head direction (in degrees))
    # only recorded for running periods
    position_data = loadmat(os.path.join(data_folder, 'Position_Data.mat'))['Position_Data']

    # Position tracking
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(position_data[:, 1], position_data[:, 2], c='k', s=1, alpha=0.15)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 40])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('x position (cm)')
    # ax.set_ylabel('y position (cm)')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./temp', 'position_tracking.png'), dpi=300)
    plt.close()

###-----Place Fields-----###
if 0:
    ######
    data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
    save_folder = './temp'
    result_folder = 'results0505/rat1_linear1'
    num_preplay, num_awake_replay, num_post_replay = 420, 232, 1852 # 544, 278, 2399
        # see train_Pfeiffer1D.py for how to count preplay events
    ######

    loaded_data = np.load(os.path.join(result_folder, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
    optim_results_hdd = loaded_data['optim_results']
    place_fields = loaded_data['place_fields']
    ripple_spike_trains = loaded_data['ripple_spike_trains']
    time_bin_size_replay = loaded_data['time_window']
    pos_bin_size = loaded_data['position_window']
    loaded_data.close()

    # Create subplots
    sampled_neurons = np.arange(0, place_fields.shape[1], 2)
    n_plots = sampled_neurons.size
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 6), sharex=False)
    for i, (ax, neuron) in enumerate(zip(axes, sampled_neurons)):
        x = np.arange(place_fields.shape[0])
        y = place_fields[:, neuron]
        ax.plot(x, y, color='black', linewidth=1.5)
        ax.fill_between(x, y, 0, color='black', alpha=0.3)
        ax.set_xlim(0, place_fields.shape[0] - 1)
        if i < n_plots - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, place_fields.shape[0]])
            ax.set_xticklabels(['0', '180 cm'])
        ax.set_yticks([])
    # plt.tight_layout()
    plt.savefig('./temp/individual_pfs.png', dpi=300)
    plt.savefig('./temp/individual_pfs.pdf')
    plt.close()
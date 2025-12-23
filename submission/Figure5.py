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
from scipy.interpolate import interp1d
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


###-----Behavior, movement and replay speed-----###
if 0:
    ### A. Calculate movement speed
    timestamp, xpos = [], []
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
    for df in data_folders:
        f1 = np.load(os.path.join(df, 'processed/processed_data.npz'), allow_pickle=True)
        x_pos_intp = f1['x_pos_intp']
        time_bin_centers = f1['time_bin_centers']
        # running speed
        if isinstance(x_pos_intp, np.ndarray) and x_pos_intp.dtype == object:
            # multi-run
            timestamp.extend([np.asarray(tbc) for tbc in time_bin_centers])
            xpos.extend([np.asarray(xpi) for xpi in x_pos_intp])
        else:
            timestamp.extend([np.asarray(time_bin_centers)])
            xpos.extend([np.asarray(x_pos_intp)])

    mov_speed = []
    for t, x in zip(timestamp, xpos):
        mov_sp = []
        # 1. Define midpoint and endpoints
        midpoint = (np.min(x) + np.max(x))/2
        endpoint_low, endpoint_high = midpoint - 25, midpoint + 25
        # 2. Detect midpoint crossings (upward and downward)
        up_cross = np.where((x[:-1] < midpoint) & (x[1:] >= midpoint))[0] + 1
        down_cross = np.where((x[:-1] > midpoint) & (x[1:] <= midpoint))[0] + 1
        crossings = [(i, 'up') for i in up_cross] + [(i, 'down') for i in down_cross]
        crossings.sort(key=lambda pair: pair[0])  # sort by index
        # 3. For each crossing, extend to endpoints and fit regression
        for idx, direction in crossings:
            if direction == 'up':
                # Start from lower endpoint, end at upper endpoint
                starts = np.where(x[:idx] <= endpoint_low)[0]
                ends   = np.where(x[idx:] >= endpoint_high)[0]
            else:
                # Start from upper endpoint, end at lower endpoint
                starts = np.where(x[:idx] >= endpoint_high)[0]
                ends   = np.where(x[idx:] <= endpoint_low)[0]
            # If no crossing at all...
            if starts.size == 0 or ends.size == 0:
                continue
            # Start and end of this lap
            start_idx, end_idx = starts[-1], idx + ends[0]
            # If too short...
            if end_idx - start_idx < 2:
                continue
            t_seg = t[start_idx:end_idx + 1]
            x_seg = x[start_idx:end_idx + 1]
            v, _ = np.polyfit(t_seg, x_seg, 1)
            mov_sp.append(v)
        mov_speed.extend(mov_sp)
    mov_speed = np.abs(np.array(mov_speed))
    
    # unit conversion
    mov_speed = mov_speed / 100 # cm/sec -> m/sec


    ### B. Retrieve behavior and replay speed
    beh_speed, replay_speed = [], []
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
    num_preplays = [420, 776, 812, 575, 631, 734, 0, 1428, 1373]
    for df, rf, num_pre in zip(data_folders, result_folders, num_preplays):
        f1 = np.load(os.path.join(df, 'processed/processed_data.npz'), allow_pickle=True)
        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        f3 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/dd_ripples.npy'))
        speed_intp = f1['speed_intp']
        optim_results_hdd = f2['optim_results']
        dd_ripple = f3
        f1.close(); f2.close()
        # behavior speed
        if isinstance(speed_intp, np.ndarray) and speed_intp.dtype == object:
            # multi-run
            run_epoch_speeds = [np.asarray(s) for s in speed_intp]
        else:
            run_epoch_speeds = [np.asarray(speed_intp)]
        beh_speed.extend(run_epoch_speeds)
        # replay speed
        dd_ripple = dd_ripple[dd_ripple >= num_pre] # remove preplay event
        replay_speed.append(np.abs(optim_results_hdd[dd_ripple, 2]))
    beh_speed = np.concatenate(beh_speed)
    replay_speed = np.concatenate(replay_speed)

    # unit conversion
    beh_speed = beh_speed / 100 # cm/sec -> m/sec
    replay_speed = replay_speed * 0.02 # bin/sec -> m/sec


    ### C. Visualization
    # empirical cdf
    # def calc_cdf(data):
    #     sorted_data = np.sort(data)
    #     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    #     cdf_function = interp1d(sorted_data, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
    #     x_val = np.linspace(min(data), max(data), 300)
    #     y_val = cdf_function(x_val)
    #     return x_val, y_val
    # mov_x_val, mov_y_val = calc_cdf(mov_speed)
    # beh_x_val, beh_y_val = calc_cdf(beh_speed)
    # replay_x_val, replay_y_val = calc_cdf(replay_speed)

    print(np.mean(mov_speed), np.mean(beh_speed), np.mean(replay_speed))
    raise EOFError

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    n_bins = 20
    bins_mov = np.linspace(0, 1.2, n_bins + 1)
    bins_beh = np.linspace(0, 1, n_bins + 1)
    bins_replay = np.linspace(0, 15, n_bins + 1)
    ###
    axes[0].hist(mov_speed, bins=bins_mov, color='grey', edgecolor='white', linewidth=1.5, density=True)
    axes[0].axvline(np.mean(mov_speed), color='tab:red', linestyle='--', linewidth=3)
    axes[0].set_xlim([0, 1.2])
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    ###
    axes[1].hist(beh_speed, bins=bins_beh, color='grey', edgecolor='white', linewidth=1.5, density=True)
    axes[1].axvline(np.mean(beh_speed), color='tab:red', linestyle='--', linewidth=3)
    axes[1].set_xlim([0, 1])
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ###
    axes[2].hist(replay_speed, bins=bins_replay, color='grey', edgecolor='white', linewidth=1.5, density=True)
    axes[2].axvline(np.mean(replay_speed), color='tab:red', linestyle='--', linewidth=3)
    axes[2].set_xlim([0, 15])
    axes[2].set_xticks([0, 5, 10, 15])
    ###
    # plt.tight_layout()
    plt.savefig('./temp/speed_dist.png', dpi=300)
    plt.savefig('./temp/speed_dist.svg')
    plt.close()

###-----Drift and diffusion params-----###
if 0:
    lamb, sigma = [], []
    result_folders = [
        'results0505/rat1_linear1',
        'results0505/rat1_linear2',
        'results0505/rat1_linear3',
        'results0505/rat2_linear1']
    num_preplays = [420, 776, 812, 631]
    for rf, num_pre in zip(result_folders, num_preplays):
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/dd_ripples.npy'))
        optim_results_hdd = f1['optim_results']
        dd_ripple = f2
        f1.close()
        dd_ripple = dd_ripple[dd_ripple >= num_pre] # remove preplay event
        lamb.append(np.abs(optim_results_hdd[dd_ripple, 2]))
        sigma.append(optim_results_hdd[dd_ripple, 4])
    lamb = np.concatenate(lamb)
    sigma = np.concatenate(sigma)

    # unit conversion
    lamb = lamb * 0.02 # bin/sec -> m/sec
    sigma = sigma * 0.02 # bin/sqrt(sec) -> m/sqrt(sec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=False)
    ax.scatter(lamb, sigma, s=10, c='black', alpha=0.2, edgecolors='white', linewidth=0.5)
    plt.savefig('./temp/lambda_sigma.png', dpi=300)
    plt.savefig('./temp/lambda_sigma.pdf')
    plt.close()
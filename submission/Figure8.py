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
from matplotlib.ticker import ScalarFormatter
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


###-----Calculate Likelihood for Denovellis 2021-----###
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

    denovellis2021 = []
    for rf, num_pre in zip(result_folders, num_preplays):
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        ripple_spike_trains = f1['ripple_spike_trains']
        place_fields = f1['place_fields']
        time_bin_size_replay = f1['time_window']
        pos_bin_size = f1['position_window']
        f1.close()
        ripples = ripple_spike_trains[num_pre:]
        for thisripple in tqdm(ripples):
            model_type = 'hybrid_drift_diffusion'
            # state_names = \
            #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
            state_names = \
                ['driftdiffusion1', 'fragmented', 'stationary']
            optim_params = np.array([
                0.98,           # diag; fixed
                0.,             # lambda1; no drift
                0.,             # lambda2; not used
                12.24745])      # sigma; in (spatial bin)/sqrt(sec); (sigma**2) * (time bin) = 6 centimeter**2
            optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
                perform_inference(
                    optim_params, place_fields, thisripple, time_bin_size_replay, model_type)
            denovellis2021.append(data_log_likelihood)
    denovellis2021 = np.array(denovellis2021)
    np.save('./temp/denovellis2021.npy', denovellis2021)

###-----Likelihood Comparison-----###
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

    hdd, hd, dd = [], [], []
    for rf, num_pre in zip(result_folders, num_preplays):
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_diffusion/optim_results.npz'), allow_pickle=True)
        f3 = np.load(os.path.join(rf, 'dynamics_models/drift_diffusion/optim_results.npz'), allow_pickle=True)
        hdd.append(f1['optim_results'][num_pre:, 0])
        hd.append(f2['optim_results'][num_pre:, 0])
        dd.append(f3['optim_results'][num_pre:, 0])
        f1.close(); f2.close(); f3.close()
    # log-likelihood
    hdd = np.concatenate(hdd)
    hd = np.concatenate(hd)
    dd = np.concatenate(dd)
    denovellis2021 = np.load('submission/Figure8/denovellis2021.npy')
    # ΔAIC
    diff_0 = (2*2 - 2*dd) - (2*3 - 2*hdd)
    diff_1 = (2*2 - 2*hd) - (2*3 - 2*hdd)
    diff_2 = (0*2 - 2*denovellis2021) - (2*3 - 2*hdd)

    n_bins = 40
    bins_0 = np.linspace(np.min(diff_0), 50, n_bins + 1)
    bins_1 = np.linspace(np.min(diff_1), 10, n_bins + 1)
    bins_2 = np.linspace(np.min(diff_1), 24, n_bins + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)
    axes[0].hist(
        diff_0, bins=bins_0,
        color='#6E98EC', edgecolor='white', linewidth=1.0, density=False)
    # axes[0].set_ylim(0, 4000)
    axes[0].set_yticks([0, 1000, 2000, 3000, 4000])
    axes[1].hist(
        diff_1, bins=bins_1,
        color='#6E98EC', edgecolor='white', linewidth=1.0, density=False)
    # axes[1].set_ylim(0, 4000)
    axes[1].set_yticks([0, 1000, 2000, 3000, 4000])
    axes[2].hist(
        diff_2, bins=bins_2,
        color='#6E98EC', edgecolor='white', linewidth=1.0, density=False)
    # axes[2].set_ylim(0, 3000)
    axes[2].set_yticks([0, 1000, 2000, 3000])
    for ax in axes.ravel():
        ax.axvline(0, color='tab:red', alpha=0.5, linestyle='--', linewidth=3)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        ax.set_xlabel(r'$\Delta$Log-likelihood')
        ax.set_ylabel('# Ripples')
    plt.savefig('./temp/likelihood_comparison.svg')
    plt.close()
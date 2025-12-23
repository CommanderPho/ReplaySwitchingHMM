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
from scipy.stats import norm, levy_stable, gaussian_kde, linregress
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from ssm.ssm import SSM
from ssm.continuous_transition import make_driftdiffusion_transition
from ssm.optimization import perform_inference
from ssm.baseline import bayesian_decoding, bayesian_decoding_smooth, weighted_correlation, simple_linear_regression
from ssm.utils import (
    ic_uniform,
    dt_diagonal,
    ct_fragmented, ct_stationary, ct_driftdiffusion)

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


###-----λ and σ Recovery-----###
if 0:
    # loaded_data = np.load('results0505/recovery/recovery_analysis/lambsig_recovery_results/lambsig_recovery_results.npz', allow_pickle=True)
    loaded_data = np.load('results0505/recovery2/recovery_analysis/optim_results.npz', allow_pickle=True)
    
    optim_results = loaded_data['optim_results']
    ground_truth = loaded_data['gt_settings']
    pos_bin_size = 0.02 # meter
    loaded_data.close()

    lambda_gt, sigma_gt = ground_truth[:, 1], ground_truth[:, 2]
    lambda_fit, sigma_fit = optim_results[:, 2], optim_results[:, 4]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    # 1. λ‐recovery, for two fixed σ values
    for ax, sig_val in zip(axes[0], [10, 20]):
        mask = (sigma_gt == sig_val)
        x = lambda_gt[mask]
        y = lambda_fit[mask]
        x_unique = np.unique(x)
        mean_y = np.array([y[x == xv].mean() for xv in x_unique])
        std_y  = np.array([y[x == xv].std()  for xv in x_unique])
        low = -65 # x_unique.min()
            # min(x_unique.min(), (mean_y - std_y).min())
        high = 480 # x_unique.max()
            # max(x_unique.max(), (mean_y + std_y).max())
        ax.plot(x_unique, mean_y, marker='o', color='xkcd:teal', linewidth=2.5)
        ax.fill_between(
            x_unique,
            mean_y - std_y,
            mean_y + std_y,
            color='xkcd:teal',
            alpha=0.2)
        ax.plot([low, high], [low, high], c='grey', ls='--', linewidth=1.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xticklabels([0, 2, 4, 6, 8])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_yticklabels([0, 2, 4, 6, 8])
        ax.set_xlabel('Ground‐truth λ')
        ax.set_ylabel('Fitted λ')
        # ax.set_title(f'λ recovery (σ = {sig_val})')
    # 2. σ‐recovery, for two fixed λ values
    for ax, lam_val in zip(axes[1], [0, 100]):
        mask = (lambda_gt == lam_val)
        x = sigma_gt[mask]
        y = sigma_fit[mask]
        x_unique = np.unique(x)
        mean_y = np.array([y[x == xv].mean() for xv in x_unique])
        std_y  = np.array([y[x == xv].std()  for xv in x_unique])
        low = 3
            # x_unique.min()
            # min(x_unique.min(), (mean_y - std_y).min())
        high = 27
            # x_unique.max()
            # max(x_unique.max(), (mean_y + std_y).max())
        ax.plot(x_unique, mean_y, marker='o', color='xkcd:teal', linewidth=2.5)
        ax.fill_between(
            x_unique,
            mean_y - std_y,
            mean_y + std_y,
            color='xkcd:teal',
            alpha=0.2)
        ax.plot([low, high], [low, high], c='grey', ls='--', linewidth=1.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high+6)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticks([5, 10, 15, 20, 25])
        ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xlabel('Ground‐truth σ')
        ax.set_ylabel('Fitted σ')
        # ax.set_title(f'σ recovery (λ = {lam_val})')
    for ax in axes.ravel():
        ax.set_box_aspect(1)
    plt.savefig('./temp/lamsig_recovery.svg')
    plt.close()

###-----Full Model λ and σ Recovery-----###
if 0:
    # loaded_data = np.load('results0505/recovery2/full_recovery_analysis/100repeats/optim_results.npz', allow_pickle=True)
    # loaded_data = np.load('results0505/recovery/full_recovery_analysis/lambsig_30repeats/lambsig_30repeats.npz', allow_pickle=True)
    loaded_data = np.load('results0505/recovery2/full_recovery_analysis2/100repeats_2024seed/optim_results.npz', allow_pickle=True)
    optim_results = loaded_data['optim_results']
    ground_truth = loaded_data['gt_settings']
    pos_bin_size = 0.02 # meter
    loaded_data.close()

    I1_gt, I2_gt = ground_truth[:, 4], ground_truth[:, 5]
    valid_inds = (I1_gt == 0) | (I2_gt == 0)
        # must include drift-diffusion dynamics
    lambda_gt, sigma_gt = \
        ground_truth[valid_inds, 1], ground_truth[valid_inds, 2]
    lambda_fit, sigma_fit = \
        optim_results[valid_inds, 2], optim_results[valid_inds, 4]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    # 1. λ‐recovery, for two fixed σ values
    for ax, sig_val in zip(axes[0], [10, 20]):
        mask = (sigma_gt == sig_val)
        x = lambda_gt[mask]
        y = lambda_fit[mask]
        x_unique = np.unique(x)
        mean_y = np.array([y[x == xv].mean() for xv in x_unique])
        std_y  = np.array([y[x == xv].std()  for xv in x_unique])
        low = -65 # x_unique.min()
            # min(x_unique.min(), (mean_y - std_y).min())
        high = 480 # x_unique.max()
            # max(x_unique.max(), (mean_y + std_y).max())
        ax.plot(x_unique, mean_y, marker='o', color='xkcd:teal', linewidth=2.5)
        ax.fill_between(
            x_unique,
            mean_y - std_y,
            mean_y + std_y,
            color='xkcd:teal',
            alpha=0.2)
        ax.plot([low, high], [low, high], c='grey', ls='--', linewidth=1.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xticklabels([0, 2, 4, 6, 8])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_yticklabels([0, 2, 4, 6, 8])
        ax.set_xlabel('Ground‐truth λ')
        ax.set_ylabel('Fitted λ')
        # ax.set_title(f'λ recovery (σ = {sig_val})')
    # 2. σ‐recovery, for two fixed λ values
    for ax, lam_val in zip(axes[1], [0, 100]):
        mask = (lambda_gt == lam_val)
        x = sigma_gt[mask]
        y = sigma_fit[mask]
        x_unique = np.unique(x)
        mean_y = np.array([y[x == xv].mean() for xv in x_unique])
        std_y  = np.array([y[x == xv].std()  for xv in x_unique])
        low = 3
            # x_unique.min()
            # min(x_unique.min(), (mean_y - std_y).min())
        high = 27
            # x_unique.max()
            # max(x_unique.max(), (mean_y + std_y).max())
        ax.plot(x_unique, mean_y, marker='o', color='xkcd:teal', linewidth=2.5)
        ax.fill_between(
            x_unique,
            mean_y - std_y,
            mean_y + std_y,
            color='xkcd:teal',
            alpha=0.2)
        ax.plot([low, high], [low, high], c='grey', ls='--', linewidth=1.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high+6)
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticks([5, 10, 15, 20, 25])
        ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xlabel('Ground‐truth σ')
        ax.set_ylabel('Fitted σ')
        # ax.set_title(f'σ recovery (λ = {lam_val})')
    for ax in axes.ravel():
        ax.set_box_aspect(1)
    plt.savefig('./temp/full_recovery.svg')
    plt.close()

###-----Discrete State Recovery-----###
if 0:
    # loaded_data = np.load('submission/Figure4/state_infer1.npz', allow_pickle=True)
    # true_discrete_states = loaded_data['true_discrete_states']
    # infer_discrete_states = loaded_data['infer_discrete_states']
    # loaded_data.close()

    ###
    # loaded_data = np.load('results0505/recovery2/full_recovery_analysis/100repeats/optim_results.npz', allow_pickle=True)
    # loaded_data = np.load('results0505/recovery/full_recovery_analysis/lambsig_30repeats/lambsig_30repeats.npz', allow_pickle=True)
    loaded_data = np.load('results0505/recovery2/full_recovery_analysis2/100repeats_2024seed/optim_results.npz', allow_pickle=True)
    ground_truth = loaded_data['gt_settings']
    pos_bin_size = 0.02 # meter
    loaded_data.close()
    true_lambda, true_sigma, true_I1, true_I2 = \
        ground_truth[:, 1], ground_truth[:, 2], ground_truth[:, 3].astype(int), ground_truth[:, 4].astype(int)
    ###
    # data = np.load('results0505/recovery2/full_recovery_analysis/100repeats/state_infer.npz')
    # data = np.load('results0505/recovery/full_recovery_analysis/lambsig_30repeats/state_infer.npz')
    data = np.load('results0505/recovery2/full_recovery_analysis2/100repeats_2024seed/state_infer.npz')
    true_discrete_states = data["true_discrete_states"]
    state_marginals = data["state_marginals"]
    ###

    # mask1: only uniform and stationary dynamics
    # mask2: drift-diffusion involved; lambda = 0 m/s
    # mask3: drift-diffusion involved; lambda = 2 m/s
    # mask4: drift-diffusion involved; lambda = 4 m/s
    # mask4: drift-diffusion involved; lambda >= 6 m/s
    mask1 = ((true_I1 == 1) & (true_I2 == 2)) | ((true_I1 == 2) & (true_I2 == 1))
    mask2 = ((true_I1 == 0) | (true_I2 == 0)) & np.isclose(true_lambda, 0)
    mask3 = ((true_I1 == 0) | (true_I2 == 0)) & np.isclose(true_lambda, 100)
    mask4 = ((true_I1 == 0) | (true_I2 == 0)) & np.isclose(true_lambda, 200)
    mask5 = ((true_I1 == 0) | (true_I2 == 0)) & (true_lambda >= 300)

    def plot_confusion_matrix(true_discrete_states, state_marginals, mask, fname, NO_DD=False):
        true_discrete_states = true_discrete_states[mask, :]
        state_marginals = state_marginals[mask, :, :]
        true_discrete_states = true_discrete_states.reshape(-1)
        state_marginals = state_marginals.reshape(-1, state_marginals.shape[-1])
        print(true_discrete_states.shape, state_marginals.shape)

        p = state_marginals
        ######
        # infer_discrete_states = np.argmax(p, axis=1)
        ######
        # Dynamic category
        T = p.shape[0]
        threshold = 0.7
        dynamic_categories = []
        labels = np.full(T, 3, dtype=int) # unclassified
        # 1) Pure states
        labels[p[:, 0] >= threshold] = 0 # 'drift_diffusion'
        labels[p[:, 1] >= threshold] = 1 # 'fragmented'
        labels[p[:, 2] >= threshold] = 2 # 'stationary'
        # 2) Mixtures (only where still unclassified)
        unclass = (labels == 3)
        sc_mix = (p[:, 2] + p[:, 0] >= threshold) & unclass
        fc_mix = (p[:, 1] + p[:, 0] >= threshold) & unclass
        labels[sc_mix] = 2 # 'st_dd' # 'stationary_driftdiffusion'; equivalent to stationary
        labels[fc_mix] = 3 # 'fragmented_driftdiffusion'; seen as unclassified
        infer_discrete_states = labels
        ######

        if NO_DD:
            conf_matrix = confusion_matrix(true_discrete_states, infer_discrete_states, labels=[1,2,3])
            conf_matrix = conf_matrix[:2, :]
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix = np.divide(conf_matrix, row_sums, where=row_sums!=0)

            conf_matrix = conf_matrix[[1, 0], :][:, [1, 0, 2]]
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap='magma', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(
                        j, i, f"{conf_matrix[i, j]:.2f}", fontsize=25,
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] < thresh else "black")
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(['Stationary', 'Uniform', 'Unclass'])
            ax.set_yticks(np.arange(2))
            ax.set_yticklabels(['Stationary', 'Uniform'])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            plt.savefig('./temp/' + fname + '.svg')
            plt.close()
        else:
            conf_matrix = confusion_matrix(true_discrete_states, infer_discrete_states, labels=[0,1,2,3])
            conf_matrix = conf_matrix[:3, :]
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix = np.divide(conf_matrix, row_sums, where=row_sums!=0)

            conf_matrix = conf_matrix[[0, 2, 1], :][:, [0, 2, 1, 3]]
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap='magma', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(
                        j, i, f"{conf_matrix[i, j]:.2f}", fontsize=25,
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] < thresh else "black")
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels(['Drift-Diffusion', 'Stationary', 'Uniform', 'Unclass'])
            ax.set_yticks(np.arange(3))
            ax.set_yticklabels(['Drift-Diffusion', 'Stationary', 'Uniform'])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            plt.savefig('./temp/' + fname + '.svg')
            plt.close()
    
    plot_confusion_matrix(true_discrete_states, state_marginals, mask=mask1, fname='confmat_nodd', NO_DD=True)
    plot_confusion_matrix(true_discrete_states, state_marginals, mask=mask2, fname='confmat_lamb0', NO_DD=False)
    plot_confusion_matrix(true_discrete_states, state_marginals, mask=mask3, fname='confmat_lamb2', NO_DD=False)
    plot_confusion_matrix(true_discrete_states, state_marginals, mask=mask4, fname='confmat_lamb4', NO_DD=False)
    plot_confusion_matrix(true_discrete_states, state_marginals, mask=mask5, fname='confmat_lamb6plus', NO_DD=False)
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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, levy_stable, gaussian_kde, linregress, ks_2samp, mannwhitneyu
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def pval_to_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''

def calc_cdf(data, x_val):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_function = interp1d(sorted_data, cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
    y_val = cdf_function(x_val)
    return y_val

def calc_kde_cdf(data, x_val, bandwidth=0.1):
    kde = gaussian_kde(data, bw_method=bandwidth)
    pdf = kde(x_val)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]  # normalize to 1
    return cdf

###-----Drift-Diffusion Length-----###
if 0:
    n_repeats = 120
    n_sess = 9
    num_preplay_sess = [420, 776, 812, 575, 631, 734, 0, 1428, 1373]
    path_timebin_sess = [
        'results2/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear2/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear3/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear4/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear1/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear2/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat3_linear3/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear3/shuffle_timebin/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear4/shuffle_timebin/drift-diffusion_time_120repeats.npz']
    path_neuron_sess = [
        'results2/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear2/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear3/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear4/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear1/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear2/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat3_linear3/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear3/shuffle_neuron/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear4/shuffle_neuron/drift-diffusion_time_120repeats.npz']
    path_pf_sess = [
        'results0505/rat1_linear1/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear2/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear3/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat1_linear4/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear1/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat2_linear2/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat3_linear3/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear3/shuffle_placefields/drift-diffusion_time_120repeats.npz',
        'results0505/rat5_linear4/shuffle_placefields/drift-diffusion_time_120repeats.npz']
    
    # A. Retrieve drift-diffusion length
    def get_times(file_path, num_preplay, num_preplay_shuffled):
        loaded_data = np.load(file_path, allow_pickle=True)
        ### due to different saving formats
        ### shuffle timebin or shuffle cell
        if len(loaded_data['dd_times_s'].shape) == 1:
            times_o = loaded_data['dd_times_o']
            times_s = loaded_data['dd_times_s']
            times_o_pre = times_o[:num_preplay]
            times_o_post = times_o[num_preplay:]
            times_s_pre = times_s[:num_preplay_shuffled]
            times_s_post = times_s[num_preplay_shuffled:]
        ### shuffle place fields
        elif len(loaded_data['dd_times_s'].shape) == 2:
            times_o = loaded_data['dd_times_o'].ravel()
            times_s = loaded_data['dd_times_s'].ravel()
            times_o_pre = loaded_data['dd_times_o'][0, :num_preplay]
            times_o_post = loaded_data['dd_times_o'][0, num_preplay:]
            times_s_pre = loaded_data['dd_times_s'][:, :num_preplay].ravel()
            times_s_post = loaded_data['dd_times_s'][:, num_preplay:].ravel()
        ###
        loaded_data.close()
        return times_o, times_s, times_o_pre, times_o_post, times_s_pre, times_s_post

    ddtimes_o_pre_sess, ddtimes_o_post_sess, \
        ddtimes_s_pre_timebin, ddtimes_s_post_timebin, \
        ddtimes_s_pre_neuron, ddtimes_s_post_neuron, \
        ddtimes_s_pre_pf, ddtimes_s_post_pf = \
        [], [], [], [], [], [], [], []
    for i in range(n_sess):
        np_sess = num_preplay_sess[i]
        np_sess_shuf = np_sess * n_repeats
        _, _, t1, t2, t3, t4 = get_times(path_timebin_sess[i], num_preplay=np_sess, num_preplay_shuffled=np_sess_shuf)
        _, _, _, _, t5, t6 = get_times(path_neuron_sess[i], num_preplay=np_sess, num_preplay_shuffled=np_sess_shuf)
        _, _, _, _, t7, t8 = get_times(path_pf_sess[i], num_preplay=np_sess, num_preplay_shuffled=np_sess_shuf)
        ddtimes_o_pre_sess.append(t1)
        ddtimes_o_post_sess.append(t2)
        ddtimes_s_pre_timebin.append(t3)
        ddtimes_s_post_timebin.append(t4)
        ddtimes_s_pre_neuron.append(t5)
        ddtimes_s_post_neuron.append(t6)
        ddtimes_s_pre_pf.append(t7)
        ddtimes_s_post_pf.append(t8)
    ddtimes_o_pre_sess = np.concatenate(ddtimes_o_pre_sess)
    ddtimes_o_post_sess = np.concatenate(ddtimes_o_post_sess)
    ddtimes_s_pre_timebin = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_pre_timebin], axis=1)
    ddtimes_s_post_timebin = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_post_timebin], axis=1)
    ddtimes_s_pre_neuron = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_pre_neuron], axis=1)
    ddtimes_s_post_neuron = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_post_neuron], axis=1)
    ddtimes_s_pre_pf = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_pre_pf], axis=1)
    ddtimes_s_post_pf = np.concatenate(
        [t.reshape(n_repeats, -1) for t in ddtimes_s_post_pf], axis=1)

    # B. Calculate proportion
    thresholds = np.arange(0, 16) # in time bin
    n_pre = ddtimes_o_pre_sess.shape[0]
    n_post = ddtimes_o_post_sess.shape[0]
    # 1. compute real proportion above each threshold
    #    real_props[i] = proportion of times_o > thresholds[i]
    real_pre = np.array([
        (ddtimes_o_pre_sess > thr).sum() / n_pre for thr in thresholds])
    real_post = np.array([
        (ddtimes_o_post_sess > thr).sum() / n_post for thr in thresholds])
    # 2. compute shuffle proportions above each threshold
    #    shuffle_props[i, j] = for threshold i, proportion in shuffle j
    shuffle_pre_timebin = np.array([
        (ddtimes_s_pre_timebin[:, :] > thr).sum(axis=1) / n_pre for thr in thresholds])
    shuffle_post_timebin = np.array([
        (ddtimes_s_post_timebin[:, :] > thr).sum(axis=1) / n_post for thr in thresholds])
    shuffle_pre_neuron = np.array([
        (ddtimes_s_pre_neuron[:, :] > thr).sum(axis=1) / n_pre for thr in thresholds])
    shuffle_post_neuron = np.array([
        (ddtimes_s_post_neuron[:, :] > thr).sum(axis=1) / n_post for thr in thresholds])
    shuffle_pre_pf = np.array([
        (ddtimes_s_pre_pf[:, :] > thr).sum(axis=1) / n_pre for thr in thresholds])
    shuffle_post_pf = np.array([
        (ddtimes_s_post_pf[:, :] > thr).sum(axis=1) / n_post for thr in thresholds])

    # C. Visualization: data VS shuffled
    thresholds = np.arange(0, 16)
    linecolor_vis = ['#56B4E9', '#D55E00']
    real_vis = [real_pre, real_post]
    shuffle_vis = [
        [shuffle_pre_timebin, shuffle_pre_neuron, shuffle_pre_pf],
        [shuffle_post_timebin, shuffle_post_neuron, shuffle_post_pf]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False, constrained_layout=True)
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(
                thresholds, real_vis[i], '-o',
                color=linecolor_vis[i], linewidth=2.5,
                label='Real data', zorder=3)
            parts = ax.violinplot(
                dataset=[shuffle_vis[i][j][t] for t in range(len(thresholds))],
                positions=thresholds,
                widths=(thresholds[1] - thresholds[0]) * 0.5,
                showextrema=False,
                showmeans=False)
            for pc in parts['bodies']:
                pc.set_facecolor('gray')
                pc.set_edgecolor('black')
                pc.set_alpha(0.8)
            for t, thr in enumerate(thresholds):
                # one‐sided p-value (shuffle ≥ real)
                pval = (shuffle_vis[i][j][t] >= real_vis[i][t]).mean()
                stars = pval_to_stars(pval)
                if stars:
                    ax.text(
                        thr, real_vis[i][t] + 0.03*real_vis[i].max(), stars,
                        ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')
            ax.set_xticks([0, 5, 10, 15])
            ax.set_xticklabels([0, 50, 100, 150])
            ax.set_ylim([0, None])
            if i == 0:
                ax.set_ylim([0, 0.2])
                ax.set_yticks([0, 0.10, 0.20])
                ax.set_yticklabels([0, 0.10, 0.20])
            else:
                ax.set_ylim([0, 0.3])
                ax.set_yticks([0, 0.15, 0.30])
                ax.set_yticklabels([0, 0.15, 0.30])
            inset = inset_axes(
                ax, width="40%", height="35%", loc='upper right',
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0.6)
            inset.plot(
                thresholds, real_vis[i]-np.mean(shuffle_vis[i][j], axis=1),
                color='#006400', linewidth=2.5, label='Diff')
            inset.set_xlim([0, 15])
            inset.set_xticks([0, 5, 10, 15])
            inset.set_xticklabels([0, 50, 100, 150])
    shuffle_patch = Patch(
        facecolor='gray', edgecolor='black', alpha=0.8, label='Shuffle')
    # axes[0, 2].legend()
    lines, labels = axes[1, 2].get_legend_handles_labels()
    # axes[1, 2].legend(lines + [shuffle_patch], labels + ['Shuffle'])
    plt.savefig('./temp/shuffle.png', dpi=300)
    plt.savefig('./temp/shuffle.svg')
    plt.close()

    # D. Visualization: shuffled preplay VS shuffled replay
    thresholds = np.arange(0, 16)
    linecolor_vis = ['#56B4E9', '#D55E00']
    shuffle_pre_vis = [shuffle_pre_timebin, shuffle_pre_neuron, shuffle_pre_pf]
    shuffle_post_vis = [shuffle_post_timebin, shuffle_post_neuron, shuffle_post_pf]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=False, constrained_layout=True)
    for i in range(3):
        ax = axes[i]
        # --- Violin plots: Preplay ---
        parts = ax.violinplot(
            dataset=[shuffle_pre_vis[i][t] for t in range(len(thresholds))],
            positions=thresholds,
            widths=(thresholds[1] - thresholds[0]) * 0.5,
            showextrema=False,
            showmeans=False)
        for pc in parts['bodies']:
            pc.set_alpha(None)
            r,g,b = mcolors.to_rgb('#56B4E9')
            pc.set_facecolor((r, g, b, 0.2))
            pc.set_edgecolor((0,0,0,0.8))
            pc.set_linewidth(0.8)
        # --- Violin plots: Replay ---
        parts = ax.violinplot(
            dataset=[shuffle_post_vis[i][t] for t in range(len(thresholds))],
            positions=thresholds,
            widths=(thresholds[1] - thresholds[0]) * 0.5,
            showextrema=False,
            showmeans=False)
        for pc in parts['bodies']:
            pc.set_alpha(None)
            r,g,b = mcolors.to_rgb('#D55E00')
            pc.set_facecolor((r, g, b, 0.2))
            pc.set_edgecolor((0,0,0,0.8))
            pc.set_linewidth(0.8)
        # --- Mean curves ---
        ax.plot(
            thresholds, [np.mean(shuffle_pre_vis[i][t]) for t in range(len(thresholds))],
            color='#56B4E9', marker='o', ms=6, lw=2.5, zorder=10, label='Preplay')
        ax.plot(
            thresholds, [np.mean(shuffle_post_vis[i][t]) for t in range(len(thresholds))],
            color='#D55E00', marker='o', ms=6, lw=2.5, zorder=10, label='Replay')
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xticklabels([0, 50, 100, 150])
        ax.set_ylim([0, None])
        ax.set_ylim([0, 0.2])
        ax.set_yticks([0, 0.1, 0.2])
        ax.set_yticklabels([0, 0.1, 0.2])
    shuffle_pre_patch = Patch(
        facecolor='#56B4E9', edgecolor='black', alpha=0.5, label='Preplay')
    shuffle_post_patch = Patch(
        facecolor='#D55E00', edgecolor='black', alpha=0.5, label='Replay')
    handles, labels = axes[2].get_legend_handles_labels()
    handles = handles + [shuffle_pre_patch, shuffle_post_patch]
    labels = [h.get_label() for h in handles]
    axes[2].legend(handles, labels)
    plt.savefig('./temp/preplayVSreplay.png', dpi=300)
    plt.savefig('./temp/preplayVSreplay.svg')
    plt.close()

    # E. Visualization: distribution
    ### Histogram
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    # ax.hist(
    #     ddtimes_o_pre_sess[ddtimes_o_pre_sess>0], bins=20,
    #     color='#56B4E9', alpha=0.5, label='Preplay',
    #     edgecolor='white', linewidth=1, density=True)
    # ax.hist(
    #     ddtimes_o_post_sess[ddtimes_o_post_sess>0], bins=20,
    #     color='#D55E00', alpha=0.5, label='Replay',
    #     edgecolor='white', linewidth=1, density=True)
    # ax.legend()
    # plt.savefig('./temp/ddtime_hist.png', dpi=300)
    # plt.savefig('./temp/ddtime_hist.pdf')
    # plt.close()

    ### CDF
    x_val = np.linspace(0, 20, 200)
    # cdf_pre = calc_cdf(ddtimes_o_pre_sess, x_val)
    # cdf_post = calc_cdf(ddtimes_o_post_sess, x_val)
    cdf_pre = calc_kde_cdf(ddtimes_o_pre_sess, x_val)
    cdf_post = calc_kde_cdf(ddtimes_o_post_sess, x_val)
    print('ddtime KS p-value:', ks_2samp(ddtimes_o_post_sess, ddtimes_o_pre_sess, alternative='less')[1])
    print('ddtime MWU p-value:', mannwhitneyu(ddtimes_o_post_sess, ddtimes_o_pre_sess, alternative='greater')[1])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax.plot(x_val, cdf_pre, color='#56B4E9', linewidth=2.5, label='Preplay')
    ax.plot(x_val, cdf_post, color='#D55E00', linewidth=2.5, label='Replay')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 1.01])
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticklabels([0, 100, 200])
    inset = inset_axes(
        ax, width="40%", height="35%", loc='lower right',
        bbox_to_anchor=(0.0, 0.08, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.6)
    inset.plot(x_val, cdf_pre-cdf_post, color='#006400', linewidth=2.5, label='Diff')
    inset.set_xlim([0, 20])
    inset.set_xticks([0, 10, 20])
    inset.set_xticklabels([0, 100, 200])
    ax.legend()
    plt.savefig('./temp/ddtime_cdf.png', dpi=300)
    plt.savefig('./temp/ddtime_cdf.svg')
    plt.close()

    cdf_s_pre_timebin = calc_kde_cdf(ddtimes_s_pre_timebin.ravel(), x_val)
    cdf_s_pre_neuron = calc_kde_cdf(ddtimes_s_pre_neuron.ravel(), x_val)
    cdf_s_pre_pf = calc_kde_cdf(ddtimes_s_pre_pf.ravel(), x_val)
    cdf_s_post_timebin = calc_kde_cdf(ddtimes_s_post_timebin.ravel(), x_val)
    cdf_s_post_neuron = calc_kde_cdf(ddtimes_s_post_neuron.ravel(), x_val)
    cdf_s_post_pf = calc_kde_cdf(ddtimes_s_post_pf.ravel(), x_val)

    linecolor_vis = ['#56B4E9', '#D55E00']
    real_ddtime = [ddtimes_o_pre_sess, ddtimes_o_post_sess]
    shuffle_ddtime = [
        [ddtimes_s_pre_timebin.ravel(), ddtimes_s_pre_neuron.ravel(), ddtimes_s_pre_pf.ravel()],
        [ddtimes_s_post_timebin.ravel(), ddtimes_s_post_neuron.ravel(), ddtimes_s_post_pf.ravel()]]
    real_vis = [cdf_pre, cdf_post]
    shuffle_vis = [
        [cdf_s_pre_timebin, cdf_s_pre_neuron, cdf_s_pre_pf],
        [cdf_s_post_timebin, cdf_s_post_neuron, cdf_s_post_pf]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False, constrained_layout=True)
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(x_val, real_vis[i], color=linecolor_vis[i], linewidth=2.5)
            ax.plot(x_val, shuffle_vis[i][j], color='gray', linewidth=2.5)
            ax.set_xlim([0, 20])
            ax.set_xticks([0, 10, 20])
            ax.set_xticklabels([0, 100, 200])
            ax.set_ylim([0, 1.01])
            ax.set_yticks([0, 0.5, 1.0])
            inset = inset_axes(
                ax, width="40%", height="35%", loc='lower right',
                bbox_to_anchor=(0.0, 0.08, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0.6)
            inset.plot(x_val, real_vis[i]-shuffle_vis[i][j], color='#006400', linewidth=2.5, label='Diff')
            inset.set_xlim([0, 20])
            inset.set_xticks([0, 10, 20])
            inset.set_xticklabels([0, 100, 200])
    plt.savefig('./temp/ddtime_cdf_shuffle.png', dpi=300)
    plt.savefig('./temp/ddtime_cdf_shuffle.svg')
    plt.close()            

    # Examine p-values
    def pvalues_threshods(real_ddtime, shuffle_ddtime, thre):
        p_value = np.zeros((2, 3), dtype=float)
        p_str = np.empty((2, 3), dtype=object)
        for i in range(2):
            for j in range(3):
                real_selected = real_ddtime[i][real_ddtime[i] > thre]
                shuffle_selected = shuffle_ddtime[i][j][shuffle_ddtime[i][j] > thre]
                p_value[i, j] = ks_2samp(real_selected, shuffle_selected, alternative='less')[1]
                p_str[i, j] = pval_to_stars(p_value[i, j])
        return p_value, p_str

    real_ddtime = [ddtimes_o_pre_sess, ddtimes_o_post_sess]
    shuffle_ddtime = [
        [ddtimes_s_pre_timebin.ravel(), ddtimes_s_pre_neuron.ravel(), ddtimes_s_pre_pf.ravel()],
        [ddtimes_s_post_timebin.ravel(), ddtimes_s_post_neuron.ravel(), ddtimes_s_post_pf.ravel()]]
    p_value, p_str = pvalues_threshods(real_ddtime, shuffle_ddtime, thre=-1)
    print('>= 0 ms:')
    print(p_value)
    print(p_str)
    p_value, p_str = pvalues_threshods(real_ddtime, shuffle_ddtime, thre=0)
    print('> 0 ms:')
    print(p_value)
    print(p_str)
    p_value, p_str = pvalues_threshods(real_ddtime, shuffle_ddtime, thre=5)
    print('> 50 ms:')
    print(p_value)
    print(p_str)
    p_value, p_str = pvalues_threshods(real_ddtime, shuffle_ddtime, thre=8)
    print('> 80 ms:')
    print(p_value)
    print(p_str)
    ######

###-----Drift & Diffusion Parameter-----###
if 0:
    lamb_pre, lamb_post = [], []
    sig_pre, sig_post = [], []
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
    num_preplays = [420, 776, 812, 575, 631, 0, 1428, 1373]
    for rf, num_pre in zip(result_folders, num_preplays):
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/dd_ripples.npy'))
        optim_results_hdd = f1['optim_results']
        dd_ripple = f2
        f1.close()
        dd_ripple_pre = dd_ripple[dd_ripple < num_pre]
        dd_ripple_post = dd_ripple[dd_ripple >= num_pre]
        lamb_pre.append(np.abs(optim_results_hdd[dd_ripple_pre, 2]))
        lamb_post.append(np.abs(optim_results_hdd[dd_ripple_post, 2]))
        sig_pre.append(optim_results_hdd[dd_ripple_pre, 4])
        sig_post.append(optim_results_hdd[dd_ripple_post, 4])
    lamb_pre = np.concatenate(lamb_pre)
    lamb_post = np.concatenate(lamb_post)
    sig_pre = np.concatenate(sig_pre)
    sig_post = np.concatenate(sig_post)

    ###
    x_val_lamb = np.linspace(0, lamb_post.max(), 200)
    # cdf_lamb_pre = calc_cdf(lamb_pre, x_val_lamb)
    # cdf_lamb_post = calc_cdf(lamb_post, x_val_lamb)
    cdf_lamb_pre = calc_kde_cdf(lamb_pre, x_val_lamb)
    cdf_lamb_post = calc_kde_cdf(lamb_post, x_val_lamb)
    print('lambda p-value:', pval_to_stars(ks_2samp(lamb_pre, lamb_post, alternative='two-sided')[1]))
    ###
    x_val_sig = np.linspace(sig_post.min(), sig_post.max(), 200)
    # cdf_sig_pre = calc_cdf(sig_pre, x_val_sig)
    # cdf_sig_post = calc_cdf(sig_post, x_val_sig)
    cdf_sig_pre = calc_kde_cdf(sig_pre, x_val_sig)
    cdf_sig_post = calc_kde_cdf(sig_post, x_val_sig)
    print('sigma p-value:', pval_to_stars(ks_2samp(sig_pre, sig_post, alternative='two-sided')[1]))
    ###
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].plot(x_val_lamb, cdf_lamb_pre, color='#56B4E9', linewidth=2.5, label='Preplay')
    axes[0].plot(x_val_lamb, cdf_lamb_post, color='#D55E00', linewidth=2.5, label='Replay')
    axes[0].set_xlim([0, None])
    axes[0].set_xticks([0, 150, 300, 450, 600, 750])
    axes[0].set_xticklabels([0, 3, 6, 9, 12, 15])
    inset = inset_axes(
        axes[0], width="40%", height="35%", loc='lower right',
        bbox_to_anchor=(0.0, 0.08, 1.0, 1.0),
        bbox_transform=axes[0].transAxes,
        borderpad=0.6)
    inset.plot(x_val_lamb, cdf_lamb_pre-cdf_lamb_post, color='#006400', linewidth=2.5, label='Diff')
    inset.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    inset.set_xlim([0, None])
    inset.set_xticks([0, 250, 500, 750])
    inset.set_xticklabels([0, 5, 10, 15])
    axes[1].plot(x_val_sig, cdf_sig_pre, color='#56B4E9', linewidth=2.5, label='Preplay')
    axes[1].plot(x_val_sig, cdf_sig_post, color='#D55E00', linewidth=2.5, label='Replay')
    axes[1].set_xlim([5, None])
    axes[1].set_xticks([5, 15, 25, 35, 45])
    axes[1].set_xticklabels([0.1, 0.3, 0.5, 0.7, 0.9])
    inset = inset_axes(
        axes[1], width="40%", height="35%", loc='lower right',
        bbox_to_anchor=(0.0, 0.08, 1.0, 1.0),
        bbox_transform=axes[1].transAxes,
        borderpad=0.6)
    inset.plot(x_val_sig, cdf_sig_pre-cdf_sig_post, color='#006400', linewidth=2.5, label='Diff')
    inset.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    inset.set_xlim([5, None])
    inset.set_xticks([5, 25, 45])
    inset.set_xticklabels([0.1, 0.5, 0.9])
    for ax in axes.ravel():
        ax.set_ylim([0, 1.01])
        ax.set_yticks([0, 0.5, 1.0])
    axes[1].legend()
    plt.savefig('./temp/lambsig_cdf.png', dpi=300)
    plt.savefig('./temp/lambsig_cdf.svg')
    plt.close()
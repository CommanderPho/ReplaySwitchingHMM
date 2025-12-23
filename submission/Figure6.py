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


###-----Simulate trajectory with Brownian motion-----###
if 0:
    rng = np.random.default_rng(2025)

    # Calculate transition matrix
    T = make_driftdiffusion_transition(n_position_bins=120, lamb=0, sig=45, dt=0.01)

    # Simulate trajectories
    num_session = 20
    num_traj = 50
    init_state = 60
    n_steps = 30
    n_states = T.shape[0]
    sim_sess = []
    for i in tqdm(range(num_session)):
        sim_traj = []
        for i in range(num_traj):
            traj = np.empty(n_steps, dtype=float)
            curr = init_state
            traj[0] = curr
            for t in range(1, n_steps):
                p = T[curr, :]
                p /= p.sum()
                curr = np.random.choice(n_states, p=p)
                traj[t] = curr
            sim_traj.append(traj)
        sim_sess.append(sim_traj)
    
    with open("temp/sim_diffusion.pkl", "wb") as f:
        pickle.dump(sim_sess, f)


###-----Stella et al. 2019 diffusion hypothesis-----###
# Untility function
def calc_metrics(dt_vis, pos_data):
    ### 1. Collect dx-dt
    dzdt = dict()
    for p_dec in pos_data:
        p_dec = np.array(p_dec)
        max_dt = p_dec.shape[0] - 3
        for dt in range(1, max_dt + 1):
            alf = np.abs(p_dec[dt:] - p_dec[:-dt]) # absolute location diff
            if dt not in dzdt:
                dzdt[dt] = []
            dzdt[dt].extend(alf)
    
    ### 2. Calculate MSD, RMSD, slope in log-log scale
    msd = np.zeros_like(dt_vis, dtype=float)
    rmsd = np.zeros_like(dt_vis, dtype=float)
    for i, dt in enumerate(dt_vis):
        sd = np.array(dzdt[dt])**2          # square distance (SD)
        msd[i] = sd.mean()                  # mean square distance (MSD)
        rmsd[i] = np.sqrt(msd[i])           # root mean square distance (RMS)
    slope = linregress(np.log10(dt_vis), np.log10(rmsd))[0]

    return msd, rmsd, slope
    ##################
    ### B. Collect dx-dt
    # dxdt_session = []
    # for p_ses in pos_session:
    #     abslocdiff = dict()
    #     for p_dec in p_ses:
    #         p_dec = np.array(p_dec)
    #         max_dt = p_dec.shape[0] - 3
    #         for dt in range(1, max_dt + 1):
    #             alf = np.abs(p_dec[dt:] - p_dec[:-dt])
    #             if dt not in abslocdiff:
    #                 abslocdiff[dt] = []
    #             abslocdiff[dt].extend(alf)
    #     dxdt_session.append(abslocdiff)
    ### C. Calculate results for visualization
    # dt_vis = np.arange(1, 11)
    # msd_vis, rmsd_vis, slope_vis = [], [], []
    # for dxdt_ses in dxdt_session:
    #     msd = np.zeros_like(dt_vis, dtype=float)
    #     rmsd = np.zeros_like(dt_vis, dtype=float)
    #     for i, dt in enumerate(dt_vis):
    #         sd = np.array(dxdt_ses[dt])**2        # square distance (SD)
    #         msd[i] = sd.mean()                    # mean square distance (MSD)
    #         rmsd[i] = np.sqrt(msd[i])             # root mean square distance (RMS)
    #     msd_vis.append(msd)
    #     rmsd_vis.append(rmsd)
    #     slope_vis.append(linregress(np.log10(dt_vis), np.log10(rmsd))[0])
    # msd_vis = np.array(msd_vis)
    # rmsd_vis = np.array(rmsd_vis)
    # slope_vis = np.array(slope_vis)
    ##################

# Retrieve inference results
if 0:
    with np.load('submission/SupFigure6b/ripples_class.npz', allow_pickle=True) as data:
        dd_ripples = list(data['dd'])
        st_ripples = list(data['st'])
        fr_ripples = list(data['fr'])

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

    optim_results_session = []
    pos_session = []
    for j, (df, rf, num_pre) in enumerate(zip(data_folders, result_folders, num_preplays)):
        optim_results_this_sess = []
        pos_this_sess = []

        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/optim_results.npz'), allow_pickle=True)
        optim_results_hdd = f1['optim_results']
        place_fields = f1['place_fields']
        ripple_spike_trains = f1['ripple_spike_trains']
        time_bin_size_replay = f1['time_window']
        f1.close()

        f2 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/dd_ripples.npy'))
        dd_ripple = f2
            # f2
            # dd_ripples[j]
            # st_ripples[j]
            # fr_ripples[j]

        # for ripple_i in tqdm(range(len(ripple_spike_trains))):
        for ripple_i in tqdm(dd_ripple):
            if ripple_i < num_pre:
                continue
            ######
            # model_type = 'hybrid_drift_diffusion'
            # # state_names = \
            # #     ['driftdiffusion1', 'driftdiffusion2', 'fragmented', 'stationary']
            # state_names = \
            #     ['driftdiffusion1', 'fragmented', 'stationary']
            # optim_results_vis = optim_results_hdd
            ######
            # model_type = 'drift_diffusion'
            # state_names = \
            #     ['driftdiffusion1', 'driftdiffusion2']
            # optim_results_vis = optim_results_dd
            ######
            model_type = 'diffusion'
            state_names = \
                ['diffusion1', 'diffusion2']
            optim_results_vis = None # optim_results_d
            optim_results_this_sess.append(optim_results_hdd[ripple_i, :])
            ######
            ripple_spike_train = ripple_spike_trains[ripple_i]
            optim_params = (0.98, 0, 0, 50) # fixed diffusion model
                # optim_results_vis[ripple_i, 1:5] # diag, lambda1, lambda2, sigma
            optim_model, causal_posterior, acausal_posterior, data_log_likelihood = \
                perform_inference(
                    optim_params, place_fields, ripple_spike_train, time_bin_size_replay, model_type)
            state_marginal = np.sum(acausal_posterior, axis=2)
            position_marginal = np.sum(acausal_posterior, axis=1)
            posterior = position_marginal.squeeze()
            position_estimate = np.argmax(posterior, axis=1)
            pos_this_sess.append(position_estimate)
        optim_results_session.append(optim_results_this_sess)
        pos_session.append(pos_this_sess)

    with open("temp/optim_results_session.pkl", "wb") as f:
        pickle.dump(optim_results_session, f)
    with open("temp/pos_session.pkl", "wb") as f:
        pickle.dump(pos_session, f)

# Organize data by sessions
if 0:
    #---Drift-diffusion ripples
    # with open("submission/Figure6/pos_session.pkl", "rb") as f:
    #     pos_session = pickle.load(f)
    # real_sess = 9
    #---All ripples
    # with open("submission/Figure6/pos_session_allSWRs.pkl", "rb") as f:
    #     pos_session = pickle.load(f)
    # real_sess = 9
    #---Drift-diffusion ripples (from Figure10.py)
    # with open("submission/SupFigure6b/dd_ripples/pos_session.pkl", "rb") as f:
    #     pos_session = pickle.load(f)
    # real_sess = 9
    #---Stationary ripples (from Figure10.py)
    # with open("submission/SupFigure6b/st_ripples/pos_session.pkl", "rb") as f:
    #     pos_session = pickle.load(f)
    # del pos_session[6] # less than 50 stationary ripples in this session
    # real_sess = 8
    #---Uniform ripples (from Figure10.py)
    with open("submission/SupFigure6b/fr_ripples/pos_session.pkl", "rb") as f:
        pos_session = pickle.load(f)
    del pos_session[6] # less than 50 uniform ripples in this session
    real_sess = 8
    
    with open("submission/Figure6/sim_diffusion.pkl", "rb") as f:
        sim_sess = pickle.load(f)

    pos_session.extend(sim_sess)

    ####---Calculate metrics for each session---###
    dt_vis = np.arange(1, 11)
    msd_vis, rmsd_vis, slope_vis = \
        np.zeros((len(pos_session), len(dt_vis)), dtype=float),\
        np.zeros((len(pos_session), len(dt_vis)), dtype=float),\
        np.zeros((len(pos_session)), dtype=float)
    for i, p_ses in enumerate(pos_session):
        msd, rmsd, slope = calc_metrics(dt_vis, p_ses)
        msd_vis[i, :] = msd
        rmsd_vis[i, :] = rmsd
        slope_vis[i] = slope

    ###---Visualization---###
    x, x_log = np.concatenate(([0], dt_vis * 10)), np.log10(dt_vis * 10) # 10 ms time bin
    rmsd_vis = rmsd_vis*2 # 2 cm spatial bin
    mean_data_lin, std_data_lin = \
         np.concatenate(([0], np.mean(msd_vis[0:real_sess, :], axis=0))), \
         np.concatenate(([0], np.std(msd_vis[0:real_sess, :], axis=0)))
    # mean_data_lin, std_data_lin = \
    #     np.concatenate(([np.nan], np.mean(msd_vis[0:real_sess, :], axis=0))), \
    #     np.concatenate(([np.nan], np.std(msd_vis[0:real_sess, :], axis=0)))
    mean_sim_lin, std_sim_lin = \
        np.concatenate(([0], np.mean(msd_vis[real_sess:, :], axis=0))), \
        np.concatenate(([0], np.std(msd_vis[real_sess:, :], axis=0)))
    mean_data_log, std_data_log = \
        np.mean(np.log10(rmsd_vis[0:real_sess, :]), axis=0), \
        np.std(np.log10(rmsd_vis[0:real_sess, :]), axis=0)
    mean_sim_log, std_sim_log = \
        np.mean(np.log10(rmsd_vis[real_sess:, :]), axis=0), \
        np.std(np.log10(rmsd_vis[real_sess:, :]), axis=0)
    slope_data, slope_sim = slope_vis[0:real_sess], slope_vis[real_sess:]
    # print(slope_data, slope_sim)
    print(np.mean(slope_data), np.mean(slope_sim))
    print(np.std(slope_data), np.std(slope_sim))
    # print(linregress(x, mean_data)[0])
    # print(linregress(x, mean_sim)[0])
    raise EOFError

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ###
    axes[0].plot(x, mean_data_lin, linestyle='-', linewidth=2, marker='o', color='#8080FF', label='replay')
    axes[0].fill_between(x, mean_data_lin - std_data_lin, mean_data_lin + std_data_lin, color='#8080FF', alpha=0.2)
    axes[0].plot(x, mean_sim_lin, linestyle='-', linewidth=2, marker='o', color='gray', label='simulation')
    axes[0].fill_between(x, mean_sim_lin - std_sim_lin, mean_sim_lin + std_sim_lin, color='gray', alpha=0.2)
    axes[0].set_xlim(-2, 102)
    axes[0].set_ylim(-20, 820)
    axes[0].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_xticklabels([0, 20, 40, 60, 80, 100])
    axes[0].set_yticks([0, 200, 400, 600, 800])
    axes[0].set_yticklabels([0, 0.08, 0.16, 0.24, 0.32])
    axes[0].set_xlabel(r'$\Delta t ~\mathrm{(ms)}$')
    axes[0].set_ylabel(r'$\langle \Delta z^2 \rangle ~\mathrm{(m^2)}$')
    ###
    axes[1].plot(x_log, mean_data_log, linestyle='-', linewidth=2, marker='o', color='#8080FF', label='replay')
    axes[1].fill_between(x_log, mean_data_log - std_data_log, mean_data_log + std_data_log, color='#8080FF', alpha=0.2)
    axes[1].plot(x_log, mean_sim_log, linestyle='-', linewidth=2, marker='o', color='gray', label='simulation')
    axes[1].fill_between(x_log, mean_sim_log - std_sim_log, mean_sim_log + std_sim_log, color='gray', alpha=0.2)
    axes[1].set_xlabel(r'$\log_{10}(\Delta t)}$')
    axes[1].set_ylabel(r'$\log_{10}(\sqrt{\langle \Delta z^2 \rangle})$')
    axes[1].set_xlim(0.95, 2.05)
    # axes[1].set_ylim(0.5, 1.4)
    axes[1].set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    axes[1].set_xticklabels([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    axes[1].legend(loc='best')
    ###
    x_data = np.random.normal(0, 0.02, size=slope_data.size)
    x_sim = np.random.normal(1, 0.02, size=slope_sim.size)
    axes[2].scatter(x_data, slope_data, c='#8080FF', s=70, alpha=1.0, edgecolors='w', linewidths=1.0)
    axes[2].scatter(x_sim, slope_sim, c='gray', s=70, alpha=1.0, edgecolors='w', linewidths=1.0)
    axes[2].axhline(y=np.mean(slope_data), color='#8080FF', linestyle='--', linewidth=2)   
    axes[2].axhline(y=np.mean(slope_sim), color='grey', linestyle='--', linewidth=2)    
    axes[2].set_xlim([-0.5, 1.5])
    axes[2].set_ylim([0.2, 1.0])
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Replay', 'Simulation'])
    axes[2].set_ylabel('Slope')
    ###
    for ax in axes:
        ax.set_box_aspect(1) 
    # plt.tight_layout()
    plt.savefig('./temp/stella2019_session.png', dpi=300, bbox_inches='tight')
    plt.savefig('./temp/stella2019_session.svg', bbox_inches='tight')
    plt.close()

# Organize data by speed groups
if 0:
    with open("submission/Figure6/optim_results_session.pkl", "rb") as f:
        optim_results_session = pickle.load(f)
    with open("submission/Figure6/pos_session.pkl", "rb") as f:
        pos_session = pickle.load(f)
    with open("submission/Figure6/sim_diffusion.pkl", "rb") as f:
        sim_session = pickle.load(f)
    
    optim_results_session = np.concatenate(optim_results_session)
    lambda_session = optim_results_session[:, 2]
    speed_session = np.abs(lambda_session) * 0.02 # bin/sec -> m/sec
    pos_session = [x for sess in pos_session for x in sess]
    sim_session = [x for sess in sim_session for x in sess]

    speed_groups = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)] # m/sec
    idx_group = []
    for gi, (lo, hi) in enumerate(speed_groups):
        if gi < len(speed_groups) - 1:
            mask = (speed_session >= lo) & (speed_session < hi)
        else:
            mask = (speed_session >= lo) & (speed_session <= hi)
        idx_group.append(np.nonzero(mask)[0])
    pos_group = [[pos_session[i] for i in idx] for idx in idx_group]
    # for i, p_grp in enumerate(pos_group):
    #     print(len(p_grp))
    # raise EOFError

    ####---Calculate metrics for each speed group---###
    dt_vis = np.arange(1, 11)
    msd_vis, rmsd_vis, slope_vis = \
        np.zeros((len(pos_group), len(dt_vis)), dtype=float),\
        np.zeros((len(pos_group), len(dt_vis)), dtype=float),\
        np.zeros((len(pos_group)), dtype=float)
    for i, p_grp in enumerate(pos_group):
        msd, rmsd, slope = calc_metrics(dt_vis, p_grp)
        msd_vis[i, :] = msd
        rmsd_vis[i, :] = rmsd
        slope_vis[i] = slope
    msd_sim, rmsd_sim, slope_sim = calc_metrics(dt_vis, sim_session)

    ###---Visualization---###
    n_group = msd_vis.shape[0]
    x, x_log = dt_vis * 10, np.log10(dt_vis * 10) # 10 ms time bin
    rmsd_vis, rmsd_sim = rmsd_vis*2, rmsd_sim*2 # 2 cm spatial bin
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ###
    for gi in range(n_group):
        axes[0].plot(
            np.concatenate(([0], x)), np.concatenate(([0], msd_vis[gi, :])),
            linestyle='-', linewidth=2, marker='o', color='#8080FF')
    axes[0].plot(
        np.concatenate(([0], x)), np.concatenate(([0], msd_sim)),
        linestyle='-', linewidth=2, marker='o', color='gray')
    axes[0].set_xlim(-2, 102)
    axes[0].set_ylim(-20, 1300)
    axes[0].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_xticklabels([0, 20, 40, 60, 80, 100])
    axes[0].set_yticks([0, 300, 600, 900, 1200])
    axes[0].set_yticklabels([0, 0.12, 0.24, 0.36, 0.48])
    axes[0].set_xlabel(r'$\Delta t ~\mathrm{(ms)}$')
    axes[0].set_ylabel(r'$\langle \Delta z^2 \rangle ~\mathrm{(m^2)}$')
    ###
    for gi in range(n_group):
        axes[1].plot(x_log, np.log10(rmsd_vis[gi, :]), linestyle='-', linewidth=2, marker='o', color='#8080FF')
    axes[1].plot(x_log, np.log10(rmsd_sim), linestyle='-', linewidth=2, marker='o', color='gray')
    axes[1].set_xlabel(r'$\log_{10}(\Delta t)}$')
    axes[1].set_ylabel(r'$\log_{10}(\sqrt{\langle \Delta z^2 \rangle})$')
    axes[1].set_xlim(0.95, 2.05)
    # axes[1].set_ylim(0.5, 1.4)
    axes[1].set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    axes[1].set_xticklabels([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ###
    for ax in axes:
        ax.set_box_aspect(1) 
    plt.savefig('./temp/stella2019_speedgroup.png', dpi=300, bbox_inches='tight')
    plt.savefig('./temp/stella2019_speedgroup.svg', bbox_inches='tight')
    plt.close()

    
###-----Speed distribution from pure diffusion model-----###
if 0:
    # Retrieve inference results
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

    optim_results_session = []
    ripple_len_session = []
    for df, rf, num_pre in zip(data_folders, result_folders, num_preplays):
        optim_results_this_sess = []
        ripple_len_this_sess = []
        
        f1 = np.load(os.path.join(rf, 'dynamics_models/hybrid_drift_diffusion/dd_ripples.npy'))
        f2 = np.load(os.path.join(rf, 'dynamics_models/diffusion/optim_results.npz'), allow_pickle=True)
        dd_ripple = f1
        optim_results_diffusion = f2['optim_results']
        ripple_spike_trains = f2['ripple_spike_trains']
        time_bin_size_replay = f2['time_window']
        f2.close()
        # for ripple_i in tqdm(range(len(ripple_spike_trains))):
        for ripple_i in tqdm(dd_ripple):
            if ripple_i < num_pre:
                continue
            optim_results_this_sess.append(optim_results_diffusion[ripple_i, :])
            ripple_len_this_sess.append(ripple_spike_trains[ripple_i].shape[0])
        optim_results_session.append(optim_results_this_sess)
        ripple_len_session.append(ripple_len_this_sess)
    optim_results_session = np.concatenate(optim_results_session)
    ripple_len_session = np.concatenate(ripple_len_session)

    # Simulate trajectory
    rng = np.random.default_rng(2025)
    sim_traj = []
    for i in tqdm(range(len(optim_results_session))):
        num_traj = 10
        init_state = 45
        n_steps = ripple_len_session[i]
        sig = optim_results_session[i, 4]
        T = make_driftdiffusion_transition(
            n_position_bins=90, lamb=0, sig=sig, dt=0.01)
            # 90 bin = 1.8 m track
        n_states = T.shape[0]
        
        for j in range(num_traj):
            traj = np.empty(n_steps, dtype=float)
            curr = init_state
            traj[0] = curr
            for t in range(1, n_steps):
                p = T[curr, :]
                p /= p.sum()
                curr = np.random.choice(n_states, p=p)
                traj[t] = curr
            sim_traj.append(traj)

    # Estimate speed
    sim_speed = []
    for i in tqdm(range(len(sim_traj))):
        traj = sim_traj[i]
        reg = LinearRegression().fit(np.arange(len(traj)).reshape(-1, 1), traj)
        velocity = reg.coef_ * 0.02 / 0.01
            # 0.02 m/space bin; 0.01 s/time bin
        sim_speed.append(np.abs(velocity))
    sim_speed = np.concatenate(sim_speed)

    fig, axes = plt.subplots(1, 1, figsize=(3.5, 4))
    n_bins = 20
    bins = np.linspace(0, 15, n_bins + 1)
    ###
    axes.hist(sim_speed, bins=bins, color='grey', edgecolor='white', linewidth=1.5, density=True)
    axes.axvline(np.mean(sim_speed), color='tab:red', linestyle='--', linewidth=3)
    axes.set_xlim([0, 15])
    axes.set_xticks([0, 5, 10, 15])
    # plt.tight_layout()
    plt.savefig('./temp/speed_from_diffusion.png', dpi=300)
    plt.savefig('./temp/speed_from_diffusion.svg')
    plt.close()

    print(np.mean(sim_speed))
import os
import math
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from preprocessing_utils import (
    estimate_speed_2d,
    create_spike_raster,
    calc_place_fields_1d,
    calc_place_fields_1d_dir)


###### Settings vary across sessions
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat1/Linear1'
data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear1')
ADJUST_ANGLE = 2            # degree, see below Linearize Position section
XY_FLIP = False
###
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat1/Linear2'
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat1/Linear2')
# ADJUST_ANGLE = 2
# XY_FLIP = False
###
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat2/Linear1'
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear1')
# ADJUST_ANGLE = 0
# XY_FLIP = True
###
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat2/Linear2'
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat2/Linear2')
# ADJUST_ANGLE = 0
# XY_FLIP = True
###
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat5/Linear3'
# data_folder = os.path.expanduser('~/Datasets/Neural_data/Pfeiffer1D/Rat5/Linear3')
# ADJUST_ANGLE = -3
# XY_FLIP = True
###
# data_folder = 'G:/Neural_data/Pfeiffer1D/Rat5/Linear4'
# ADJUST_ANGLE = -3
# XY_FLIP = True
###

###### Settings shared across sessions
TIME_BIN_RUN = 0.05         # sec
TIME_BIN_REPLAY = 0.005     # sec
POS_BIN = 2                 # cm
TRACK_LEN = 180             # cm, not used
SPEED_THRE = 5              # cm/sec
KERNEL_WID = 4              # cm
RATE_THRE = 1               # Hz
######

###-----Load Data-----###
# Spike data
# (Time of spike, cell ID of spike)
spike_data = loadmat(os.path.join(data_folder, 'Spike_Data.mat'))['Spike_Data']
exci_neu = loadmat(os.path.join(data_folder, 'Spike_Data.mat'))['Excitatory_Neurons']
inhi_neu = loadmat(os.path.join(data_folder, 'Spike_Data.mat'))['Inhibitory_Neurons']
num_neurons = exci_neu.shape[0] + inhi_neu.shape[0]
assert num_neurons == int(np.max(np.concatenate([exci_neu, inhi_neu]))) # neuron index starts from 1

# Position data
# (Time, x-position (in cm), y-position (in cm), and head direction (in degrees))
# only recorded for running periods
position_data = loadmat(os.path.join(data_folder, 'Position_Data.mat'))['Position_Data']
if XY_FLIP:
    position_data = position_data[:, [0, 2, 1, 3]]

# LFP data
# (Time, voltage)
lfp_data = loadmat(os.path.join(data_folder, 'LFP_Data.mat'))['LFP_Data']

if 0:
    # Timestamps
    print('Timestamps')
    print('spike data:', np.min(spike_data[:, 0]), np.max(spike_data[:, 0]))
    print('position data:', np.min(position_data[:, 0]), np.max(position_data[:, 0]))
    print('lfp_data:', np.min(lfp_data[:, 0]), np.max(lfp_data[:, 0]))
    print('LFP sampling rate:', 1 / (lfp_data[1, 0] - lfp_data[0, 0]))

    # Head direction
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].plot(position_data[:, 0], position_data[:, 3])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Angle (degree)')
    axes[0].set_title('Head direction')
    axes[1].hist(position_data[:, 3], bins=20)
    axes[1].set_xlim([0, 90])
    axes[1].set_xlabel('Angle (degree)')
    axes[1].set_ylabel('Counts')
    axes[1].set_title('Head direction')
    plt.savefig(os.path.join('temp/', 'test_head_direction.png'))
    plt.close()

    # Position tracking
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(position_data[:, 1], position_data[:, 2], s=1, alpha=0.3)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 40])
    ax.set_xlabel('x position (cm)')
    ax.set_ylabel('y position (cm)')
    ax.set_title('Position')
    plt.savefig(os.path.join('temp/', 'test_position_tracking.png'))
    plt.close()

    # LFP data
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(lfp_data[:, 0], lfp_data[:, 1])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('LFP')
    axes[1].plot(lfp_data[0:10000, 0], lfp_data[0:10000, 1])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_title('LFP First 10k steps')
    plt.savefig(os.path.join('temp/', 'test_lfp.png'))
    plt.close()

    raise EOFError

###-----Calculate Speed-----###
# Estimate speed from changes in xy coordinates
speed = estimate_speed_2d(position_data[:, 1], position_data[:, 2], position_data[:, 0])
speed = np.clip(speed, 0, 150) # upper limit 150 cm/s

if 0:
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes[0].scatter(position_data[:, 0], speed, s=1)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Speed (cm/s)')
    axes[0].set_title('Speed')
    axes[1].scatter(position_data[0:2000, 0], speed[0:2000], s=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed (cm/s)')
    axes[1].set_title('Speed first 2k steps')
    axes[2].hist(speed, bins=20)
    axes[2].set_xlabel('Speed (cm/s)')
    axes[2].set_ylabel('Counts')
    axes[2].set_title('Speed hist')
    plt.savefig(os.path.join('temp/', 'speed.png'))
    plt.close()

    raise EOFError

###-----Linearize Position-----###
# Postions are slightly tilted, see test_position_tracking.png
# Find the tilted angle from test_head_direction.png
# Only use x coordinates for anlysis

xy_position = position_data[:, 1:3].copy()
x_mean, y_mean = np.mean(xy_position[:, 0]), np.mean(xy_position[:, 1])
xy_position[:, 0] -= x_mean
xy_position[:, 1] -= y_mean

angle_deg = ADJUST_ANGLE # rotation angle, can vary across sessions
angle_rad = np.radians(angle_deg)
rotation_matrix = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]])
xy_position = np.dot(xy_position, rotation_matrix)
xy_position[:, 0] += x_mean
xy_position[:, 1] += y_mean

if 0:
    # Adjusted position tracking
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(xy_position[:, 0], xy_position[:, 1], s=1, alpha=0.3)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 40])
    ax.set_xlabel('x position (cm)')
    ax.set_ylabel('y position (cm)')
    ax.set_title('Position')
    plt.savefig(os.path.join('temp/', 'adjusted_position_tracking.png'))
    plt.close()

    raise EOFError

###-----Setup Time Bins-----###
time_bin_size = TIME_BIN_RUN # in sec
time_bin_edges = np.linspace(
    position_data[0, 0], position_data[-1, 0] + 1e-10,
    int(1 + np.ceil((position_data[-1, 0] + 1e-10 - position_data[0, 0]) / TIME_BIN_RUN)))
    # add 1e-10: time value wont reach rightmost edge
time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2

# Notice time range is larger for replay analysis
start_time = np.min([np.min(spike_data[:, 0]), lfp_data[0, 0]])
end_time = np.max([np.max(spike_data[:, 0]), lfp_data[-1, 0]])
time_bin_edges_replay = np.linspace(
    start_time, end_time + 1e-10,
    int(1 + np.ceil((end_time + 1e-10 - start_time) / TIME_BIN_REPLAY)))
    # add 1e-10: time value wont reach rightmost edge
time_bin_centers_replay = (time_bin_edges_replay[:-1] + time_bin_edges_replay[1:]) / 2

if 0:
    print(time_bin_size)
    print(time_bin_edges.shape, time_bin_centers.shape)
    print(time_bin_edges[1]-time_bin_edges[0], time_bin_centers[1]-time_bin_centers[0])

    raise EOFError

###-----Interpolate Position & Speed-----###
x_pos_intp = np.interp(time_bin_centers, position_data[:, 0], xy_position[:, 0])
x_pos_intp -= np.min(x_pos_intp)
speed_intp = np.interp(time_bin_centers, position_data[:, 0], speed)

if 0:
    print(position_data[:, 0].shape, xy_position[:, 0].shape, speed.shape)
    print(time_bin_centers.shape, x_pos_intp.shape, speed_intp.shape)

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))
    axes[0].scatter(position_data[:, 0], xy_position[:, 0], s=1)
    axes[0].scatter(time_bin_centers, x_pos_intp, s=1)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('X-Position (cm)')
    axes[0].set_title('X-Position Interpolation')
    axes[1].scatter(position_data[:, 0], speed, s=1)
    axes[1].scatter(time_bin_centers, speed_intp, s=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed (cm/s)')
    axes[1].set_title('Speed Interpolation')
    plt.savefig(os.path.join('temp/', 'position_speed_interpolation.png'))
    plt.close()

    raise EOFError

###-----Setup Position Bins-----###
pos_bin_size = POS_BIN # in cm
pos_bin_edges = np.linspace(
    np.min(x_pos_intp), np.max(x_pos_intp) + 1e-10,
    int(1 + np.ceil((np.max(x_pos_intp) + 1e-10 - np.min(x_pos_intp)) / pos_bin_size)))
    # add 1e-10: pos value wont reach rightmost edge
pos_bin_centers = (pos_bin_edges[:-1] + pos_bin_edges[1:]) / 2

if 0:
    print(pos_bin_size)
    print(pos_bin_edges.shape, pos_bin_centers.shape)
    print(pos_bin_edges[1]-pos_bin_edges[0], pos_bin_centers[1]-pos_bin_centers[0])

    raise EOFError

###-----Spike Raster-----###
spike_neu_id = spike_data[:, 1] - 1 # neuron index starts from 1
spike_raster_run = create_spike_raster(
    spike_time=spike_data[:, 0],
    spike_neu_id=spike_neu_id,
    num_neurons=num_neurons,
    time_bin_edges=time_bin_edges)
spike_raster_replay = create_spike_raster(
    spike_time=spike_data[:, 0],
    spike_neu_id=spike_neu_id,
    num_neurons=num_neurons,
    time_bin_edges=time_bin_edges_replay)

if 0:
    print(spike_raster_run.shape)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.matshow(spike_raster_run[0:200, :].T)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_title('Spike Raster Plot All Neurons Running Period')
    plt.savefig(os.path.join('temp/', 'spike_raster_run.png'))
    plt.close()

    raise EOFError

###-----Place Fields-----###
place_fields = calc_place_fields_1d(
    spike_raster=spike_raster_run,
    pos=x_pos_intp,
    speed=speed_intp,
    pos_bin_edges=pos_bin_edges,
    pos_bin_size=pos_bin_size,
    time_bin_size=time_bin_size,
    speed_threshold=SPEED_THRE,
    kernel_wid=KERNEL_WID)

place_fields_LtoR, place_fields_RtoL = calc_place_fields_1d_dir(
    spike_raster=spike_raster_run,
    pos=x_pos_intp,
    speed=speed_intp,
    pos_bin_edges=pos_bin_edges,
    pos_bin_size=pos_bin_size,
    time_bin_size=time_bin_size,
    speed_threshold=SPEED_THRE,
    kernel_wid=KERNEL_WID)

if 0:
    print(place_fields.shape)
    output_dir = './place_fields'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for neu_id in range(place_fields.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(place_fields[:, neu_id])
        ax.set_xlabel('Position Bin')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'Place Field of Neuron {neu_id}')
        plt.savefig(os.path.join(output_dir, f'neuron_{neu_id}.png'))
        plt.close()
    
    raise EOFError

###-----Processing Place Fields-----###
# 1. Exclude inhibitory and non-firing neurons
inhi_neu = (inhi_neu - 1).astype(np.int64) # neuron index starts from 1
valid_indicies = np.array([True] * num_neurons)
valid_indicies[inhi_neu] = False
max_rates = np.max(place_fields, axis=0)
valid_indicies[np.where(max_rates < RATE_THRE)] = False

place_fields_final = place_fields[:, valid_indicies]
place_fields_LtoR, place_fields_RtoL = \
    place_fields_LtoR[:, valid_indicies], place_fields_RtoL[:, valid_indicies]
spike_raster_run_final = spike_raster_run[:, valid_indicies]
spike_raster_replay_final = spike_raster_replay[:, valid_indicies]

# 2. Sort place fields
sorted_indices = np.argsort(np.argmax(place_fields_final, axis=0))
place_fields_final = place_fields_final[:, sorted_indices]
place_fields_LtoR, place_fields_RtoL = \
    place_fields_LtoR[:, sorted_indices], place_fields_RtoL[:, sorted_indices]
spike_raster_run_final = spike_raster_run_final[:, sorted_indices]
spike_raster_replay_final = spike_raster_replay_final[:, sorted_indices]

if 0:
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))
    # axes[0].matshow(spike_raster_run_final[400:700, :].T)
    # axes[0].matshow(spike_raster_run_final[800:1300, :].T)
    # axes[0].matshow(spike_raster_run_final[1300:1800, :].T)
    axes[0].matshow(spike_raster_run_final[2400:2900, :].T)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Neuron')
    axes[0].set_title('Spike Raster Final')
    # axes[1].plot(x_pos_intp[400:700])
    # axes[1].plot(x_pos_intp[800:1300])
    # axes[1].plot(x_pos_intp[1300:1800])
    axes[1].plot(x_pos_intp[2400:2900])
    axes[1].set_ylim([0, 180])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (cm)')
    axes[1].set_title('X-Position Tracking 1D')
    plt.savefig(os.path.join('temp/', 'spike_raster_run_final.png'))
    plt.close()

    raise EOFError

if 0:
    output_dir = './place_fields_dir'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for neu_id in range(place_fields_final.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(place_fields_LtoR[:, neu_id], label='LtoR')
        ax.plot(place_fields_RtoL[:, neu_id], label='RtoL')
        ax.plot(place_fields_final[:, neu_id], label='Bidir')
        ax.set_xlabel('Position Bin')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'Place Field of Neuron {neu_id}')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'neuron_{neu_id}.png'))
        plt.close()

    raise EOFError

###-----Save Processed Data-----###
if 0:
    np.savez_compressed(
        'processed_data.npz',
        place_fields=place_fields_final,
        spike_raster_run=spike_raster_run_final,
        spike_raster_replay=spike_raster_replay_final,
        x_pos_intp=x_pos_intp,
        speed_intp=speed_intp,
        time_bin_size=time_bin_size,
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        time_bin_size_replay=TIME_BIN_REPLAY,
        time_bin_edges_replay=time_bin_edges_replay,
        time_bin_centers_replay=time_bin_centers_replay,
        pos_bin_size=pos_bin_size,
        pos_bin_edges=pos_bin_edges,
        pos_bin_centers=pos_bin_centers)


###-----submission/Figure9.py-----###
import matplotlib as mpl
from ssm.baseline import bayesian_decoding
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


if 0:
    bidir = place_fields_final / (1e-10 + np.max(place_fields_final, axis=0, keepdims=True))
    LtoR = place_fields_LtoR / (1e-10 + np.max(place_fields_LtoR, axis=0, keepdims=True))
    RtoL = place_fields_RtoL / (1e-10 + np.max(place_fields_RtoL, axis=0, keepdims=True))
    # normalized

    ### Plot as heatmaps
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # im = axes[0].imshow(
    #     bidir.T, aspect='auto', cmap='viridis', origin='upper')
    # axes[0].set_ylabel('Neuron')
    # axes[0].set_title('Bidirection')
    # im = axes[1].imshow(
    #     LtoR.T, aspect='auto', cmap='viridis', origin='upper')
    # axes[1].set_title('Left to right')
    # im = axes[2].imshow(
    #     RtoL.T, aspect='auto', cmap='viridis', origin='upper')
    # axes[2].set_title('Right to left')
    # for ax in axes:
    #     ax.set_xticks([0, 20, 40, 60, 80])
    #     ax.set_xticklabels([0, 0.4, 0.8, 1.2, 1.6])
    #     ax.set_xlabel('Position (m)')
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    # plt.savefig(os.path.join('temp/', 'sorted_place_fields.png'))
    # plt.savefig(os.path.join('temp/', 'sorted_place_fields.svg'))
    # plt.savefig(os.path.join('temp/', 'sorted_place_fields.pdf'))
    # plt.close()
    ###

    ### Plot as curves
    pfs = [bidir, LtoR, RtoL]
    n_bins = bidir.shape[0]
    n_neurons = bidir.shape[1]
    n_plots = n_neurons
    fig, axes = plt.subplots(nrows=n_plots, ncols=3, figsize=(12, 12), sharex=True, constrained_layout=False)
    for j in range(3):
        for i, ax in enumerate(axes[:, j]):
            x = np.arange(n_bins)
            y = pfs[j][:, i]
            ax.plot(x, y, color='black', linewidth=1.5)
            ax.fill_between(x, y, 0, color='black', alpha=0.3)
            ax.set_xlim(0, n_bins - 1)
            if i < n_plots - 1:
                ax.set_xticks([])
            else:
                ax.set_xticks([0, n_bins])
                ax.set_xticklabels(['0', '1.8'])
            ax.set_yticks([])
        axes[-1, j].set_xlabel('Position (m)', fontsize=30, labelpad=14)
    left = axes[0,0].get_position().x0
    bottom = axes[-1,0].get_position().y0
    top = axes[0,0].get_position().y1
    fig.text(
        left-0.015, bottom, '1',
        ha='right', va='bottom', fontsize=30)
    fig.text(
        left-0.015, top, f'{n_neurons}',
        ha='right', va='top', fontsize=30)
    fig.supylabel(
        'Neurons',
        x=left-0.1,
        fontsize=30, rotation=90, va='center')

    for i, ax in enumerate(axes.ravel()):
        ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=30, direction='out')
        ax.tick_params(axis='both', which='minor', length=3, width=1.0)

    # plt.tight_layout()
    plt.savefig('./temp/sorted_place_fields.png', dpi=300)
    plt.savefig('./temp/sorted_place_fields.svg')
    plt.savefig('./temp/sorted_place_fields.pdf')
    plt.close()

if 0:
    post, x_hat = bayesian_decoding(
        n=spike_raster_run_final.T,
        f_x=place_fields_final.T,
        dx=pos_bin_size,
        dt=time_bin_size,
        pos_bin_centers=pos_bin_centers,
        epsilon=1e-10)

    chunk_size = 3000
    total_bins = len(time_bin_centers)
    n_chunks = math.ceil(total_bins / chunk_size)
    fig, axes = plt.subplots(n_chunks, 1, figsize=(20, 4 * n_chunks), sharex=False, sharey=True, constrained_layout=True)
    if n_chunks == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_bins)
        inds = np.arange(start, end)

        norm_post = post[inds, :] / (1e-10 + np.max(post[inds, :], axis=1, keepdims=True))
        t0, t1 = time_bin_centers[inds][0], time_bin_centers[inds][-1]
        x0, x1 = pos_bin_centers[0], pos_bin_centers[-1]
        ax.imshow(norm_post.T, aspect='auto', cmap='hot', origin='lower', extent=[t0, t1, x0, x1])
        ax.plot(
            time_bin_centers[inds], x_pos_intp[inds],
            color='white', linewidth=4, alpha=0.8, label='True Pos')
        ax.set_xlim(t0, t1)
        ax.set_ylim(x0, x1)
        ax.set_yticks([x0, x1])
        ax.set_yticklabels([0, 1.8])
        ax.set_ylabel('Position (m)', fontsize=30)
        # ax.set_title(f'Time {t0:.1f}–{t1:.1f} s')
        # ax.legend(loc='upper right', facecolor='black', framealpha=0.5)
    axes[-1].set_xlabel('Time in session (s)', fontsize=30)

    for i, ax in enumerate(axes):
        ax.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=30, pad=10, direction='out')
        ax.tick_params(axis='both', which='minor', length=3, width=1.0)

    plt.savefig(os.path.join('temp/', 'bayesian_decoding.png'))
    plt.savefig(os.path.join('temp/', 'bayesian_decoding.svg'))
    plt.savefig(os.path.join('temp/', 'bayesian_decoding.pdf'))
    plt.close()
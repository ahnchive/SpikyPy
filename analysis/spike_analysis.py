import numpy as np
from polyface.PolyfaceUtil import get_room_loc, get_face_pos, eye_face_interaction
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import os

def moving_average(data, k):
    """
    Perform moving average smoothing.

    Parameters:
        data (list or numpy array): The input data to be smoothed.
        k (int): Number of data points to include in the moving average (including the current point).

    Returns:
        numpy array: Smoothed data.
    """
    if k < 1:
        raise ValueError("k must be greater than or equal to 1.")
    
    data = np.array(data)  # Ensure input is a numpy array for easy manipulation
    window_size = k
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # To maintain the length of the original data, prepend with the original data
    # for the initial points where the window is incomplete
    padding = np.cumsum(data[:window_size-1]) / np.arange(1, window_size)
    return np.concatenate([padding, smoothed_data])


def PSTH_by_face_polyface(trial_data, trial_info, wall_layout, 
                          tolerance=8, stim_time=0.45, bin_size=0.02,
                            baseline_buffer=0.15, num_smooth=3, drop_invalid=True,
                          cell_type='rML', save_path=None, subset=None, strict=False):
    """ Compute the PSTH for the onset of different events, i.e., looking at 
        target/non-target/not-face. It only considers the spike firing rates for the first
        bin after onset. Events shorter than the specified bin size will be dropped by default.
        Note that the code is specific to polyface environment.

        Inputs:
            - trial_data: pre-processed trial data.
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - tolerance: tolerance angle between eye/object ray (to be changed to distance-based measurement).
            - stim_time: time after the stimulus onset to compute the PSTH.
            - bin_size: bin size (in second) to compute the fine-grained PSTH.
            - num_smooth: number of historical bins for smoothing.
            - baseline_buffer: K ms before event onset as baseline.
            - drop_invalid: dropping the invalid data (shorter than specified stim_time) or not .
            - cell_type: type of cells for consideration
            - save_path: path for saving the histogram plot
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - strict: If yes, only consider the target face responses from the correct trials, and non-target
                    face response from the wrong trials.
    """
    os.makedirs(save_path, exist_ok=True)

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]

    # get the locations of faces in different trials
    face_loc = get_face_pos(trial_info)
    
    # psth averaged over the whole stim onset
    psth = {'target_face': [[] for _ in range(len(cell_idx))], 
            'non_target_face':[[] for _ in range(len(cell_idx))], 
            'non_face': [[] for _ in range(len(cell_idx))]}

    # fine-grained PSTH smoothed by bins (for face interaction only)
    psth_fg = {'target_face': [[] for _ in range(len(cell_idx))], 
            'non_target_face':[[] for _ in range(len(cell_idx))], 
            'non_face': [[] for _ in range(len(cell_idx))]}

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue

        # determine the type of the current trial
        for type_ in ['End_Correct', 'End_Miss', 'End_Wrong']:
            if not np.isnan(trial_data['Paradigm']['PolyFaceNavigator'][type_][trial_idx]):
                cur_type = type_
                break
    
        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx]
        trial_offset = trial_data['Paradigm']['PolyFaceNavigator'][cur_type][trial_idx]

        # obtain eye interaction periods
        interaction_block = eye_face_interaction(trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx],
                                                trial_data['Paradigm']['PolyFaceNavigator']['Eye_arena'][trial_idx],
                                                face_loc[trial_number], tolerance=tolerance, wall_layout=wall_layout)

        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]

            # measure PSTH for each individual face interaction event
            all_face_event = []
            for face in interaction_block:
                for event in interaction_block[face]:
                    if event[0]-trial_onset <= stim_time: # ignore events that are too close to cue phase
                        continue
                    all_face_event.append((event[0], event[1]))
                    event_time = event[1]-event[0]
                    event_onset = event[0]
                    if event_time<stim_time:
                        if drop_invalid:
                            continue
                        else:
                            event_offset = event[1]
                    else:
                        event_offset = event_onset + stim_time
                    
                    # compute the averaged firing rate throughout the whole stim onset time
                    spike_count = len([1 for spike_time in neuron_spikes 
                                if spike_time>=event_onset and spike_time<=event_offset])
                    firing_rate = spike_count/event_time

                    # compute the fine PSTH
                    num_bin = int((stim_time+baseline_buffer)/bin_size)
                    bin_time = np.linspace(event_onset-baseline_buffer, event_offset, num_bin+1)
                    tmp_count = np.zeros(num_bin, )
                    for bin_idx in range(num_bin):
                        tmp_count[bin_idx] = len([1 for spike_time in neuron_spikes 
                                if spike_time>=bin_time[bin_idx] and spike_time<=bin_time[bin_idx+1]])
                        tmp_count[bin_idx] /= bin_size

                    if face_loc[trial_number][face]['isTarget']:
                        if not strict or (strict and cur_type=='End_Correct'):
                            psth['target_face'][neuron_idx].append(firing_rate)
                            psth_fg['target_face'][neuron_idx].append(moving_average(tmp_count, num_smooth))                            
                    else:
                        if not strict or (strict and cur_type=='End_Wrong'):
                            psth['non_target_face'][neuron_idx].append(firing_rate)
                            psth_fg['non_target_face'][neuron_idx].append(moving_average(tmp_count, num_smooth))

            # measure PSTH for non-face interaction event
            buffer = 0.2 # ignore the period right before/after looking at a face
            all_face_event = sorted(all_face_event)
            non_face_event = []
            current_time = trial_onset
            for start, end in all_face_event:
                adjusted_start = max(trial_onset, start - buffer)
                adjusted_end = min(trial_offset, end + buffer)
                if adjusted_start > current_time:
                    non_face_event.append((current_time, adjusted_start ))
                current_time = max(current_time, adjusted_end)       

            if current_time < trial_offset:
                non_face_event.append((current_time, trial_offset))

            for event in non_face_event:
                event_time = event[1]-event[0]
                if event_time<stim_time:
                    continue
                spike_count = len([1 for spike_time in neuron_spikes 
                                if spike_time>=event[0] and spike_time<=event[0]+stim_time])   
                firing_rate = spike_count/event_time             
                psth['non_face'][neuron_idx].append(firing_rate)

    # averaging
    for k in psth:
        for neuron_idx in range(len(psth[k])):
            if len(psth[k][neuron_idx]) == 0:
                psth[k][neuron_idx] = 0
            else:
                psth[k][neuron_idx] = np.mean(psth[k][neuron_idx])
    
    # draw the distribution of firing rate as a histogram plot
    num_categories = len(psth)
    categories = list(psth.keys())

    # Create a figure with 1xN subplots
    plt.close('all')
    fig, axes = plt.subplots(1, num_categories, figsize=(5 * num_categories, 5), sharey=True, sharex=True)
    for i, category in enumerate(categories):
        firing_rates = psth[category]
        # Create a histogram
        axes[i].hist(firing_rates, bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(category)
        axes[i].set_xlabel("Firing Rate")
        axes[i].set_ylabel("Number of Neurons" if i == 0 else "")
    
    fig.savefig(os.path.join(save_path, 'average_histogram.png'), bbox_inches='tight')

    # reorganze the data to plot the differences between category
    plt.close('all')
    psth_diff = {}
    psth_diff['target-baseline'] = [psth['target_face'][idx]-psth['non_face'][idx] 
                                    for idx in range(len(psth['target_face']))]
    psth_diff['non_target-baseline'] = [psth['non_target_face'][idx]-psth['non_face'][idx] 
                                    for idx in range(len(psth['non_target_face']))]
    psth_diff['target-non_target'] = [psth['target_face'][idx]-psth['non_target_face'][idx] 
                                    for idx in range(len(psth['target_face']))]
    num_categories = len(psth_diff)
    categories = list(psth_diff.keys())

    fig, axes = plt.subplots(1, num_categories, figsize=(5 * num_categories, 5), sharey=True)
    for i, category in enumerate(categories):
        firing_rates_diff = psth_diff[category]
        # Create a histogram
        axes[i].hist(firing_rates_diff, bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(category)
        axes[i].set_xlabel("Firing Rate Difference")
        axes[i].set_ylabel("Number of Neurons" if i == 0 else "")
    
    fig.savefig(os.path.join(save_path, 'average_histogram_diff.png'), bbox_inches='tight')

    # plot the fine PSTH by neuron
    os.makedirs(os.path.join(save_path, 'PSTH'), exist_ok=True)
    tick_label = [round((bin_size*idx_-baseline_buffer)*1000) for idx_ in range(num_bin)]
    x_tick = np.linspace(0, len(tick_label)-1, 5, dtype=int)
    tick_label = [tick_label[int(cur)] for cur in np.linspace(0, len(tick_label)-1, 5, dtype=int)]
    bin_interval = (stim_time+baseline_buffer)/num_bin

    # sort the neuron by target-selective-index (TSI)
    avg_psth = []
    tsi_val = []
    for neuron_idx in range(len(psth_fg['target_face'])):
        target_face = np.array(psth_fg['target_face'][neuron_idx]).mean(0)
        non_target_face = np.array(psth_fg['non_target_face'][neuron_idx]).mean(0)    
        avg_psth.append({'target': target_face, 'non_target': non_target_face})   
        tsi_val.append((target_face.mean()-non_target_face.mean())/(target_face.mean()+non_target_face.mean())) 

    sort_idx = np.argsort(tsi_val)[::-1]

    for sort_id, neuron_idx in enumerate(sort_idx):
        plt.close('all')
        target_face = avg_psth[neuron_idx]['target']
        non_target_face = avg_psth[neuron_idx]['non_target']

        fig = plt.figure()
        plt.plot(np.arange(len(target_face)), target_face, 'r-', label='Target')
        plt.plot(np.arange(len(non_target_face)), non_target_face, 'b-', label='Non-target')
        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.ylabel('Firing Rating')
        plt.xlabel('Time')
        plt.xticks(x_tick, tick_label)  # Custom labels
        plt.legend()
        fig.savefig(os.path.join(save_path, 'PSTH', str(sort_id+1)+'_'+str(neuron_idx+1)+'.png'), 
                    bbox_inches='tight')

import numpy as np
from polyface.PolyfaceUtil import get_room_loc, get_face_pos, eye_face_interaction, find_transition, find_closest, find_passage
from polyface.PolyfaceUtil import map_coordinates, compute_spatial_frequency, compute_reward_frequency, compute_face_map
from polyface.PolyfaceUtil import get_face_interaction_data, get_replay_data
from matplotlib import pyplot as plt
import os
from analysis.util import moving_average, compute_population_response, check_eye_interaction
from analysis.decoding import kfold_cross_validation, kfold_cross_validation_regression
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from matplotlib.patches import Circle
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

room_color = {'circle': 'red', 'rectangle': 'aqua',
              'diamond': 'green', 'triangle': 'goldenrod',
              'corridor': 'purple', 
              'circle_rectangle': 'red',
              'circle_diamond': 'aqua',
              'rectangle_circle': 'green',
              'diamond_circle': 'goldenrod',
              'rectangle_diamond': 'purple',
              'diamond_rectangle': 'blue',
              'diamond_triangle': 'dodgerblue',
              'triangle_diamond': 'slategrey'
              }

room_connectivity = {'triangle': ['diamond'],
                     'diamond': ['triangle', 'rectangle', 'circle'],
                     'rectangle': ['circle', 'diamond'],
                     'circle': ['diamond', 'rectangle']
                    }

def PSTH_by_face_polyface(trial_data, trial_info, wall_layout, 
                          tolerance=8, stim_time=0.45, bin_size=0.02,
                            baseline_buffer=0.15, num_smooth=3, stim_start=0.05,
                            stim_end=0.25, drop_invalid=True,
                          cell_type='rML', save_path=None, subset=None, strict=False,
                          fsi_score = None, fsi_thres=0.2, cue_time=0.6):
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
            - stim_start/stim_end: time window to compute the average firing rate (for scatter plot and histogram)
            - baseline_buffer: K ms before event onset as baseline.
            - drop_invalid: dropping the invalid data (shorter than specified stim_time) or not .
            - cell_type: type of cells for consideration
            - save_path: path for saving the histogram plot
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - strict: If yes, only consider the target face responses from the correct trials, and non-target
                    face response from the wrong trials.
            - fsi_score: if not None, only consider cell with fsi score later than a predefined threshold. Note that
                    the neuron index should be pre-aligned
            - fsi_thres: fsi threshold
            - cue_time: time before cue offset for computing the cue firing rate
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
    
    # compute the average firing for cue first
    cue_firing = compute_cue_firing_rate(trial_data, cue_time, stim_start, stim_end, cell_type, subset).mean(0)

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
                    if event[0]-trial_onset <= (stim_time+baseline_buffer): # ignore events that are too close to cue phase
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
                                if spike_time>=event_onset+stim_start and spike_time<=event_onset+stim_end])
                    firing_rate = spike_count/(stim_end-stim_start)

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
                                if spike_time>=event[0]+stim_start and spike_time<=event[0]+stim_end])   
                firing_rate = spike_count/(stim_end-stim_start)             
                psth['non_face'][neuron_idx].append(firing_rate)

    # averaging
    for k in psth:
        for neuron_idx in range(len(psth[k])):
            if len(psth[k][neuron_idx]) == 0:
                psth[k][neuron_idx] = 0
            else:
                psth[k][neuron_idx] = np.mean(psth[k][neuron_idx])

    if fsi_score is not None:
        valid_idx = [idx for idx in range(len(cell_idx)) if fsi_score[idx]>fsi_thres]
    else:
        valid_idx = [idx for idx in range(len(cell_idx))]

    # draw a scatter plot of the average firing rate between target/non-target
    plt.close('all')
    fig = plt.figure()
    x = psth['non_target_face']
    y = psth['target_face']
    if fsi_score is None:
        plt.scatter(x, y, c='gray', marker='d', s=25, alpha=0.2)
    else:
        x_high, y_high = np.array(x)[valid_idx], np.array(y)[valid_idx]
        x_low = np.array(x)[[idx for idx in range(len(cell_idx)) if idx not in valid_idx]]
        y_low = np.array(y)[[idx for idx in range(len(cell_idx)) if idx not in valid_idx]]
        plt.scatter(x_low, y_low, c='gray', marker='d', s=25, alpha=0.2)
        plt.scatter(x_high, y_high, c='black', marker='d', s=25)

    plt.plot(np.linspace(0, np.max(x)), np.linspace(0, np.max(x)), color='gray', linestyle='--')
    fig.set_size_inches(5, 5)
    fig.savefig(os.path.join(save_path, 'scatter_plot.png'), bbox_inches='tight', dpi=400)

    # draw a scatter plot of the average firing rate between target face in arena and cue
    plt.close('all')
    fig = plt.figure()
    x = cue_firing
    y = psth['target_face']    
    if fsi_score is None:
        plt.scatter(x, y, c='gray', marker='d', s=25, alpha=0.2)
    else:
        x_high, y_high = np.array(x)[valid_idx], np.array(y)[valid_idx]
        x_low = np.array(x)[[idx for idx in range(len(cell_idx)) if idx not in valid_idx]]
        y_low = np.array(y)[[idx for idx in range(len(cell_idx)) if idx not in valid_idx]]
        plt.scatter(x_low, y_low, c='gray', marker='d', s=25,alpha=0.2)
        plt.scatter(x_high, y_high, c='black', marker='d', s=25)

    plt.plot(np.linspace(0, np.max(x)), np.linspace(0, np.max(x)), color='gray', linestyle='--')
    fig.set_size_inches(5, 5)
    fig.savefig(os.path.join(save_path, 'scatter_plot_cue.png'), bbox_inches='tight', dpi=400)

    # scatter plot for difference between arena and cue target, vs fsi
    plt.close('all')
    fig = plt.figure()
    task_diff = y-x
    fsi = fsi_score
    plt.scatter(fsi, task_diff, c='black', marker='d', s=25)
    corr = np.corrcoef(fsi, task_diff)[0, 1]
    corr_line = np.polyfit(fsi, task_diff, 1)  # Degree 1 for a line
    m, b = corr_line
    plt.plot(np.linspace(0, np.max(fsi)), m * np.linspace(0, np.max(fsi)) + b, color='gray', linestyle='--',
             label= 'Pearson Correlation: %.2f'%corr)
    fig.set_size_inches(5, 5)
    plt.legend()
    fig.savefig(os.path.join(save_path, 'scatter_plot_task_fsi.png'), bbox_inches='tight', dpi=400)

    # draw the distribution of firing rate as a histogram plot
    num_categories = len(psth)
    categories = list(psth.keys())

    # Create a figure with 1xN subplots
    plt.close('all')
    fig, axes = plt.subplots(1, num_categories, figsize=(5 * num_categories, 5), sharey=True, sharex=True)
    for i, category in enumerate(categories):
        firing_rates = np.array(psth[category])[valid_idx]
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
        firing_rates_diff = np.array(psth_diff[category])[valid_idx]
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
    
    # plot the average psth over all neurons
    plt.close('all')
    if fsi_score is not None:
        target_face_avg = np.array([avg_psth[idx]['target'] for idx in sort_idx if idx in valid_idx])
        distractor_face_avg = np.array([avg_psth[idx]['non_target'] for idx in sort_idx if idx in valid_idx])
    else:
        target_face_avg = np.array([avg_psth[idx]['target'] for idx in sort_idx])
        distractor_face_avg = np.array([avg_psth[idx]['non_target'] for idx in sort_idx])
    target_face_avg = target_face_avg.mean(0)
    distractor_face_avg = distractor_face_avg.mean(0)
    fig = plt.figure()
    plt.plot(np.arange(len(target_face_avg)), target_face_avg, 'r-', label='Target')
    plt.plot(np.arange(len(distractor_face_avg)), distractor_face_avg, 'b-', label='Non-target')
    plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
    plt.ylabel('Firing Rate')
    plt.xlabel('Time')
    plt.xticks(x_tick, tick_label)  # Custom labels
    plt.legend()
    fig.savefig(os.path.join(save_path, 'average_psth.png'), 
                bbox_inches='tight')

    for sort_id, neuron_idx in enumerate(sort_idx):
        if fsi_score is not None and neuron_idx not in valid_idx:
            continue
        plt.close('all')
        target_face = avg_psth[neuron_idx]['target']
        non_target_face = avg_psth[neuron_idx]['non_target']

        fig = plt.figure()
        plt.plot(np.arange(len(target_face)), target_face, 'r-', label='Target')
        plt.plot(np.arange(len(non_target_face)), non_target_face, 'b-', label='Non-target')
        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time')
        plt.xticks(x_tick, tick_label)  # Custom labels
        plt.legend()
        fig.savefig(os.path.join(save_path, 'PSTH', str(sort_id+1)+'_'+str(neuron_idx+1)+'.png'), 
                    bbox_inches='tight')

def PSTH_by_room_polyface(trial_data, trial_info, wall_layout, 
                          tolerance=8, stim_time=0.45, bin_size=0.02,
                          baseline_buffer=0.15, num_smooth=3, cell_type='rEC', 
                          save_path=None, subset=None, filter_eye=False, filter_onset=False,
                          plot_transistion=False, sort_selectivity=False):
    """ Compute the PSTH for the onset of different events, i.e., entering a specific room/corridor. 
        It only considers the spike firing rates for the first
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
            - cell_type: type of cells for consideration
            - save_path: path for saving the histogram plot
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - filter_eye: If yes, only consider the period where the monkey is not looking at a face.
            - filter_onset: If yes, filter out the periods when the monkey enter the room right after stim onset.
            - plot_transistion: If yes, plot the PSTH based on room transition (instead of just the room).
            - sort_selectivity: If yes, compute the selectivity of each neuron and sort them based on the strength 
                            of selectivity. The visualization will be save in separate folders for different room.
    """
    os.makedirs(save_path, exist_ok=True)

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]

    # get the locations of faces in different trials
    face_loc = get_face_pos(trial_info)
    
    # psth averaged over the whole stim onset
    if not plot_transistion:
        psth = {k: [[] for _ in range(len(cell_idx))] for k in room_color if not '_' in k}
    else:
        psth = {k: [[] for _ in range(len(cell_idx))] for k in room_color if '_' in k}

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

        # get the periods for different spatial locations
        room_block = get_room_loc(trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator'][cur_type][trial_idx])
        
        if plot_transistion:
            room_block = find_transition(room_block)
            if len(room_block) == 0:
                continue

        # obtain eye interaction periods
        if filter_eye:
            interaction_block = eye_face_interaction(trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx],
                                                    trial_data['Paradigm']['PolyFaceNavigator']['Eye_arena'][trial_idx],
                                                    face_loc[trial_number], tolerance=tolerance, wall_layout=wall_layout)


        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]

            # measure PSTH for each individual face interaction event
            for room in room_block:
                for event in room_block[room]:
                    # filter onset
                    if filter_onset and event[0]-trial_onset<(stim_time+baseline_buffer):
                        continue

                    event_time = event[1]-event[0]
                    event_onset = event[0]
                    if event_time<stim_time:
                        continue
                    event_offset = event_onset + stim_time                    

                    # ignore periods when he is looking at a face
                    if filter_eye and check_eye_interaction(interaction_block, [event_onset-baseline_buffer, event_offset]):
                        continue

                    # compute the PSTH
                    num_bin = int((stim_time+baseline_buffer)/bin_size)
                    bin_time = np.linspace(event_onset-baseline_buffer, event_offset, num_bin+1)
                    tmp_count = np.zeros(num_bin, )
                    for bin_idx in range(num_bin):
                        tmp_count[bin_idx] = len([1 for spike_time in neuron_spikes 
                                if spike_time>=bin_time[bin_idx] and spike_time<=bin_time[bin_idx+1]])
                        tmp_count[bin_idx] /= bin_size

                    psth[room][neuron_idx].append(moving_average(tmp_count, num_smooth))

    tick_label = [round((bin_size*idx_-baseline_buffer)*1000) for idx_ in range(num_bin)]
    x_tick = np.linspace(0, len(tick_label)-1, 5, dtype=int)
    tick_label = [tick_label[int(cur)] for cur in np.linspace(0, len(tick_label)-1, 5, dtype=int)]
    bin_interval = (stim_time+baseline_buffer)/num_bin

    # get the average firing rate for each neuron
    avg_psth = dict()
    for room in psth:
        avg_psth[room] = []
        for neuron_idx in range(len(psth[room])):
            cur_psth = np.array(psth[room][neuron_idx]).mean(0)
            avg_psth[room].append(cur_psth)

    # plot the average PSTH
    all_neuron_psth = dict()
    for room in psth:
        all_neuron_psth[room] = np.array(avg_psth[room]).mean(0)
    plt.close('all')
    fig = plt.figure()
    for room in all_neuron_psth:
        plt.plot(np.arange(len(all_neuron_psth[room])), all_neuron_psth[room], 
                        color=room_color[room], label=room)

        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time')
        plt.xticks(x_tick, tick_label)  # Custom labels
        plt.legend()
        fig.savefig(os.path.join(save_path, 'average.png'), 
                    bbox_inches='tight')    
        
    
    # sorting neurons based on room selectivity
    if sort_selectivity:
        # compute the selectivity modulation value for each neuron
        all_room = list(avg_psth.keys())
        modulation_value = np.zeros([len(cell_idx), len(all_room)])
        for i in range(len(cell_idx)):
            for j in range(len(all_room)):
                modulation_value[i, j] = np.mean(avg_psth[all_room[j]][i])
        
        room_selectivity = dict()
        for i in range(len(cell_idx)):
            room_selectivity[i] = dict()
            selected_idx = np.argmax(modulation_value[i])
            non_selected_val = np.mean([modulation_value[i, k] for k in range(len(all_room)) if k!=selected_idx])
            room_selectivity[i]['room'] = all_room[selected_idx]
            room_selectivity[i]['modulation_val'] = (np.max(modulation_value[i])-non_selected_val)/(np.max(modulation_value[i])+non_selected_val)

        # sort the neurons selected for each room
        for room in all_room:
            cur_neuron_pool = [idx for idx in range(len(cell_idx)) if room_selectivity[idx]['room']==room]
            cur_modulation_val = [room_selectivity[idx]['modulation_val'] for idx in cur_neuron_pool]
            sort_idx = np.argsort(cur_modulation_val)[::-1]
            for i in range(len(sort_idx)):
                room_selectivity[cur_neuron_pool[sort_idx[i]]]['sort_idx'] = i


    # plot the fine PSTH by neuron
    for neuron_idx in range(len(cell_idx)):
        plt.close('all')
        fig = plt.figure()
        for room in avg_psth:
            plt.plot(np.arange(len(avg_psth[room][neuron_idx])), avg_psth[room][neuron_idx], 
                            color=room_color[room], label=room)

        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time')
        plt.xticks(x_tick, tick_label)  # Custom labels
        plt.legend()
        if not sort_selectivity:
            fig.savefig(os.path.join(save_path, str(neuron_idx+1)+'.png'), 
                        bbox_inches='tight')
        else:
            selected_room = room_selectivity[neuron_idx]['room']
            sort_idx = room_selectivity[neuron_idx]['sort_idx']
            if not os.path.exists(os.path.join(save_path, selected_room)):
                os.makedirs(os.path.join(save_path, selected_room), exist_ok=True)
            fig.savefig(os.path.join(save_path, selected_room, str(sort_idx+1)+'_'+str(neuron_idx+1)+'.png'), 
                        bbox_inches='tight')            
            

def room_decoding_polyface(trial_data, stim_time=1, bin_size=0.05,
                        baseline_buffer=0.2, cell_type='rEC', 
                        classifier='logistic', k_fold=5, save_path=None, 
                        subset=None, force_balanced=False, filter_onset=False,
                        reg_para=1, kernel='linear', degree=3):
    """ Analysis regarding the temporal decoding of spatial locations (e.g.,
        different rooms and corridor. The decoding is based on population responses,
        and assumes linear classifiers). The results are based on a K-fold evaluation.

        Inputs:
            - trial_data: pre-processed trial data.
            - stim_time: time after entering a specific location for consideration.
            - bin_size: bin size (in second) for decoding, i.e., sample period.
            - cell_type: neuron type for collecting the population response.
            - classifier: classifer for decoding, e.g., logistic or svc.
            - k_fold: number of folds for k-fold validation.
            - save_path: path for saving the results.
            - subset: if not None, only consider samples from the subset.
            - force_balanced: if yes, sample a balanced subset for experiment.
            - filter_onset: filter periods right after stim onset.
            - reg_para: regularization parameter for the classifier.
            - kernel: kernel for svm.
            - degree: degree for svm kernel.
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]    

    temporal_population_response = dict()
    num_bin = int((stim_time+baseline_buffer)/bin_size)+1
    for i in range(num_bin):
        temporal_population_response[(i*bin_size)-baseline_buffer] = []
    
    temporal_population_label = []

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue

        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['On'][trial_idx]

        # determine the type of the current trial
        for type_ in ['End_Correct', 'End_Miss', 'End_Wrong']:
            if not np.isnan(trial_data['Paradigm']['PolyFaceNavigator'][type_][trial_idx]):
                cur_type = type_
                break

        
        room_block = get_room_loc(trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator'][cur_type][trial_idx])

        for room in room_block:
            if room == 'corridor':
                continue
            for block in room_block[room]:
                if filter_onset and block[0]-baseline_buffer<=trial_onset:
                    continue
                # filter the cue phase
                if block[1] <= trial_onset:
                    continue
                elif block[0] <= trial_onset:
                    block[0] = trial_onset

                event_time = block[1]-block[0]
                event_onset = block[0]
                if event_time<stim_time:
                    continue
                else:
                    event_offset = event_onset + stim_time

                neuron_spikes = [trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron_idx] 
                                for neuron_idx in cell_idx]

                # storing the label for the current block
                temporal_population_label.append(room)

                # collect the samples from each bin
                bin_time = np.linspace(event_onset-baseline_buffer, event_offset, num_bin+1)
                for bin_idx in range(num_bin):
                    start, end = bin_time[bin_idx], bin_time[bin_idx+1]
                    population_response = compute_population_response(neuron_spikes, start, end)
                    temporal_population_response[(bin_idx*bin_size)-baseline_buffer].append(population_response)

    # perform k-fold evaluation with the collected data
    print('Number of samples %d' %len(temporal_population_label))
    class_count = dict()
    for k in np.unique(temporal_population_label):
        class_count[k] = np.count_nonzero(np.array(temporal_population_label)==k)
        print('Percentage of samples for %s: %.2f' 
              %(k, class_count[k]/len(temporal_population_label)))
    
    # convert labels to indices
    label_mapping = dict()
    for k in np.unique(temporal_population_label):
        label_mapping[k] = len(label_mapping)
    for i in range(len(temporal_population_label)):
        temporal_population_label[i] = label_mapping[temporal_population_label[i]]
    temporal_population_label = np.array(temporal_population_label)
    random_guess = 1/len(label_mapping)

    # sample a balanced subset
    if force_balanced:
        min_num_sample = np.min(list(class_count.values()))
        sample_idx = []
        for room in class_count:
            cur_class_idx = np.where(temporal_population_label==label_mapping[room])[0]
            sample_idx.extend(list(np.random.choice(cur_class_idx, min_num_sample, replace=False)))
        sample_idx = np.array(sample_idx)
        np.random.shuffle(sample_idx)
        temporal_population_label = temporal_population_label[sample_idx]
        for t in temporal_population_response:
            temporal_population_response[t] = [temporal_population_response[t][cur] for cur in sample_idx]

    # perform the k-fold cross-validation for each temporal bin
    # overall decoding performance
    result = dict() 
    
    # framing binary classification for each room/corridor types
    # equal amount of samples for each class are sampled
    result_by_room  = dict() 
    result_by_room_connected = dict()
    result_by_room_disconnected = dict()

    room_idx_data_general = dict()
    room_idx_data_connected = dict()
    room_idx_data_disconnected = dict()
    room_label = dict()

    for room in label_mapping:
        result_by_room[room] = dict()
        result_by_room_connected[room] = dict()
        room_idx_data_connected[room] = dict()
        positive_idx = np.where(temporal_population_label==label_mapping[room])[0]
        
        # sample negative pairs by room connectivity
        negative_idx_general = np.where(temporal_population_label!=label_mapping[room])[0]
        negative_idx_general = np.random.choice(negative_idx_general, size=len(positive_idx), replace=False)
        negative_pool_connected = [label_mapping[cur] for cur in label_mapping if cur in room_connectivity[room]]
        negative_idx_connected = np.array([idx for idx in range(len(temporal_population_label)) 
                    if temporal_population_label[idx] in negative_pool_connected])
        negative_idx_connected = np.random.choice(negative_idx_connected, size=len(positive_idx), replace=False)
        room_idx_data_general[room] = np.concatenate((positive_idx, negative_idx_general), axis=0)
        room_idx_data_connected[room] = np.concatenate((positive_idx, negative_idx_connected), axis=0)

        # Diamond is connected to all other rooms
        if room != 'diamond':
            result_by_room_disconnected[room] = dict()
            room_idx_data_disconnected[room] = dict()
            negative_pool_disconnected = [label_mapping[cur] for cur in label_mapping 
                                    if cur!=room and cur not in room_connectivity[room]]
            negative_idx_disconnected = np.array([idx for idx in range(len(temporal_population_label)) 
                        if temporal_population_label[idx] in negative_pool_disconnected])
            negative_idx_disconnected = np.random.choice(negative_idx_disconnected, size=len(positive_idx), replace=False)
            room_idx_data_disconnected[room] = np.concatenate((positive_idx, negative_idx_disconnected), axis=0)

        room_label[room] = np.array([1]*len(positive_idx) + [0]*len(positive_idx))


    for t in temporal_population_response:
        result[t] = kfold_cross_validation(temporal_population_response[t], 
                                     temporal_population_label,
                                     k_fold, classifier, reg_para, kernel=kernel, degree=degree)
        
        # per-room decoding performance
        for (room_idx_data, cur_decoding_result) in zip(
                [room_idx_data_general, room_idx_data_connected, room_idx_data_disconnected],
                [result_by_room, result_by_room_connected, result_by_room_disconnected]
                ):
            for room in cur_decoding_result:
                tmp_data = [temporal_population_response[t][idx] for idx in room_idx_data[room]]
                cur_decoding_result[room][t] = kfold_cross_validation(tmp_data, 
                                            room_label[room],
                                            k_fold, classifier, reg_para, report_confidence=True, kernel=kernel,
                                            degree=degree)


    # plot the figure for the decoding results
    # sort the temporal bin for plotting
    sorted_bin = sorted(list(temporal_population_response.keys()))
    x_tick = np.linspace(0, len(sorted_bin)-1, 5, dtype=int)
    tick_label = [int(sorted_bin[cur]*1000) for cur in np.linspace(0, len(sorted_bin)-1, 5, dtype=int)]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(np.arange(len(sorted_bin)), [result[t] for t in sorted_bin], 'b-', linewidth=2, 
                label='Overall Accuracy')
    axs[0].axhline(y=random_guess, color='gray', linestyle='--', linewidth=2) 
    axs[0].axvline(x=baseline_buffer/bin_size, color='gray', linestyle='--')
    axs[0].set_title('Temporal decoding of room category')
    axs[0].legend()
    axs[0].set_xticks(x_tick, tick_label) 
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Accuracy')

    result_by_room_pool = [result_by_room, result_by_room_connected, result_by_room_disconnected]
    name = [('General'), '(Connected)', '(Disconnected)']
    for fig_idx in range(1, 4):
        axs[fig_idx].axhline(y=0.5, color='gray', linestyle='--', linewidth=2) 
        for room in result_by_room_pool[fig_idx-1]:
            axs[fig_idx].plot(np.arange(len(sorted_bin)), [result_by_room_pool[fig_idx-1][room][t] for t in sorted_bin], 
                        color=room_color[room], linewidth=2, 
                        label=room)
        axs[fig_idx].set_xticks(x_tick, tick_label) 
        axs[fig_idx].axvline(x=baseline_buffer/bin_size, color='gray', linestyle='--')
        axs[fig_idx].set_title('Per-room decoding '+name[fig_idx-1])
        axs[fig_idx].set_xlabel('Time')
        if fig_idx == 1:
            axs[fig_idx].set_ylabel('Confidence on Correct Label')
            axs[fig_idx].set_ylim(0.45, 0.7)
        else:
            axs[fig_idx].set_ylabel(None)
            axs[fig_idx].sharey(axs[1])

        axs[fig_idx].legend()


    fig.savefig(save_path, bbox_inches='tight')

def room_vector_polyface(trial_data, cell_type='rEC', save_path=None, 
                        subset=None, filter_onset=False):
    """ Analysis regarding average firing rate when navigating within different rooms.

        Inputs:
            - trial_data: pre-processed trial data.
            - cell_type: neuron type for collecting the population response.
            - save_path: path for saving the results.
            - subset: if not None, only consider samples from the subset.
            - filter_onset: filter periods right after stim onset.
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]    

    avg_firing = {k: [] for k in ['circle', 'rectangle', 'diamond', 'triangle']}

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue

        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['On'][trial_idx]

        # determine the type of the current trial
        for type_ in ['End_Correct', 'End_Miss', 'End_Wrong']:
            if not np.isnan(trial_data['Paradigm']['PolyFaceNavigator'][type_][trial_idx]):
                cur_type = type_
                break

        
        room_block = get_room_loc(trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx],
                                    trial_data['Paradigm']['PolyFaceNavigator'][cur_type][trial_idx])


        neuron_spikes = [trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron_idx] 
                        for neuron_idx in cell_idx]

        for room in room_block:
            if room == 'corridor':
                continue
            for block in room_block[room]:
                if filter_onset and block[0]-0.2<=trial_onset:
                    continue
                # filter the cue phase
                if block[1] <= trial_onset:
                    continue

                neuron_spikes = [trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron_idx] 
                                for neuron_idx in cell_idx]
                firing_rate = compute_population_response(neuron_spikes, block[0], block[1])
                avg_firing[room].append(firing_rate)

    for k in avg_firing:
        avg_firing[k] = np.array(avg_firing[k]).mean(0)

    room_pool = ['circle', 'rectangle', 'diamond', 'triangle']
    heatmap = np.zeros([len(room_pool), len(room_pool)])
    for i in range(len(room_pool)):
        for j in range(i, len(room_pool)):
            heatmap[i, j] = cosine_similarity(avg_firing[room_pool[i]].reshape([1, -1]), 
                                              avg_firing[room_pool[j]].reshape(1, -1))[0, 0]
            # heatmap[i, j] = np.linalg.norm(avg_firing[room_pool[i]]-avg_firing[room_pool[j]])
    plt.close('all')
    fig = plt.figure(figsize=(4, 4))
    mask = np.triu(np.ones_like(heatmap, dtype=bool), k=1)
    heatmap = np.where(mask, heatmap, np.nan)  # Set the lower triangle to NaN
    heatmap = (heatmap - np.nanmin(heatmap))/(np.nanmax(heatmap) - np.nanmin(heatmap))

    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    plt.xticks(ticks=range(len(room_pool)), labels=room_pool, rotation=45)
    plt.yticks(ticks=range(len(room_pool)), labels=room_pool)    
    plt.colorbar(label='Normalized cosine similarity')
    fig.savefig(os.path.join(save_path, 'cosine_similarity.png'), bbox_inches='tight')

def room_regressor_polyface(trial_data, wall_layout, stim_start=-0.15, stim_end=0.2, bin_size=0.1, step_size=0.05,
                            cell_type='rEC', k_fold=5, save_path=None, subset=None, reg_para=1,
                            vmin=None, vmax=None, regressor='ridge', kernel='rbf'):
    """ Analysis regarding the temporal decoding of spatial locations. By default, it
    uses a standard linear regression. The results are based on a K-fold evaluation.

        Inputs:
            - trial_data: pre-processed trial data.
            - wall_layout: layout of the room for computing the spatial error map.
            - stim_start/end: time w.r.t. the location "onset" to extract the neural responses.
            - bin_size: bin size to compute the population response
            - step_size: step interval for performing the neural response sweep
            - cell_type: neuron type for collecting the population response.
            - k_fold: number of folds for k-fold validation.
            - save_path: path for saving the results.
            - subset: if not None, only consider samples from the subset.
            - reg_para: regularization parameter for the classifier.
            - vmin/vmax: parameters for the hexagon plot.
            - regressor: model used for decoding
    """
    os.makedirs(save_path, exist_ok=True)

    # hardcode the x-z min/max for normalization
    min_x, max_x = -12, 5
    min_z, max_z = -6.5, 11

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]    

    num_bin = int((stim_end-stim_start-bin_size)//step_size) + 1
    temporal_population_response = [[] for _ in range(num_bin)]
    temporal_label = []

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

        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['On'][trial_idx]
        trial_offset = trial_data['Paradigm']['PolyFaceNavigator'][cur_type][trial_idx]

        # interpolate/reorganize player data with eye data 
        player_data = trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx]
        eye_data = trial_data['Paradigm']['PolyFaceNavigator']['Eye_arena'][trial_idx]
        processed_player = {'time': [], 'position': []}
        for t in eye_data['SyncedTime']:
            player_idx = find_closest(t, player_data['SyncedTime'])
            pos = player_data['Pos'][player_idx]
            processed_player['time'].append(t)
            cur_x = (pos[0] - min_x)/(max_x-min_x)
            cur_z = (pos[2] - min_z)/(max_z-min_z)
            processed_player['position'].append([cur_x, cur_z])

        # collect the neural data when passing through different locations
        # reorganize the neuron responses (may lose tiny precision)
        time_window = int((trial_offset-trial_onset)*1000)
        processed_neuron_spikes = np.zeros([time_window, len(cell_idx)])
        for neuron_idx, neuron in enumerate(cell_idx):
            for cur_spike in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]:
                spike_time = int((cur_spike-trial_onset)*1000)-1
                processed_neuron_spikes[spike_time, neuron_idx] += 1

        # iterate through each valid period/location
        for t_idx, t in enumerate(processed_player['time']):
            if t+stim_start<trial_onset or t+stim_end>trial_offset:
                continue
            
            for bin_idx in range(num_bin):
                cur_start, cur_end = stim_start+bin_idx*step_size, stim_start+bin_idx*step_size + bin_size
                cur_start = int((t+cur_start-trial_onset)*1000)
                cur_end = int((t+cur_end-trial_onset)*1000)
                firing_rate = processed_neuron_spikes[cur_start:cur_end, :].sum(0)/bin_size
                temporal_population_response[bin_idx].append(firing_rate)

            temporal_label.append(processed_player['position'][t_idx])

    temporal_label = np.array(temporal_label)
    temporal_population_response = np.array(temporal_population_response)

    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    array_size = 1008

    # k-fold validation with regression
    for bin_idx in range(num_bin):
        record = kfold_cross_validation_regression(temporal_population_response[bin_idx],
                                                    temporal_label, k_fold, reg_para, regressor,
                                                    kernel)

        # convert normalized position back to original one
        record['gt'][:, 0] = record['gt'][:, 0]*(max_x-min_x) + min_x
        record['gt'][:, 1] = record['gt'][:, 1]*(max_z-min_z) + min_z
        record['pred'][:, 0] = record['pred'][:, 0]*(max_x-min_x) + min_x
        record['pred'][:, 1] = record['pred'][:, 1]*(max_z-min_z) + min_z
        cur_error = np.sqrt(((record['gt']-record['pred'])**2).sum(-1))

        # initialize the error map for each bin
        error_map = [[[] for _ in range(array_size)] for _ in range(array_size)]

        # assign the error to different spatial locations
        for idx in range(len(record['gt'])):
            x, y = record['gt'][idx]
            x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
            error_map[y_mapped][x_mapped].append(cur_error[idx])

        for x in range(array_size):
            for y in range(array_size):
                error_map[y][x] = np.mean(error_map[y][x]) if len(error_map[y][x])>0 else -1

        # visualize the error as a hexbin heatmap
        error_map = np.array(error_map)
        # Parameters
        hex_size = 0.5  # Size of each hexagon

        # Create grid of points
        rows, cols = error_map.shape
        x = np.arange(cols) * hex_size * 3 / 4  # x-coordinates of the hexagon centers
        y = np.arange(rows) * hex_size * np.sqrt(3) / 2  # y-coordinates of the hexagon centers
        xx, yy = np.meshgrid(x, y)

        # Compute average value for each hexagon
        hex_values = []
        for i in range(rows):
            for j in range(cols):
                val = error_map[i, j]
                if val != -1:  # Ignore -1
                    hex_values.append((xx[i, j], yy[i, j], val))
                else:
                    hex_values.append((xx[i, j], yy[i, j], np.nan))  # Mark as NaN

        hex_x, hex_y, hex_avg = zip(*hex_values)
        hex_avg = np.array(hex_avg)
        hex_avg = hex_avg/np.nanmax(hex_avg)

        # Create a custom colormap with white for NaN
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size for larger array
        sc = ax.hexbin(
            hex_x, hex_y, C=hex_avg, gridsize=30, cmap=cmap, mincnt=1,
            linewidths=0.2, edgecolors='gray', reduce_C_function=np.nanmean
        )

        vmin = vmin if vmin is not None else np.nanmin(hex_avg)
        vmax = vmax if vmax is not None else np.nanmax(hex_avg)
        sc.set_clim(vmin=vmin, vmax=vmax)

        # Add colorbar
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Decoding Error")
        cur_start, cur_end = stim_start+bin_idx*bin_size, stim_start+(bin_idx+1)*bin_size
        cur_start, cur_end = round(cur_start*1000), round(cur_end*1000)
        fig.savefig(os.path.join(save_path, 
                                 'spatial_decoding_error_'+str(cur_start)+'_'+str(cur_end)+cell_type+'.png'), 
                    bbox_inches='tight')
        
    return temporal_population_response, temporal_label

def place_field_visualization(trial_data, trial_info, wall_layout, cell_type='rEC', 
                          save_path=None, subset=None, pf_size=10):
    """ Visualizing the place field for each neuron as a heatmap.

        Inputs:
            - trial_data: pre-processed trial data.
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - cell_type: types of cells for computation.
            - save_path: directory for saving the visualization.
            - subset: if not None, only consider trial numbers in the subset.
            - pf_size: size of place field for computing the passage, size in pixel.
    """
    os.makedirs(os.path.join(save_path, 'heatmap'), exist_ok=True)

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]    
    
    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)

    # for heatmap visualization
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    array_size = 1008

    # compute the spatial and reward frequency of different locations for normalization later
    spatial_frequency = compute_spatial_frequency(trial_data, wall_layout, array_size, subset)
    reward_frequency = compute_reward_frequency(trial_data, wall_layout, array_size, subset)
    face_map = compute_face_map(trial_info, wall_layout, array_size, subset)
    
    # initialize the place field for different neurons
    place_fields = np.zeros([len(cell_idx), array_size, array_size])

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue    

        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx]
        player_data = trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx]

        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]

            # find the location for each spike
            for spike in neuron_spikes:
                if spike<=trial_onset:
                    continue
                loc_idx = find_closest(spike, player_data['SyncedTime'])
                x, y = player_data['Pos'][loc_idx][0], player_data['Pos'][loc_idx][2] # actually x-z in Unity
                x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
                place_fields[neuron_idx, y_mapped, x_mapped] += 1

    # generate the heatmaps
    extent = [x_min, x_max, y_min, y_max]
    final_place_field = []
    for neuron_idx, neuron in enumerate(cell_idx):
        plt.close('all')
        fig, ax = plt.subplots()
        # # plot the walls
        # for wall in wall_layout:
        #     x = [wall['startPoint']['x'], wall['endPoint']['x']]
        #     y = [wall['startPoint']['y'], wall['endPoint']['y']]
        #     plt.plot(x, y, c='black')
        # plt.axis('off')
        # fig.set_size_inches(8, 8)

        # normalization (with reward and spatial frequency) and Gaussian blurring
        cur_place_field = place_fields[neuron_idx]
        normalized_place_field = cur_place_field/reward_frequency/(spatial_frequency+1e-7)
        normalized_place_field /= normalized_place_field.max()
        normalized_place_field = gaussian_filter(normalized_place_field, sigma=15)
        final_place_field.append(normalized_place_field)

        # # draw a circle to highlight the place field of the given size
        # pf_y_idx, pf_x_idx = np.unravel_index(normalized_place_field.argmax(), normalized_place_field.shape)
        # overlay = np.zeros((array_size, array_size, 4), dtype=np.uint8)  # RGBA overlay
        # cv2.circle(overlay, (pf_x_idx, pf_y_idx), pf_size, (255, 0, 0, 255), 2)  # Red circle with full opacity

        # im = ax.imshow(normalized_place_field, origin='lower', cmap='viridis', extent=extent, alpha=0.6, interpolation='nearest')
        # ax.imshow(overlay, origin='lower', extent=extent, interpolation='nearest')

        # colormap = plt.cm.viridis
        # colormap.set_under(color='white', alpha=0)

        # visualize place field as hexagon plot
        hex_size = 0.5  # Size of each hexagon
        rows, cols = normalized_place_field.shape
        x = np.arange(cols) * hex_size * 3 / 4  # x-coordinates of the hexagon centers
        y = np.arange(rows) * hex_size * np.sqrt(3) / 2  # y-coordinates of the hexagon centers
        xx, yy = np.meshgrid(x, y)

        # Compute average value for each hexagon
        hex_values = []
        for i in range(rows):
            for j in range(cols):
                val = normalized_place_field[i, j]
                if val != 0:  # Ignore 0
                    hex_values.append((xx[i, j], yy[i, j], val))
                else:
                    hex_values.append((xx[i, j], yy[i, j], np.nan))  # Mark as NaN

        hex_x, hex_y, hex_avg = zip(*hex_values)
        hex_avg = np.array(hex_avg)

        # Create a custom colormap with white for NaN
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size for larger array
        sc = ax.hexbin(
            hex_x, hex_y, C=hex_avg, gridsize=30, cmap=cmap, mincnt=1,
            linewidths=0.2, edgecolors='gray', reduce_C_function=np.nanmean
        )

        vmin = np.nanmin(hex_avg)
        vmax = np.nanmax(hex_avg)
        sc.set_clim(vmin=vmin, vmax=vmax)

        fig.savefig(os.path.join(save_path, 'heatmap', str(neuron)+'.png'), bbox_inches='tight')

    # compute the average correlation between place field and reward/face maps
    avg_reward_corr = []
    avg_face_corr = []
    face_map /= face_map.max()
    face_map = gaussian_filter(face_map, sigma=15)
    reward_frequency -= 1
    reward_frequency /= reward_frequency.max()
    reward_frequency = gaussian_filter(reward_frequency, sigma=15)

    for neuron_idx, neuron in enumerate(cell_idx):
        cur_place_field = place_fields[neuron_idx]/(spatial_frequency+1e-7)
        cur_place_field /= cur_place_field.max()
        cur_place_field = gaussian_filter(cur_place_field, sigma=15)
        avg_reward_corr.append(np.corrcoef(cur_place_field.reshape(-1), reward_frequency.reshape(-1))[0, 1])
        avg_face_corr.append(np.corrcoef(cur_place_field.reshape(-1), face_map.reshape(-1))[0, 1])

    print('Average/Max correlation with reward map: %.3f/%.3f' %(np.mean(avg_reward_corr), np.max(avg_reward_corr)))
    print('Average/Max correlation with face map: %.3f/%.3f' %(np.mean(avg_face_corr), np.max(avg_face_corr)))

    # return the place field of each neuron
    record_place_field = []
    for neuron_idx in range(len(final_place_field)):
        cur_place_field = final_place_field[neuron_idx]
        y, x = np.unravel_index(cur_place_field.argmax(), cur_place_field.shape)
        record_place_field.append([x, y])
    
    return record_place_field

def place_field_psth(trial_data, place_field, wall_layout, cell_type='rEC',
                     stim_time=0.45, bin_size=0.02, baseline_buffer=0.15, num_smooth=3, 
                     save_path=None, subset=None, pf_size=8):
    """ Compute the PSTH and raster for place field passage.

        Inputs:
            - trial_data: pre-processed trial data.
            - place_field: precomputed location of place field for each neuron.
            - wall_layout: wall positions of the polyface environment.
            - cell_type: types of cells for computation.
            - stim_time: time after the stimulus onset to compute the PSTH.
            - bin_size: bin size (in second) to compute the fine-grained PSTH.
            - num_smooth: number of historical bins for smoothing.
            - baseline_buffer: K ms before event onset as baseline.
            - save_path: directory for saving the visualization.
            - subset: if not None, only consider trial numbers in the subset.
            - pf_size: size of place field for computing the passage, size in pixel.
    """
    os.makedirs(os.path.join(save_path, 'psth'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'raster'), exist_ok=True)

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]    
    
    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)

    # for heatmap visualization
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    array_size = 1008

    psth = [[] for _ in range(len(cell_idx))]
    raster = [[] for _ in range(len(cell_idx))] 

    # iterate through all trials and gather the spike data during passage
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue    

        trial_onset = trial_data['Paradigm']['PolyFaceNavigator']['Start'][trial_idx]
        player_data = trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx]

        for neuron_idx, neuron in enumerate(cell_idx):
            # check if the current trial pass through the place field
            passage_block = find_passage(player_data, place_field[neuron_idx], pf_size,
                                             x_min, x_max, y_min, y_max, array_size) 
            
            if len(passage_block) == 0:
                continue
            
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]
            
            for passage in passage_block:
                # each passage is organized as:
                # start time for entering the "big place field"
                # end time for leaving the "big place field"
                # time closest to the center of the place field
                event_onset = passage[-1]
                event_offset = passage[-1] + stim_time
                # ignore events that are too close to cue phase
                if event_onset-trial_onset <= (stim_time+baseline_buffer): 
                    continue

                # raster for the current passage
                spike = [(spike_time-event_onset)*1000 for spike_time in neuron_spikes 
                        if spike_time>=event_onset-baseline_buffer and spike_time<=event_offset]
                raster[neuron_idx].append(spike)

                # compute the psth
                num_bin = int((stim_time+baseline_buffer)/bin_size)
                bin_time = np.linspace(event_onset-baseline_buffer, event_offset, num_bin+1)
                tmp_count = np.zeros(num_bin, )
                for bin_idx in range(num_bin):
                    tmp_count[bin_idx] = len([1 for spike_time in neuron_spikes 
                            if spike_time>=bin_time[bin_idx] and spike_time<=bin_time[bin_idx+1]])
                    tmp_count[bin_idx] /= bin_size
                
                psth[neuron_idx].append(moving_average(tmp_count, num_smooth))

    # average psth
    bin_interval = (stim_time+baseline_buffer)/num_bin
    tick_label = [round((bin_size*idx_-baseline_buffer)*1000) for idx_ in range(num_bin)]
    x_tick = np.linspace(0, len(tick_label)-1, 5, dtype=int)
    tick_label = [tick_label[int(cur)] for cur in np.linspace(0, len(tick_label)-1, 5, dtype=int)]

    for neuron_idx in range(len(psth)):
        if len(psth[neuron_idx])==0:
            continue

        plt.close('all')
        fig = plt.figure()
        avg_psth = np.array(psth[neuron_idx]).mean(0)
        plt.plot(np.arange(len(avg_psth)), avg_psth, 'b-')
        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.ylabel('Firing Rate')
        plt.xlabel('Time')
        plt.xticks(x_tick, tick_label)  # Custom labels
        fig.savefig(os.path.join(save_path, 'psth', str(cell_idx[neuron_idx])+'.png'), 
                    bbox_inches='tight')        

    # raster plot for each neuron
    for neuron_idx in range(len(cell_idx)):
        if len(raster[neuron_idx])==0:
            continue
        plt.close('all')
        fig = plt.figure()            
        ax = plt.gca()
        ax.set_ylim(0, len(raster[neuron_idx])+2)
        for i in range(len(raster[neuron_idx])):
            ax.scatter(raster[neuron_idx][i], [i+1]*len(raster[neuron_idx][i]), c='black', marker='.', s=0.8)
        plt.axvline(x=baseline_buffer/bin_interval, color='gray', linestyle='--')
        plt.xlabel('Time')
        plt.xticks(tick_label)
        plt.yticks([])

        fig.savefig(os.path.join(save_path, 'raster', str(cell_idx[neuron_idx])+'.png'), bbox_inches='tight')


def compute_cue_firing_rate(trial_data, cue_time=0.8, stim_start=0.05, stim_end=0.25,
                          cell_type='rML', subset=None, face2id=None,
                          trial_info=None
                          ):
    """ Computer the average firing rate during the cue phase.

        Inputs:
            - trial_data: pre-processed trial data.
            - cue_time: time for triggering the cue fixation event.
            - stim_start/end: time to compute the average response
            - cell_type: type of cells for consideration
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - face2id: if not None, also compute the face identity for the cue
            - trial_info: for extracting face identity
        
        Returns:
            Average firing rate for each neuron , (and face identity)
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]   

    avg_firing = []
    if face2id is not None:
        cue_label = []
        face_loc = get_face_pos(trial_info)
        id2face = {face2id[k]: k for k in face2id}

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue

        if face2id is not None:
            cue_face = [k for k in face_loc[trial_number] if face_loc[trial_number][k]['isTarget']][0]
            cue_face = id2face[cue_face].replace('target_', '').split('_')[0]
            cue_label.append(cue_face)
        
        trial_offset = trial_data['Paradigm']['PolyFaceNavigator']['Off'][trial_idx]

        tmp_firing = []
        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]
            spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time>=trial_offset-cue_time+stim_start and spike_time<=trial_offset-cue_time+stim_end])
            firing_rate = spike_count/(stim_end-stim_start)
            tmp_firing.append(firing_rate)
        
        avg_firing.append(tmp_firing)
    
    avg_firing = np.array(avg_firing)

    if face2id is None:
        return avg_firing
    else:
        return avg_firing, cue_label

def face_replay_decoding(trial_data, trial_info, wall_layout, face2id, 
                          tolerance=8, bin_size=0.1, step_size=0.05, stim_start=0.05,
                          stim_end=0.4, drop_invalid=True,
                          cell_type='rML', replay_thres=0.7, save_path=None, subset=None,
                          fsi_score = None, fsi_thres=0.2, face_response=None, top_k=10,
                          classifier='logistic', reg_para=1, kernel='linear', degree=3,
                          replay_response=None, replay_target=None, num_repeat=10):
    """ Face decoding for the neural responses when he is potentially replaying.

    Inputs:
            - trial_data: pre-processed trial data.
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - face2id: face2id mapping
            - tolerance: tolerance angle between eye/object ray (to be changed to distance-based measurement).
            - stim_start/end: start/end time to extract the neural responses, aligned to onset.
            - bin_size: breaking the neural responses into bins for temporal analysis
            - drop_invalid: dropping the invalid data (shorter than specified stim_time) or not .
            - cell_type: type of cells for consideration
            - save_path: a pickle file to save the extracted data.
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - fsi_score: if not None, only consider the top face selective cell.
            - fsi_thres: threshold for face cell selection.
            - face_response: if not None, use the preloaded face_response
            - top_k: select the top-k target/distractor faces for decoding
            - classifier: model for decoding
            - reg_para: model parameter
            - kernel: kernel for svm.
            - degree: degree for poly kernel
            - num_repeat: repeat for random seeding
    """
    # compute the face interaction response from scratch
    if face_response is None:
        face_response = get_face_interaction_data(trial_data, trial_info, wall_layout, 
                        face2id, tolerance=tolerance, stim_start=stim_start, stim_end=stim_end, 
                        bin_size=bin_size, step_size=step_size,
                        subset=subset, cell_type=cell_type, fsi_score=fsi_score, fsi_thres=fsi_thres)

    # get the potentially replaying response
    all_face_firing = []
    for k in face_response:
        all_face_firing.extend(face_response[k])
    all_face_firing = np.array(all_face_firing)
    filter_rate = np.mean(all_face_firing)*replay_thres

    if replay_response is None:
        replay_response, replay_target = get_replay_data(trial_data, trial_info, wall_layout, 
                            face2id, filter_rate, tolerance=tolerance, stim_start=stim_start, stim_end=stim_end, 
                            bin_size=bin_size, step_size=step_size,
                            subset=subset, cell_type=cell_type, fsi_score=fsi_score, fsi_thres=fsi_thres)
        replay_target = [cur.replace('target_', '').split('_')[0] for cur in replay_target]

    # select the top-k faces
    face_label = dict()
    for k in face2id:
        if 'target' in k:
            face_label[k.replace('target_', '').split('_')[0]] = 1
        else:
            face_label[k.replace('target_', '').split('_')[0]] = 0
            
    target_face = {k: len(face_response[k]) for k in face_response if face_label[k]==1}
    distractor_face = {k: len(face_response[k]) for k in face_response if face_label[k]==0}
    sort_target = {k: v for k, v in sorted(target_face.items(), key=lambda item: item[1], reverse=True)}  
    sort_distractor = {k: v for k, v in sorted(distractor_face.items(), key=lambda item: item[1], reverse=True)}     
    target_face = [k for k in list(sort_target.keys())[:top_k]]
    distractor_face = [k for k in list(sort_distractor.keys())[:top_k]]

    overall_trial_acc, overall_binary_acc, overall_val_binary_acc = [], [], []
    for repeat_idx in range(num_repeat):
        label2cls = dict()
        train_feat = [] 
        train_label = []
        face_pool = dict()
        min_number = min([len(face_response[k]) for k in target_face+distractor_face])
        for k in target_face:
            label2cls[k] = len(label2cls)
            face_pool[k] = 1
            sample_idx = np.random.choice(len(face_response[k]), min_number, replace=False)
            train_feat.extend([face_response[k][idx] for idx in sample_idx])
            train_label.extend([label2cls[k]]*min_number)
        for k in distractor_face:
            label2cls[k] = len(label2cls)
            face_pool[k] = 0
            sample_idx = np.random.choice(len(face_response[k]), min_number, replace=False)
            train_feat.extend([face_response[k][idx] for idx in sample_idx])
            train_label.extend([label2cls[k]]*min_number)
        train_feat = np.array(train_feat)
        train_label = np.array(train_label)
        cls2label = {label2cls[k]: k for k in label2cls}

        train_label_binary = np.array([face_pool[cls2label[label]] for label in train_label])
        train_feat_binary, val_feat_binary, train_label_binary, val_label_binary = train_test_split(train_feat, train_label_binary,
                                                                        test_size=0.2, stratify=train_label_binary)

        train_feat, val_feat, train_label, val_label = train_test_split(train_feat, train_label,
                                                                        test_size=0.2, stratify=train_label)

        # train the classifier
        # decoding without bining
        if step_size is None or bin_size is None:
            if classifier == 'logistic':
                cls = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
                cls_binary = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat_binary, train_label_binary)
            elif classifier == 'svc':
                if kernel == 'linear':
                    cls = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
                    cls_binary = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat_binary, train_label_binary)
                else:
                    cls = SVC(kernel=kernel, C=reg_para, degree=degree, max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
                    cls_binary = SVC(kernel=kernel, C=reg_para, degree=degree, max_iter=3000, class_weight='balanced').fit(train_feat_binary, train_label_binary)
            else:
                raise NotImplementedError(
                    "Support for the selected classifier is not implemented yet")
            
            overall_val_binary_acc.append(cls_binary.score(val_feat_binary, val_label_binary))
            pred_label = cls.predict(replay_response)
            pred_label = [cls2label[cur] for cur in pred_label]

            # compute the binary classification rate
            binary_cls = np.mean(cls_binary.predict(replay_response[:, bin_idx]))
            overall_binary_acc.append(binary_cls)

            # compute target face (for the current trial) matching accuracy
            trial_face_acc = (np.array(pred_label) == np.array(replay_target)).mean()
            overall_trial_acc.append(trial_face_acc)
        else:
            n_sample, n_bin, n_neuron = train_feat.shape
            binary_cls = []
            trial_face_acc = []
            val_binary_cls = []
            for bin_idx in range(n_bin):
                if classifier == 'logistic':
                    cls = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat[:, bin_idx], train_label)
                    cls_binary = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat_binary[:, bin_idx], train_label_binary)
                elif classifier == 'svc':
                    if kernel == 'linear':
                        cls = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat[:, bin_idx], train_label)
                        cls_binary = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat_binary[:, bin_idx], train_label_binary)
                    else:
                        cls = SVC(kernel=kernel, C=reg_para, degree=degree, max_iter=3000, class_weight='balanced').fit(train_feat[:, bin_idx], train_label)
                        cls_binary = SVC(kernel=kernel, C=reg_para, degree=degree, max_iter=3000, class_weight='balanced').fit(train_feat_binary[:, bin_idx], train_label_binary)
                else:
                    raise NotImplementedError(
                        "Support for the selected classifier is not implemented yet")
                
                pred_label = cls.predict(replay_response[:, bin_idx])
                pred_label = [cls2label[cur] for cur in pred_label]
                binary_cls.append(np.mean(cls_binary.predict(replay_response[:, bin_idx])))
                trial_face_acc.append((np.array(pred_label) == np.array(replay_target)).mean())
                val_binary_cls.append(cls_binary.score(val_feat_binary[:, bin_idx], val_label_binary))
            
            overall_binary_acc.append(binary_cls)
            overall_trial_acc.append(trial_face_acc)
            overall_val_binary_acc.append(val_binary_cls)


    if step_size is None or bin_size is None:
        overall_val_binary_acc = np.mean(overall_val_binary_acc)
        overall_binary_acc = np.mean(overall_binary_acc)
        overall_trial_acc = np.mean(overall_trial_acc)       
        print('Validation Accuracy for binary classification %.2f' %overall_val_binary_acc)
        print('Average rate for choosing target faces %.2f' %overall_binary_acc)
        print('Accuracy for replaying the correct face for the current trial %.2f' %(overall_trial_acc))
    else:
        overall_val_binary_acc = np.array(overall_val_binary_acc).mean(0)
        overall_binary_acc = np.array(overall_binary_acc).mean(0)
        overall_trial_acc = np.array(overall_trial_acc).mean(0)
        print('Validation Accuracy:')
        print(overall_val_binary_acc) 

        plt.close('all')
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True)
        axes[0].plot(np.arange(1, n_bin+1), overall_binary_acc, c='gray')
        axes[0].set_xticks(np.arange(1, n_bin+1))
        axes[0].set_xticklabels([round((stim_start+step_size*bin_idx)*1000) for bin_idx in range(n_bin)])
        axes[0].set_title('Target vs Distractor')
        axes[1].plot(np.arange(1, n_bin+1), overall_trial_acc, c='gray')
        axes[1].set_title('Classification of the current target face')
        fig.savefig(os.path.join(save_path, 'replaying_decoding.png'), bbox_inches='tight', dpi=400)

def face_cue_decoding(trial_data, cue_time=0.8, stim_start=0.05, stim_end=0.25,
                          cell_type='rML', subset=None, face2id=None,
                          trial_info=None, classifier='svc', kernel='linear', degree=3,
                          reg_para=1, forced_balanced=False, top_k=None, k_fold=8):
    """ Decoding the identity of cued faces.

        Inputs:
            - trial_data: pre-processed trial data.
            - cue_time: time to trigger the cue fixation events
            - stim_start/end: time to take the average firing rate
            - cell_type: type of cells for consideration
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - face2id: if not None, also compute the face identity for the cue
            - trial_info: for extracting face identity
            - classifier: classifier to be used
            - kernel/degree: svm parameters
            - reg_para: regularization parameter
            - forced_balanced: force balanced class labels.
            - top_k: only select the top-k face 
    """

    cue_firing, cue_label = compute_cue_firing_rate(trial_data, cue_time, stim_start, stim_end, cell_type, 
                                                    subset, face2id, trial_info)

    if top_k is not None:
        valid_label, count = np.unique(cue_label, return_counts=True)
        valid_label = valid_label[np.argsort(count)[-top_k:]]
        cue_firing = [cue_firing[i] for i in range(len(cue_label)) if cue_label[i] in valid_label]
        cue_label = [cue_label[i] for i in range(len(cue_label)) if cue_label[i] in valid_label]

    label2cls = dict()
    for k in np.unique(cue_label):
        label2cls[k] = len(label2cls)
    cue_label = [label2cls[k] for k in cue_label]
    
    if forced_balanced:
        class_data = dict()
        for i in range(len(cue_label)):
            if cue_label[i] not in class_data:
                class_data[cue_label[i]] = []
            class_data[cue_label[i]].append(cue_firing[i])
        
        min_sample = min([len(class_data[k]) for k in class_data])
        
        feat, label = [], []
        for k in class_data:
            select_idx = np.random.choice(np.arange(len(class_data[k])), min_sample, replace=False)
            feat.extend([class_data[k][idx] for idx in select_idx])
            label.extend([k]*min_sample)
        feat = np.array(feat)
        label = np.array(label)
    else:
        feat = np.array(cue_firing)
        label = np.array(cue_label)
    
    """
    train_feat, val_feat, train_label, val_label = train_test_split(feat, label,
                                                                    test_size=0.2, stratify=label)
    
    if classifier == 'logistic':
        cls = LogisticRegression(C=reg_para, max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
    elif classifier == 'svc':
        if kernel == 'linear':
            cls = LinearSVC(C=reg_para, dual='auto', max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
        else:
            cls = SVC(kernel=kernel, C=reg_para, degree=degree, max_iter=3000, class_weight='balanced').fit(train_feat, train_label)
    else:
        raise NotImplementedError(
            "Support for the selected classifier is not implemented yet")
    print('Decoding Accuracy is %.2f (baseline: %.2f)' %(cls.score(val_feat, val_label), 1/len(label2cls)))

    """

    acc = kfold_cross_validation(feat, label, k_fold, classifier, reg_para, 
                                 kernel=kernel, degree=degree)
    print('Decoding Accuracy is %.2f (baseline: %.2f)' %(acc, 1/len(label2cls)))

        

            

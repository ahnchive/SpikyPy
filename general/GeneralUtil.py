import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import re
import json
from minos.MinosData import MinosData
from scipy.io import loadmat
from polyface.PolyfaceUtil import get_room_loc, get_face_pos, eye_face_interaction, process_trial, align_trial, find_closest

# predefine color for different rooms, corridor has white color
room_color = {'circle': 'red', 'rectangle': 'aqua',
              'diamond': 'green', 'triangle': 'yellow',
              'corridor': 'purple'
              }

def cropped_spike_period(spike_data, start, end):
    spike_data = [np.array(unit) for unit in spike_data]  # Convert to NumPy arrays
    return [unit[(unit > start) & (unit < end)].tolist() for unit in spike_data]

def MatStruct2Dict(struct):
    """ Converting Matlab struct data into dictionary in Python.
    """
    processed_dict = dict()
    for k in struct[0]:
        processed_dict[k] = []
    for idx in range(len(struct)):
        for k in struct[idx]:
            processed_dict[k].append(struct[idx][k])

    return processed_dict

def MinosMatlabWrapper(minos_dir, tmp_dir, ephys_offset_before=0, ephys_offset_after=0, eye_offset=False):
    """ A temporarily super-script for reading Minos data
    from temporary Matlab files, and merging them into one dictionary.

    Input:
        minos_dir: Directory storing raw Minos/Ephys files.
        tmp_dir: Directory storing temporary files, i.e., 
                some raw files can not be directly loaded by Python and require
                manual conversion with Matlab.
        ephys_offset_before/after: include offset before and after each trial as baseline
        eye_offset: temporary option to also save the eye data for offset periods.
    """
    session_name = os.path.basename(tmp_dir)

    # read preprocessed files for trial, eye tracking, and player
    trial_data = loadmat(os.path.join(tmp_dir, 'trial.mat'), simplify_cells=True)['trial']
    # convert the struct data to dictionary
    processed_trial = dict()
    for paradigm in trial_data:
        processed_trial[paradigm] = MatStruct2Dict(trial_data[paradigm]['Data'])
    eye_data = loadmat(os.path.join(tmp_dir, 'eye.mat'), simplify_cells=True)['eye']
    processed_eye = MatStruct2Dict(eye_data['Data'])
    processed_eye['SyncedTime'] = eye_data['T_']

    # for polyface only
    if os.path.exists(os.path.join(tmp_dir, 'player.mat')):
        player_data = loadmat(os.path.join(tmp_dir, 'player.mat'), simplify_cells=True)['player']
        processed_player = MatStruct2Dict(player_data['Data'])
        processed_player['SyncedTime'] = player_data['T_']

    # load the ephys data
    ephys_data = loadmat(os.path.join(minos_dir, 
                        session_name+'.spiky.ephys.SpikeInfo.mat'), simplify_cells=True)['data']['Value']
    spike_data = [ephys_data['Spikes']['Value'][unit]['T_'] for unit in range(len(ephys_data['Spikes']['Value']))]
    neuron_type = json.load(open(os.path.join(tmp_dir, 'neuron_type.json')))

    # merge the behavioral data and neural data
    # since the preprocessed files does not have timestamps for each phase, use SpikyPy instead
    for paradigm in trial_data:
        # TODO: use the recorded data for syncing instead
        tmp_trial_data = MinosData(os.path.join(minos_dir, 'Minos', ' '.join(re.split(r'(?<!^)(?=[A-Z])', paradigm)), 
                                'Trials.bin')).Values
        if paradigm != 'PolyFaceNavigator':
            tmp_trial_data = process_trial(tmp_trial_data, filtered_start='Start_Align', filtered_end='End')
            processed_trial[paradigm]['Eye'] = []
        else:   
            tmp_trial_data = process_trial(tmp_trial_data, filtered_start='Start', filtered_end='End')
            processed_trial[paradigm]['Player'] = [] 
            processed_trial[paradigm]['Eye_cue'] = []
            processed_trial[paradigm]['Eye_arena'] = []
            processed_trial[paradigm]['isContinuous'] = []
        
        processed_trial[paradigm]['Spike'] = []

        prev_correct = False
        for idx in range(len(processed_trial[paradigm]['Number'])):
            trial_num = processed_trial[paradigm]['Number'][idx]
            if paradigm != 'PolyFaceNavigator':
                # align eye data
                start_idx, end_idx = align_trial(trial_num, processed_eye, tmp_trial_data, 'Start_Align', 'End')

                if eye_offset:
                    adjusted_start_time = processed_eye['SyncedTime'][start_idx]-ephys_offset_before
                    start_idx = find_closest(adjusted_start_time, processed_eye['SyncedTime'])
                    adjusted_end_time = processed_eye['SyncedTime'][start_idx]+ephys_offset_after
                    end_idx = find_closest(adjusted_end_time, processed_eye['SyncedTime'])

                aligned_eye = {k: processed_eye[k][start_idx:end_idx] for k in processed_eye}
                processed_trial[paradigm]['Eye'].append(aligned_eye)
            else:
                # align eye data during cue phase
                start_idx, end_idx = align_trial(trial_num, processed_eye, tmp_trial_data, 'On', 'Off')
                aligned_eye_cue = {k: processed_eye[k][start_idx:end_idx] for k in processed_eye}

                # align eye data during navigation phase
                start_idx, end_idx = align_trial(trial_num, processed_eye, tmp_trial_data, 'Start', 'End')
                aligned_eye_arena = {k: processed_eye[k][start_idx:end_idx] for k in processed_eye}

                # align player data
                start_idx, end_idx = align_trial(trial_num, processed_player, tmp_trial_data, 'Start', 'End')
                aligned_player = {k: processed_player[k][start_idx:end_idx] for k in processed_player}

                processed_trial[paradigm]['Eye_cue'].append(aligned_eye_cue)
                processed_trial[paradigm]['Eye_arena'].append(aligned_eye_arena)
                processed_trial[paradigm]['Player'].append(aligned_player)

            # use the synced time from the preprocessed data for merging spike data
            if paradigm != 'PolyFaceNavigator':
                start_flag, end_flag = 'Start_Align', 'End'
            else:
                start_flag = 'On'
                for flag in ['End_Correct', 'End_Miss', 'End_Wrong']:
                    if not np.isnan(processed_trial[paradigm][flag][idx]):
                        end_flag = flag 
                        break
                
                # continuous navigation trial or not?
                if prev_correct:
                    processed_trial[paradigm]['isContinuous'].append(1)
                else:
                    processed_trial[paradigm]['isContinuous'].append(0)
                
                prev_correct = True if end_flag == 'End_Correct' else False

            start_time, end_time = processed_trial[paradigm][start_flag][idx], processed_trial[paradigm][end_flag][idx]
            start_time -= ephys_offset_before
            end_time += ephys_offset_after
            trial_spike = cropped_spike_period(spike_data, start_time, end_time)
            processed_trial[paradigm]['Spike'].append(trial_spike)
        
    
    final_data = {'Neuron_type': neuron_type, 'Paradigm': processed_trial}

    return final_data

def plot_raster(trial_data, paradigm, save_dir, save_format='pdf', trial_info=None,
                eye_interaction=False, spatial_overlay=False,
                continuous_only=False, tolerance=15, wall_layout=None):
    """ Visualizing the raster plot for a single trial. The
    plot is divided into different blocks based on a set of discrete events.
    It also supports using background color to incorporate eye interactions
    within the visual scene (e.g., for Polyface).

    Inputs:
        trial_data: preprocessed dictionary.
        paradigm: paradigm for visualization.
        save_format: format for saving the visualization.
        trial_info: trial info read by MinosData, for eye interaction.
        eye_interaction: adding eye interaction data or not.
        spatial_overlay: add spatial position as background color
        save_dir: directory for the saved visualization.
        continuous_only: only consider continuous trials (Polyface only).
        tolerance: tolerance for matching eye ray and ray cast toward the object.
        wall_layout: wall positions of the polyface environment.
    """

    if paradigm == 'PolyFaceNavigator':
        # discrete event for creating the blocks
        discrete_events = ['On','Off', 'Start', 'End']
        # types of trials based on behaviors
        trial_type = ['End_Correct', 'End_Miss', 'End_Wrong']
    else:
        # discrete event for creating the blocks
        discrete_events = ['Start_Align','End']
        # types of trials based on behaviors
        trial_type = ['End_Correct', 'End_Miss']
    
    for type_ in trial_type:
        os.makedirs(os.path.join(save_dir, type_), exist_ok=True)

    if eye_interaction:
        assert trial_info is not None, 'Eye interaction enabled but trial info not given'
        face_loc = get_face_pos(trial_info)

    # one plot for each trial
    for trial_idx in range(len(trial_data['Paradigm'][paradigm]['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        # skip easy trials 
        if continuous_only and not trial_data['Paradigm'][paradigm]['isContinuous']:
            continue

        trial_number = trial_data['Paradigm'][paradigm]['Number'][trial_idx]

        # determine the type of the current trial
        for type_ in trial_type:
            if not np.isnan(trial_data['Paradigm'][paradigm][type_][trial_idx]):
                cur_type = type_
                break
        
        offset = 0.2 # 200ms offset of visualization
        trial_onset = trial_data['Paradigm'][paradigm]['On'][trial_idx] if paradigm == 'PolyFaceNavigator' else trial_data['Paradigm'][paradigm]['Start_Align'][trial_idx]
        trial_len = trial_data['Paradigm'][paradigm][cur_type][trial_idx] - trial_onset 

        # basic raster plot
        plt.close('all')
        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim(0, len(trial_data['Neuron_type'])+10)

        for neuron in range(len(trial_data['Paradigm'][paradigm]['Spike'][trial_idx])):
            spike_time = [cur-trial_onset for cur in trial_data['Paradigm'][paradigm]['Spike'][trial_idx][neuron]]
            ax.scatter(spike_time, [neuron+5]*len(spike_time), c='black', marker='.', s=0.2)
        
        for event in discrete_events:
            if paradigm == 'PolyFaceNavigator' and event == 'End':
                event = cur_type
            ax.axvline(x = trial_data['Paradigm'][paradigm][event][trial_idx]-trial_onset, color = 'gray', linewidth=2, alpha=1)

        # add spatial background
        if spatial_overlay:
            room_block = get_room_loc(trial_data['Paradigm'][paradigm]['Player'][trial_idx],
                                      trial_data['Paradigm'][paradigm]['Start'][trial_idx],
                                      trial_data['Paradigm'][paradigm][cur_type][trial_idx])

            for room in room_block:
                for block in room_block[room]:
                    # filter the cue phase
                    if block[1] <= trial_data['Paradigm'][paradigm]['Off'][trial_idx]:
                        continue
                    elif block[0] <= trial_data['Paradigm'][paradigm]['Off'][trial_idx]:
                        block[0] = trial_data['Paradigm'][paradigm]['Off'][trial_idx]

                    ax.axvspan(block[0]-trial_onset, block[1]-trial_onset, facecolor=room_color[room], alpha=0.2)

        # add eye-face interaction as stripes below the x-axis
        if eye_interaction:
            interaction_block = eye_face_interaction(trial_data['Paradigm'][paradigm]['Player'][trial_idx],
                                                    trial_data['Paradigm'][paradigm]['Eye_arena'][trial_idx],
                                                    face_loc[trial_number], tolerance=tolerance, wall_layout=wall_layout)
            
            for face in interaction_block:
                color = 'red' if face_loc[trial_number][face]['isTarget'] else 'blue'
                for block in interaction_block[face]:
                    rect = patches.Rectangle((block[0]-trial_onset, 0), block[1]-block[0], 3, 
                                                color=color, alpha=1)   
                    ax.add_patch(rect)         

        # save the plot
        fig.set_size_inches(10, 10)
        fig.savefig(os.path.join(save_dir, type_, str(trial_number)+'.'+save_format), bbox_inches='tight')

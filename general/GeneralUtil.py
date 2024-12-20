import numpy as np
from matplotlib import pyplot as plt
import os
import re
import json
from minos.MinosData import MinosData
from scipy.io import loadmat
from polyface.PolyfaceUtil import process_trial, align_trial, find_closest

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
    
    if os.path.exists(os.path.join(tmp_dir, 'reward.mat')):
        reward_data = loadmat(os.path.join(tmp_dir, 'reward.mat'), simplify_cells=True)['reward']
        processed_reward = MatStruct2Dict(reward_data['Data'])
        processed_reward['SyncedTime'] = reward_data['T_']

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
            processed_trial[paradigm]['Reward'] = []

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
                    adjusted_end_time = processed_eye['SyncedTime'][end_idx]+ephys_offset_after
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

                # align reward
                start_time, end_time = aligned_eye_arena['SyncedTime'][0], aligned_eye_arena['SyncedTime'][-1]
                aligned_reward = [cur for cur in processed_reward['SyncedTime'] if cur >=start_time and cur<=end_time]

                processed_trial[paradigm]['Eye_cue'].append(aligned_eye_cue)
                processed_trial[paradigm]['Eye_arena'].append(aligned_eye_arena)
                processed_trial[paradigm]['Player'].append(aligned_player)
                processed_trial[paradigm]['Reward'].append(aligned_reward)

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

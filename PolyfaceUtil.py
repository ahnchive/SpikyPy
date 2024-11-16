import numpy as np
import math
from MinosData import MinosData
from scipy.io import loadmat
import os
import json
import re
import time

def find_closest(number, num_list):
    diff = np.abs(number-num_list)
    return np.argmin(diff)

def tand(degrees):
    radians = math.radians(degrees)
    return math.tan(radians)

def cropped_spike_period(spike_data, start, end):
    spike_data = [np.array(unit) for unit in spike_data]  # Convert to NumPy arrays
    return [unit[(unit > start) & (unit < end)].tolist() for unit in spike_data]


def align_trial(trial_num, behavior_data, trial_data, start_event='Start', end_event='End_Correct'):
    """ Function that aligns the start and end time of trial data
        from three files.
        
        Inputs:
            trial_num: internal trial id for Minos.
            behavior_data: player/eye data file.
            trial_data: trial data file.
            start_event: event indicating the start of the stage
            end_event: event indicating the end of the stage
        
        Return:
            [start_idx, end_idx]: corresponding start and end indices 
            in the player/eye data for the current trial.
    """
    cur_trial = trial_data[trial_num]
    start_time = cur_trial[start_event]
    if end_event == 'End' and end_event not in cur_trial:
        end_event = [cur for cur in ['End_Correct', 'End_Miss', 'End_Wrong'] if cur in cur_trial][0]
    end_time = cur_trial[end_event]
    start_idx = find_closest(start_time, behavior_data['Timestamp'])
    end_idx = find_closest(end_time, behavior_data['Timestamp'])
    assert start_idx<end_idx, "Misaligned trial detected"
    return start_idx, end_idx

def process_trial(trial_data, filtered_start='Start', filtered_end='End_Correct'):
    """ Function for processing the trial data into
        a more organized manner.
    """
    processed_trial = dict()
    for i in range(len(trial_data['Number'])):
        trial_num = trial_data['Number'][i]
        # initilization trial
        if trial_num == 0:
            continue
        # valid trial
        if trial_num not in processed_trial:
            processed_trial[trial_num] = dict()
        processed_trial[trial_num][trial_data['Event'][i]] = trial_data['Timestamp'][i]
    
    for k in list(processed_trial.keys()):
        if not filtered_start in processed_trial[k]:
            del processed_trial[k]
        else:
            if filtered_end !='End' and not filtered_end in processed_trial[k]:
                del processed_trial[k]
            elif filtered_end =='End':
                check_valid = [cur for cur in ['End','End_Correct', 'End_Miss', 'End_Wrong'] if cur in processed_trial[k]]
                if len(check_valid) == 0:
                    del processed_trial[k]
    return processed_trial

def get_valid_face(trial_info, id2face):
    """ Filter out invalid face (e.g., faces that have been visited before)
        based on trial info. 
    """
    valid_face = dict()
    for i in range(len(trial_info['FaceId'])):
        if trial_info['Type'][i] == 'Stimulus':
            continue
        cur_num = trial_info['Number'][i]
        if cur_num not in valid_face:
            valid_face[cur_num] = []
        valid_face[cur_num].append(id2face[trial_info['FaceId'][i]])
    
    return valid_face

def get_trial_json_mapping(trial_info):
    trial2json = dict()
    for i in range(len(trial_info['Number'])):
        cur_num = trial_info['Number'][i]
        if cur_num in trial2json:
            continue

        cur_json = trial_info['Index'][i]
        trial2json[cur_num] = cur_json

    return trial2json

def get_stimulus_mapping(trial_info, id2stim):
    trial2stim = dict()
    for i in range(len(trial_info['Number'])):
        cur_num = trial_info['Number'][i]
        if cur_num in trial2stim:
            continue
        trial2stim[cur_num] = id2stim[trial_info['Stimulus'][i]]

    return trial2stim    

def MinosEyeConversion(eye_x, eye_y, eye_z, stim_size):
    projected_x, projected_y = [], []
    for i in range(len(eye_x)):
        projected_x.append(eye_x[i]/(eye_z[i]+1e-7)/tand(stim_size/2)*0.5*(9/16)+0.5)
        projected_y.append(eye_y[i]/(eye_z[i]+1e-7)/tand(stim_size/2)*0.5+0.5)
    return projected_x, projected_y


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

def MinosMatlabWrapper(minos_dir, tmp_dir):
    """ A temporarily super-script for reading Minos data
    from temporary Matlab files, and merging them into one dictionary.

    Input:
        minos_dir: Directory storing raw Minos/Ephys files.
        tmp_dir: Directory storing temporary files, i.e., 
                some raw files can not be directly loaded by Python and require
                manual conversion with Matlab.
    """
    session_name = os.path.basename(tmp_dir)

    # read preprocessed files for trial, eye tracking, and player
    trial_data = loadmat(os.path.join(tmp_dir, 'trial.mat'), simplify_cells=True)['trial']
    eye_data = loadmat(os.path.join(tmp_dir, 'eye.mat'), simplify_cells=True)['eye']
    player_data = loadmat(os.path.join(tmp_dir, 'player.mat'), simplify_cells=True)['player']

    # convert the struct data to dictionary
    processed_trial = dict()
    for paradigm in trial_data:
        processed_trial[paradigm] = MatStruct2Dict(trial_data[paradigm]['Data'])
    processed_eye = MatStruct2Dict(eye_data['Data'])
    processed_eye['SyncedTime'] = eye_data['T_']
    processed_player = MatStruct2Dict(player_data['Data'])
    processed_player['SyncedTime'] = player_data['T_']

    # load the ephys data
    ephys_offset = 0.2 # for each trial, include data before and after the offset as baseline
    ephys_data = loadmat(os.path.join(minos_dir, 
                        session_name+'.spiky.ephys.SpikeInfo.mat'), simplify_cells=True)['data']['Value']
    spike_data = [ephys_data['Spikes']['Value'][unit]['T_'] for unit in range(len(ephys_data['Spikes']['Value']))]
    neuron_type = json.load(open(os.path.join(tmp_dir, 'neuron_type.json')))

    # merge the behavioral data and neural data
    # since the preprocessed files does not have timestamps for each phase, use SpikyPy instead
    for paradigm in trial_data:
        tmp_trial_data = MinosData(os.path.join(minos_dir, 'Minos', ' '.join(re.split(r'(?<!^)(?=[A-Z])', paradigm)), 
                                'Trials.bin')).Values
        
        
        if paradigm in ['FiveDot', 'PassiveFixation']:
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
            if paradigm in ['FiveDot', 'PassiveFixation']:
                # align eye data
                start_idx, end_idx = align_trial(trial_num, processed_eye, tmp_trial_data, 'Start_Align', 'End')
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
            if paradigm in ['FiveDot', 'PassiveFixation']:
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
            start_time -= ephys_offset
            end_time += ephys_offset
            trial_spike = cropped_spike_period(spike_data, start_time, end_time)
            processed_trial[paradigm]['Spike'].append(trial_spike)
        
    
    final_data = {'Neuron_type': neuron_type, 'Paradigm': processed_trial}

    return final_data
            
            




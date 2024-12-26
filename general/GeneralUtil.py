import numpy as np
from matplotlib import pyplot as plt
import os
import re
import json
from minos.MinosData import MinosData
from scipy.io import loadmat
from polyface.PolyfaceUtil import process_trial, align_trial, find_closest
from glob import glob
from numba import njit

@njit
def cropped_spike_period_unit(unit, start, end):
    result = []
    for value in unit:
        if start < value < end:
            result.append(value)
    return np.array(result)

def cropped_spike_period(spike_data, start, end):
    results = [cropped_spike_period_unit(unit, start, end) for unit in spike_data]
    return results

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

def process_trial_info(trial_info, paradigm):
    """ Reorganize the trial info based on the type of paradigm 
    for downstream processing. The stored information follows the
    convention of the original Spiky (remaining information can be read from Json file
    in Polyface).
    """
    processed_trial_info = dict()
    if paradigm == 'FiveDot':
        processed_trial_info['Timestamp'] = trial_info['Timestamp']
        processed_trial_info['Number'] = trial_info['Number']
        processed_trial_info['Size'] = trial_info['Size']
        processed_trial_info['Pos'] = []
        for i in range(len(trial_info['Number'])):
            processed_trial_info['Pos'].append([trial_info['PosX'][i],
                                              trial_info['PosY'][i],
                                              trial_info['PosZ'][i]])
    elif paradigm == 'PassiveFixation':
        # passive fixation has unique trial number
        processed_trial_info['Timestamp'] = trial_info['Timestamp']
        processed_trial_info['Number'] = trial_info['Number']
        processed_trial_info['DotSize'] = trial_info['DotSize']
        processed_trial_info['StimSize'] = trial_info['StimSize']
        processed_trial_info['Stimulus'] = trial_info['Stimulus']
        processed_trial_info['DotPos'] = []
        processed_trial_info['StimPos'] = []
        for i in range(len(trial_info['Number'])):
            processed_trial_info['DotPos'].append([trial_info['DotPosX'][i],
                                              trial_info['DotPosY'][i],
                                              trial_info['DotPosZ'][i]])
            processed_trial_info['StimPos'].append([trial_info['StimPosX'][i],
                                              trial_info['StimPosY'][i],
                                              trial_info['StimPosZ'][i]])
    elif paradigm == 'PolyFaceNavigator':
        # polyface has multiple stimuli and thus need to reorganize them by trial number first
        processed_trial_info['Timestamp'] = []
        processed_trial_info['Number'] = []
        processed_trial_info['Index'] = []
        processed_trial_info['FaceId'] = []
        processed_trial_info['Type'] = []
        processed_trial_info['Position'] = []
        processed_trial_info['Rotation'] = []
        for i in range(len(trial_info['Number'])):
            # only consider the cue for data storage (use downstream analysis code to access the other data)
            if trial_info['Type'][i] != 'Stimulus':
                continue
            processed_trial_info['Timestamp'].append(trial_info['Timestamp'][i])
            processed_trial_info['Number'].append(trial_info['Number'][i])
            processed_trial_info['Index'].append(trial_info['Index'][i])
            processed_trial_info['FaceId'].append(trial_info['FaceId'][i])
            processed_trial_info['Type'].append(trial_info['Type'][i])
            processed_trial_info['Position'].append([trial_info['PositionX'][i],
                                              trial_info['PositionY'][i],
                                              trial_info['PositionZ'][i]])            
            processed_trial_info['Rotation'].append([trial_info['RotationX'][i],
                                              trial_info['RotationY'][i],
                                              trial_info['RotationZ'][i]])    
    else:
        NotImplementedError('Selected paradigm not supported for processing')

    return processed_trial_info
            
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
    spike_data = [np.array(unit) for unit in spike_data] 
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

def MinosPythonWrapper(minos_dir, eye_offset_before=0, eye_offset_after=0, eye_offset=False):
    """ Super-script for reading raw Minos data from the output directory, and merging them 
    into one dictionary. Note that the code only process the behavioral data, without synchronization. 
    For synchronization/integration of neural data, please further use MinosIO.

    Input:
        minos_dir: Directory storing raw Minos files.
        eye_offset_before/after: include offset before and after each trial (time in second)
        eye_offset: temporary option to also save the eye data for offset periods.
    Return:
        A dictionary storing the behavioral data for all available paradigms.
    """
    # pre specify a set of keyword related to time
    time_related_key = ['Timestamp', 'Start_Align', 'End', 'End_Miss', 'End_Correct', 'Loading', 'Align_Loading',
                        'On', 'Off', 'End_Wrong', 'ParadigmStop', 'ParadigmStart']

    # load the eye, player, reward and sync data independent of paradigms
    eye_data = MinosData(os.path.join(minos_dir, 'Minos', 'Eye.bin')).Values
    player_data = MinosData(os.path.join(minos_dir, 'Minos', 'Player.bin')).Values
    reward_data = MinosData(os.path.join(minos_dir, 'Minos', 'Reward.bin')).Values
    sync_data = MinosData(os.path.join(minos_dir, 'Minos', 'Sync.bin')).Values
    sync_start = sync_data['Timestamp'][0] # use the first sync time as start time of the system

    # iterate through all paradigms
    processed_trial = dict()
    all_paradigm = [os.path.basename(cur).replace(' ', '') for cur in glob(os.path.join(minos_dir, 'Minos', '*')) 
                if os.path.isdir(cur) and 'Assets' not in cur]
    for paradigm in all_paradigm:
        processed_trial[paradigm] = dict()
        tmp_trial_data = MinosData(os.path.join(minos_dir, 'Minos', ' '.join(re.split(r'(?<!^)(?=[A-Z])', paradigm)), 
                                'Trials.bin')).Values
        tmp_trial_info = MinosData(os.path.join(minos_dir, 'Minos', ' '.join(re.split(r'(?<!^)(?=[A-Z])', paradigm)), 
                                'TrialInfo.bin')).Values
        tmp_trial_info = process_trial_info(tmp_trial_info, paradigm)
        for k in tmp_trial_info:
            processed_trial[paradigm][k] = []

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

        for idx in range(len(tmp_trial_info['Number'])):
            trial_num = tmp_trial_info['Number'][idx]
            if trial_num not in tmp_trial_data:
                continue
            if paradigm != 'PolyFaceNavigator':
                # align eye data
                start_idx, end_idx = align_trial(trial_num, eye_data, tmp_trial_data, 'Start_Align', 'End')

                if eye_offset:
                    adjusted_start_time = tmp_trial_data[trial_num]['Start_Align']-eye_offset_before*1e7
                    start_idx = find_closest(adjusted_start_time, eye_data['Timestamp'])
                    if 'End_Miss' in tmp_trial_data[trial_num]:
                        adjusted_end_time = tmp_trial_data[trial_num]['End_Miss']+eye_offset_after*1e7
                    else:
                        adjusted_end_time = tmp_trial_data[trial_num]['End_Correct']+eye_offset_after*1e7
                    end_idx = find_closest(adjusted_end_time, eye_data['Timestamp'])

                aligned_eye = {k: eye_data[k][start_idx:end_idx] for k in eye_data}
                processed_trial[paradigm]['Eye'].append(aligned_eye)
            else:        
                # align eye data during cue phase
                start_idx, end_idx = align_trial(trial_num, eye_data, tmp_trial_data, 'On', 'Off')
                aligned_eye_cue = {k: eye_data[k][start_idx:end_idx] for k in eye_data}

                # align eye data during navigation phase
                start_idx, end_idx = align_trial(trial_num, eye_data, tmp_trial_data, 'Start', 'End')
                aligned_eye_arena = {k: eye_data[k][start_idx:end_idx] for k in eye_data}

                # align player data
                start_idx, end_idx = align_trial(trial_num, player_data, tmp_trial_data, 'Start', 'End')
                aligned_player = {k: player_data[k][start_idx:end_idx] for k in player_data}

                # align reward
                start_time, end_time = aligned_eye_arena['Timestamp'][0], aligned_eye_arena['Timestamp'][-1]
                aligned_reward = [cur for cur in reward_data['Timestamp'] if cur >=start_time and cur<=end_time]
                aligned_reward = {'Timestamp': aligned_reward}

                processed_trial[paradigm]['Eye_cue'].append(aligned_eye_cue)
                processed_trial[paradigm]['Eye_arena'].append(aligned_eye_arena)
                processed_trial[paradigm]['Player'].append(aligned_player)
                processed_trial[paradigm]['Reward'].append(aligned_reward)

            # merging data from trial data
            for k in tmp_trial_data[trial_num]:
                if k not in processed_trial[paradigm]:
                    processed_trial[paradigm][k] = []
                processed_trial[paradigm][k].append(tmp_trial_data[trial_num][k])
            
            # merge data from trial info
            for k in tmp_trial_info:
                if k not in processed_trial[paradigm]:
                    processed_trial[paradigm][k] = []
                processed_trial[paradigm][k].append(tmp_trial_info[k][idx])                

    # correct the time based on the first sync time (from 100ns to second)
    for k in processed_trial:
        for t_keyword in time_related_key:
            if t_keyword in processed_trial[k]:
                processed_trial[k][t_keyword] = [(t-sync_start)/1e7 if not np.isnan(t) else t for t in processed_trial[k][t_keyword]]
        for trial_idx in range(len(processed_trial[k]['Number'])):
            if 'Eye' in processed_trial[k]:
                processed_trial[k]['Eye'][trial_idx]['Timestamp'] = [(t-sync_start)/1e7 for t in processed_trial[k]['Eye'][trial_idx]['Timestamp']]
            else:
                processed_trial[k]['Eye_cue'][trial_idx]['Timestamp'] = [(t-sync_start)/1e7 for t in processed_trial[k]['Eye_cue'][trial_idx]['Timestamp']]
                processed_trial[k]['Eye_arena'][trial_idx]['Timestamp'] = [(t-sync_start)/1e7 for t in processed_trial[k]['Eye_arena'][trial_idx]['Timestamp']]
                processed_trial[k]['Player'][trial_idx]['Timestamp'] = [(t-sync_start)/1e7 for t in processed_trial[k]['Player'][trial_idx]['Timestamp']]
                processed_trial[k]['Reward'][trial_idx]['Timestamp'] = [(t-sync_start)/1e7 for t in processed_trial[k]['Reward'][trial_idx]['Timestamp']]

    sync_data['Timestamp'] = [(t-sync_start)/1e7 for t in sync_data['Timestamp']] 

    return {'Paradigm': processed_trial, 'Sync':sync_data}






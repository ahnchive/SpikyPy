import numpy as np
import math
from MinosData import MinosData
from scipy.io import loadmat
import os
import json
import re
import time


# room boundary of the polyface task
PolyFaceLayout = {'circle': [[-5, -1], [-6.4, -2]],
               'rectangle': [[1.6, 4.7], [-5.2, 2.7]],
               'diamond': [[-5.4, -0.5], [-0.5, 4.5]],
               'triangle': [[-11.8, -5.4], [4.4, 10.6]],
                }

CorridorLayout = [[[-5.8, -4.2], [3, 4.8]],
                [[-3, -2.24], [-1.9, -0.56]],
                [[-0.83, 1.7], [-4.3, -3.6]],
                [[-0.33, 1.75], [1.2, 1.95]]]

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

def check_corridor(x, z):
    flag = False
    for corridor in CorridorLayout:
        if (x >= corridor[0][0] and x <= corridor[0][1]
                            and z >= corridor[1][0] and z <= corridor[1][1]):
            flag = True
            break

    return flag        

def get_room_loc(player_data, trial_start, trial_end):
    """ finding the real-time location of the player (w.r.t. rooms and corridors).

        Inputs:
            player_data: preprocessed player data for the current trial.
            trial_start: trial_end time.
            trial_end: trial_end time.
        Returns:
            A dictionary storing room locations over time (in blocks)
    """

    loc_data = np.array(player_data['Pos'])
    cur_room = None
    prev_start = None
    room_block = {'circle': [], 'diamond': [], 'triangle': [],
                  'rectangle': [], 'corridor': []}
    for i in range(len(loc_data)):
        x, z = loc_data[i, 0], loc_data[i, 2]
        
        if check_corridor(x, z):
            tmp_room = 'corridor'
        else:
            tmp_room = None
            for room in PolyFaceLayout:
                if (x >= PolyFaceLayout[room][0][0] and x <= PolyFaceLayout[room][0][1]
                    and z >= PolyFaceLayout[room][1][0] and z <= PolyFaceLayout[room][1][1]):
                    tmp_room = room
                    break
            if tmp_room is None:
                continue 

        if cur_room is None:
            cur_room = tmp_room
            room_block[cur_room].append([trial_start, player_data['SyncedTime'][i]])
            prev_start = player_data['SyncedTime'][i]
        elif tmp_room != cur_room:
            room_block[cur_room].append([prev_start, player_data['SyncedTime'][i]])
            cur_room = tmp_room
            prev_start = player_data['SyncedTime'][i]
    
    # somehow the player data stopped when the monkey is not moving
    room_block[cur_room].append([prev_start, trial_end])

    return room_block

def convert_face_pos(body_pos, body_rot):
    """ Convert the face location given the recorded body location.
    """
    # local head position
    head_local = [0.096, 1.695, 0.101]

    # Convert rotation angle from degrees to radians
    theta = math.radians(body_rot)
    
    # Calculate sine and cosine of the rotation angle
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    # Extract local coordinates
    x_local, y_local, z_local = head_local
    
    # Apply rotation around the y-axis
    x_rotated = cos_theta * x_local + sin_theta * z_local
    y_rotated = y_local  # Y-coordinate remains the same
    z_rotated = -sin_theta * x_local + cos_theta * z_local
    
    # Translate by the object's global position
    x_global = x_rotated + body_pos[0]
    y_global = y_rotated + body_pos[1]
    z_global = z_rotated + body_pos[2]
    
    return [x_global, y_global, z_global]    

def get_face_pos(trial_info):
    """ Get the face locations for all trials.
        Input:
            - trial_info: trial info load by MinosData
        Return:
            - A dictionary storing the converted location and label of each face
    """
    face_loc = dict()
    for i in range(len(trial_info['FaceId'])):
        if trial_info['Type'][i] == 'Stimulus':
            continue
        cur_num = trial_info['Number'][i]
        if cur_num not in face_loc:
            face_loc[cur_num] = dict()
        face_loc[cur_num][trial_info['FaceId'][i]] = dict()
        face_loc[cur_num][trial_info['FaceId'][i]]['isTarget'] = True if 'Correct' in trial_info['Type'][i] else False
        
        body_loc = [trial_info['PositionX'][i], trial_info['PositionY'][i], trial_info['PositionZ'][i]]
        body_rot =trial_info['RotationY'][i]
        face_loc[cur_num][trial_info['FaceId'][i]]['location'] = convert_face_pos(body_loc, body_rot)
    
    return face_loc

def compute_ray_vector(player_location, player_yaw, object_location):
    """
    Computes the vector representing the ray cast from the player to the object,
    in the player's local space, considering only rotation around the Y-axis.

    Parameters:
    - player_location: [x, y, z] coordinates of the player.
    - player_yaw: Rotation angle around Y-axis in degrees.
    - object_location: [x, y, z] coordinates of the object.

    Returns:
    - ray_vector: The vector in player's local space pointing from the player to the object.
    """

    # Compute the direction vector from the player to the object in world space
    dx = object_location[0] - player_location[0]
    dy = object_location[1] - player_location[1]
    dz = object_location[2] - player_location[2]

    # Convert yaw angle to radians and invert it for the coordinate transformation
    yaw_rad = math.radians(-player_yaw)

    # Calculate cosine and sine of the yaw angle
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Rotate the direction vector into the player's local space
    x_local = cos_yaw * dx - sin_yaw * dz
    y_local = dy  # Y-axis remains the same in this rotation
    z_local = sin_yaw * dx + cos_yaw * dz

    ray_vector = [x_local, y_local, z_local]

    return ray_vector

def compute_angle_between_vectors(v1, v2):
    """
    Computes the angle in degrees between two 3D vectors.

    Parameters:
    - v1: First vector [x, y, z].
    - v2: Second vector [x, y, z].

    Returns:
    - angle_deg: The angle between the vectors in degrees.
    """
    # Compute the dot product of the two vectors
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Compute the magnitudes of the vectors
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    
    # Check for zero magnitude to avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("Cannot compute angle with zero-length vector.")
    
    # Compute the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clamp the cosine value to the valid range [-1, 1] to handle numerical errors
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    # Compute the angle in radians and then convert to degrees
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def eye_face_interaction(player_data, eye_data, face_loc, tolerance=10):
    """ Compute the interaction between eye tracking data and different faces.

        Inputs:
            - player_data: player locations for the current trials.
            - eye_data: eye tracking data for the current trials.
            - face_loc: processed dictionary for the location/label of each face within the environment.
            - tolerance: consider looking at a face if the angle between eye and player-object rays is less
                        the tolerance.

        Return:
            - A dictionary storing the time (start and end time) when looking at different faces. 
    """

    # TODO
    pass


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
            
            




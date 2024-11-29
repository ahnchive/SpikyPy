import numpy as np
import math
import os
import json
import time
import pickle
from polyface.MiscUtil import do_intersect
from matplotlib import pyplot as plt

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

def compute_distance(player, obj):
    distance = math.sqrt(sum((player[i]-obj[i])**2 for i in range(len(player))))
    return distance

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

def CorrectMinosEye(eye_data):
    """ Correct the convergence value of Minos eye data 
        if either of the eye gaze is nan.
    """
    for i in range(len(eye_data['Convergence'])):
        if any(np.isnan(eye_data['LeftGaze'][i])) or any(np.isnan(eye_data['RightGaze'][i])):
            eye_data['Convergence'][i] = np.array([np.nan]*3)
        
    return eye_data

def MinosEyeConversion(eye_x, eye_y, eye_z, stim_size):
    projected_x, projected_y = [], []
    for i in range(len(eye_x)):
        if any([np.isnan(cur) for cur in [eye_x[i], eye_y[i], eye_z[i]]]):
            projected_x.append(-1)
            projected_y.append(-1)
        else:
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

def get_current_room(x, z):
    tmp_room = None
    for room in PolyFaceLayout:
        if (x >= PolyFaceLayout[room][0][0] and x <= PolyFaceLayout[room][0][1]
            and z >= PolyFaceLayout[room][1][0] and z <= PolyFaceLayout[room][1][1]):
            tmp_room = room
            break    
    return tmp_room

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
            tmp_room = get_current_room(x, z)
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

def find_transition(room_block):
    """ Auxiliary function for converting room into room transition.
    """
    transition_block = dict()
    all_block = []
    for room in room_block:
        for block in room_block[room]:
            transition_block[tuple(block)] = room
            all_block.append(block)
    all_block.sort(key=lambda x: x[0])
    prev_room = transition_block[tuple(all_block[0])]
    converted_block = dict()
    for i in range(1, len(all_block)):
        cur_room = transition_block[tuple(all_block[i])]
        if cur_room == 'corridor':
            continue
        if cur_room != prev_room:
            cur_transition = prev_room + '_' + cur_room
            if cur_transition not in converted_block:
                converted_block[cur_transition] = []
            converted_block[cur_transition].append(all_block[i])
        
        prev_room = cur_room
    return converted_block

def convert_face_pos(body_pos, body_rot):
    """ Convert the face location given the recorded body location.
    """
    # local head position
    head_local = [0.096, 1.695, 0.101]

    # # Convert rotation angle from degrees to radians
    # theta = math.radians(body_rot)
    
    # # Calculate sine and cosine of the rotation angle
    # cos_theta = math.cos(theta)
    # sin_theta = math.sin(theta)
    
    # # Extract local coordinates
    # x_local, y_local, z_local = head_local
    
    # # Apply rotation around the y-axis
    # x_rotated = cos_theta * x_local + sin_theta * z_local
    # y_rotated = y_local  # Y-coordinate remains the same
    # z_rotated = -sin_theta * x_local + cos_theta * z_local
    
    # # Translate by the object's global position
    # x_global = x_rotated + body_pos[0]
    # y_global = y_rotated + 0.02 # fixed y-loc
    # z_global = z_rotated + body_pos[2]
    
    # just use the simple addition
    x_global = body_pos[0]
    y_global = 1.695 + 0.02
    z_global = body_pos[2]

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
        face_loc[cur_num][trial_info['FaceId'][i]]['location'] = [body_loc[0], 0.5, body_loc[2]]
    
    return face_loc

def compute_ray_vector(player_position, player_yaw_degrees, object_position):
    """
    Compute the object ray in the player's local space, considering rotation around the y-axis.

    Inputs:
        - player_position: numpy array of shape (3,)
        - player_yaw_degrees: float, player's rotation around the y-axis in degrees
        - object_position: numpy array of shape (3,)

    Returns:
        - object_ray: numpy array of shape (3,), normalized vector in player's local space
    """
    # Compute the vector from the player to the object in world space
    to_object_world = object_position - player_position  # Shape: (3,)

    # Convert yaw angle to radians
    yaw_radians = np.deg2rad(player_yaw_degrees)

    # Compute the rotation matrix for the inverse rotation (rotate by -yaw)
    cos_theta = np.cos(-yaw_radians)
    sin_theta = np.sin(-yaw_radians)

    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0,         1, 0       ],
        [-sin_theta,0, cos_theta]
    ])

    # Transform the vector into the player's local space
    to_object_local = rotation_matrix @ to_object_world

    # Normalize the vector to get the direction
    object_ray = to_object_local / np.linalg.norm(to_object_local)

    return object_ray

def compute_angle_between_vectors(object_ray, eye_ray_input):
    """
    Compute the angle between the object ray and eye movement ray.

    Inputs:
        - object_ray: numpy array of shape (3,), normalized vector
        - eye_ray_input: numpy array of shape (3,), where x and y are in [-1, 1], and z is depth

    Returns:
        - angle_in_degrees: float, the angle between the two rays in degrees
    """
    # Construct the eye ray vector and normalize it
    eye_ray = eye_ray_input / np.linalg.norm(eye_ray_input)

    # Ensure the object ray is normalized
    normalized_object_ray = object_ray / np.linalg.norm(object_ray)

    # Calculate the dot product between the two rays
    dot_product = np.dot(normalized_object_ray, eye_ray)

    # Clamp the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_in_radians = np.arccos(dot_product)
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees


def occlusion_detection(player_pos, face_id, face_loc, wall_layout):
    """Check if a wall blocks the view from the player to a specific object.

    Inputs:
        - player_pos: Tuple (x, y, z) representing the player's position.
        - face_id: ID of the specific object.
        - face_loc: Dictionary mapping face IDs to their positions (x, z).
        - wall_layout: List of walls, each defined by two points [(x1, z1), (x2, z2)].
    Returns:
        - True if occlusion is detected, False otherwise.
    """
    player_point = (player_pos[0], player_pos[2])  # (x, z)
    object_point = (face_loc[face_id]['location'][0], face_loc[face_id]['location'][2])  # (x, z)

    for wall in wall_layout:
        p2 = [wall['startPoint']['x'], wall['startPoint']['y']]
        q2 = [wall['endPoint']['x'], wall['endPoint']['y']]
        if do_intersect(player_point, object_point, p2, q2):
            return True  # Occlusion detected

    return False  # No occlusion

def eye_face_interaction(player_data, eye_data, face_loc, tolerance=10, wall_layout=None):
    """ Compute the interaction between eye tracking data and different faces.

        Inputs:
            - player_data: player locations for the current trials.
            - eye_data: eye tracking data for the current trials.
            - face_loc: processed dictionary for the location/label of each face within the environment.
            - tolerance: consider looking at a face if the angle between eye and player-object rays is less
                        the tolerance.
            - wall_layout: layout of the walls for polyface

        Return:
            - A dictionary storing the time (start and end time) when looking at different faces. 
    """

    # iterate through all eye data to compute the interaction
    recorded_interaction = dict()

    for step in range(len(eye_data['SyncedTime'])):
        eye_time = eye_data['SyncedTime'][step]
        if math.sqrt(sum(a * a for a in eye_data['Convergence'][step])) == 0:
            continue

        # align the player data to the eye data
        # this step is necessarily because player data is only recorded when the agent is moving
        # while the eye data is recorded all the time
        if eye_time < player_data['SyncedTime'][0]:
            aligned_idx = 0 # agent starts moving after some delay
        elif eye_time > player_data['SyncedTime'][-1]:
            aligned_idx = -1 # agent stops moving at the end of the trial
        else:
            aligned_idx = find_closest(eye_time, player_data['SyncedTime'])

        player_pos = player_data['Pos'][aligned_idx]
        player_rot = player_data['Rot'][aligned_idx][1]

        # iterate through all faces and compute their viewing angle
        # select the one with minimal angle between eye/object rays
        matched_face = []
        dist_pool = [] 
        for face in face_loc:
            # ignore faces that are too far away (typically in another room)
            dist = compute_distance(player_pos, face_loc[face]['location'])

            face_ray = compute_ray_vector(player_pos, player_rot, 
                                face_loc[face]['location'])
            eye_ray = eye_data['Convergence'][step]
            angle_between = compute_angle_between_vectors(face_ray, eye_ray)

            # consider the agent looking at a face if 
            # (1) if eye ray agrees with the ray cast from the player to the object
            # (2) no occlusion is detected along the way 
            if (angle_between < tolerance and 
                not occlusion_detection(player_pos, face, face_loc, wall_layout)):
                matched_face.append(face)
                dist_pool.append(dist)
        
        if len(matched_face)>0:
            matched_face = matched_face[np.argmin(dist_pool)]
            if matched_face not in recorded_interaction:
                recorded_interaction[matched_face] = []
            recorded_interaction[matched_face].append(eye_data['SyncedTime'][step])

    # group discrete time stamps into blocks
    group_interaction = dict()
    for face in recorded_interaction:
        group_interaction[face] = []
        prev_t = recorded_interaction[face][0]
        start_t = recorded_interaction[face][0]
        for cur_t in recorded_interaction[face]:
            if cur_t - prev_t > 0.2: # allow 200ms maximal gaps (i.e., similar to off-tolerance in Passive)
                group_interaction[face].append([start_t, prev_t])
                start_t = cur_t
            prev_t = cur_t
        group_interaction[face].append([start_t, prev_t])
    
    return group_interaction

# Function to map the original coordinates to the new array coordinates (for visualization)
def map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size):
    x_scaled = (x - x_min) / (x_max - x_min) * (array_size - 1)
    y_scaled = (y - y_min) / (y_max - y_min) * (array_size - 1)
    return int(x_scaled), int(y_scaled)    

def compute_spatial_frequency(trial_data, wall_layout, array_size=1008, subset=None):
    """ Compute the frequency of visiting different location. The result are
        remapped onto an array consistent for other downstream analysis.

        Inputs:
            - trial_data: pre-processed trial data.
            - wall_layout: wall positions of the polyface environment.
            - array_size: size for visualiztaion for downstream analysis.
            - subset: If not None, only compute the stat based on the given subset of trial number.
    
        Return:
            An array storing the frequency of visiting different location.
    """

    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # initialize the frequency array
    spatial_frequency = np.zeros([array_size, array_size])

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue    

        # use eye data as anchor to compute the frequency
        # this is because player data is only recorded when the agent is moving
        eye_time = trial_data['Paradigm']['PolyFaceNavigator']['Eye_arena'][trial_idx]['SyncedTime']
        cur_player_data = trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx]
        aligned_player_loc = [cur_player_data['Pos'][
                            find_closest(eye_time[cur], cur_player_data['SyncedTime'])] 
                            for cur in range(len(eye_time))]
        
        for i in range(len(aligned_player_loc)):
            x, y, = aligned_player_loc[i][0], aligned_player_loc[i][2]
            x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
            spatial_frequency[y_mapped, x_mapped] += 1

    spatial_frequency /= spatial_frequency.max()
    
    return spatial_frequency

def compute_reward_frequency(trial_data, wall_layout, array_size=1008, subset=None):
    """ Compute the frequency of reward at different locations. The result are
        remapped onto an array consistent for other downstream analysis.

        Inputs:
            - trial_data: pre-processed trial data.
            - wall_layout: wall positions of the polyface environment.
            - array_size: size for visualiztaion for downstream analysis.
            - subset: If not None, only compute the stat based on the given subset of trial number.
    
        Return:
            An array storing the unnormalized frequency of rewards at different location.
    """

    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # initialize the frequency array
    reward_frequency = np.ones([array_size, array_size])

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break

        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        if subset is not None and trial_number not in subset:
            continue    

        # account for the locations with reward
        reward_time = trial_data['Paradigm']['PolyFaceNavigator']['Reward'][trial_idx]
        if len(reward_time) == 0:
            continue

        cur_player_data = trial_data['Paradigm']['PolyFaceNavigator']['Player'][trial_idx]
        aligned_player_loc = [cur_player_data['Pos'][
                            find_closest(reward_time[cur], cur_player_data['SyncedTime'])] 
                            for cur in range(len(reward_time))]
        
        for i in range(len(aligned_player_loc)):
            x, y, = aligned_player_loc[i][0], aligned_player_loc[i][2]
            x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
            reward_frequency[y_mapped, x_mapped] += 1
    
    return reward_frequency

def compute_face_map(trial_info, wall_layout, array_size=1008, subset=None):
    """ Compute a heatmap based on the locations of faces.

        Inputs:
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - array_size: size for visualiztaion for downstream analysis.
            - subset: If not None, only compute the stat based on the given subset of trial number.
    
        Return:
            An array storing the unnormalized frequency of faces at different location.
    """
    # gather the boundary of the environments
    all_x, all_y = [], []
    for wall in wall_layout:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        all_x.extend(x)
        all_y.extend(y)
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # initialize the frequency array
    face_map = np.zeros([array_size, array_size])

    # get the face locations for all trial
    face_loc = get_face_pos(trial_info)

    for trial_num in face_loc:
        # temporarily remove bad data
        if trial_num == 1443:
            break
        for face in face_loc[trial_num]:
            x, _, y = face_loc[trial_num][face]['location']
            x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
            face_map[y_mapped, x_mapped] += 1

    return face_map
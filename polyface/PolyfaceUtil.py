import numpy as np
import math
import os
import json
import time
import pickle
from polyface.MiscUtil import do_intersect
from matplotlib import pyplot as plt
import matplotlib.patches as patches

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

# predefine color for different rooms, corridor has white color
room_color = {'circle': 'red', 'rectangle': 'aqua',
              'diamond': 'green', 'triangle': 'yellow',
              'corridor': 'purple'
              }

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
        end_event = [cur for cur in ['End_Correct', 'End_Miss', 'End_Wrong'] 
                    if cur in cur_trial and not np.isnan(cur_trial[cur])][0]    
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
    
    valid_end_condition = dict()

    for k in list(processed_trial.keys()):
        if not filtered_start in processed_trial[k]:
            del processed_trial[k]
        else:
            if filtered_end !='End' and not filtered_end in processed_trial[k]:
                del processed_trial[k]
            elif filtered_end =='End':
                check_valid = [cur for cur in ['End', 'End_Correct', 'End_Miss', 'End_Wrong'] if cur in processed_trial[k]]
                if len(check_valid) == 0:
                    del processed_trial[k]
                
                for end_cond in check_valid:
                    valid_end_condition[end_cond] = 1

    # fill in all end condition (e.g., fill nan in End_Miss for correct trials)
    for trial_num in processed_trial:
        for end_cond in valid_end_condition:
            if end_cond not in processed_trial[trial_num]:
                processed_trial[trial_num][end_cond] = np.nan

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

    # spatial_frequency /= spatial_frequency.max()
    
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

def find_passage(player_data, place_field, pf_size,
            x_min, x_max, y_min, y_max, array_size):
    """ Find the time (blocks) when the agent passing through a place field.

        Inputs:
            - player_data: player data for the current trial.
            - place_field: x, z location for the center of the place field.
            - pf_size: tolerance for qualifying a passage.
            - x_min/x_max/y_min/y_max, array_size: parameter for coordinate projection.
    """
    passage = []
    start = None

    # recording the time when the agent is closest to the center of the place field
    closest_dist = None 
    closest_time = None 

    for i in range(len(player_data['Pos'])):
        x, y, = player_data['Pos'][i][0], player_data['Pos'][i][2]
        x_mapped, y_mapped = map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size)
        dist_to_center = np.sqrt((x_mapped-place_field[0])**2+(y_mapped-place_field[1])**2)
        if dist_to_center<=pf_size:
            if start is None:
                start = player_data['SyncedTime'][i]
                closest_dist = dist_to_center
                closest_time = player_data['SyncedTime'][i]
            else:
                if dist_to_center<closest_dist:
                    closest_dist = dist_to_center
                    closest_time = player_data['SyncedTime'][i]
        else:
            if start is not None:
                passage.append([start, player_data['SyncedTime'][i], closest_time])
            start = None
    
    return passage

def get_face_interaction_data(trial_data, trial_info, wall_layout, face2id,
                        tolerance=8, stim_start=0.05, stim_end=0.4, bin_size=None, step_size=None,
                        drop_invalid=True, cell_type='rML', subset=None, fsi_score=None, fsi_thres=0.2):
    """ Extract the neural data when looking at a face in the arena.

        Inputs:
            - trial_data: pre-processed trial data.
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - face2id: face2id mapping
            - tolerance: tolerance angle between eye/object ray (to be changed to distance-based measurement).
            - stim_start/end: start/end time to extract the neural responses, aligned to onset.
            - step_size: step size for bining, None for not bining
            - bin_size: breaking the neural responses into bins for temporal analysis, None for not bining
            - drop_invalid: dropping the invalid data (shorter than specified stim_time) or not .
            - cell_type: type of cells for consideration
            - save_path: a pickle file to save the extracted data.
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - fsi_score: if not None, only consider the top face selective cell.
            - fsi_thres: threshold for face cell selection.        
        Returns:
            Neural responses when interacting with different faces (stored as a dictionary).
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]

    if fsi_score is not None:
        cell_idx = [cell_idx[idx] for idx in range(len(cell_idx)) if fsi_score[idx]>fsi_thres]

    # get the locations of faces in different trials
    face_loc = get_face_pos(trial_info)
    id2face = {face2id[k]: k for k in face2id}

    # initialize face response structure
    face_response = dict()

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

        # measure PSTH for each individual face interaction event
        for face in interaction_block:
            face_name = id2face[face].replace('target_', '').split('_')[0]
            if face_name not in face_response:
                face_response[face_name] = []

            for event in interaction_block[face]:
                if event[0]-trial_onset <= stim_start: # ignore events that are too close to cue phase
                    continue
                event_time = event[1]-event[0]
                event_onset = event[0]+stim_start
                if event_time<(stim_end-stim_start):
                    if drop_invalid:
                        continue
                    else:
                        event_offset = event[1]
                else:
                    event_offset = event_onset + stim_end

                if bin_size is None or step_size is None:
                    tmp_firing = []
                    for neuron_idx, neuron in enumerate(cell_idx):
                        neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]

                        # compute the averaged firing rate throughout the whole stim onset time
                        spike_count = len([1 for spike_time in neuron_spikes 
                                    if spike_time>=event_onset and spike_time<=event_offset])
                        firing_rate = spike_count/(stim_end-stim_start)
                        tmp_firing.append(firing_rate)
                else:
                    num_bin = int((stim_end-stim_start-bin_size)//step_size) + 1
                    tmp_firing = [[] for _ in range(num_bin)]

                    for neuron_idx, neuron in enumerate(cell_idx):
                        neuron_spikes = [cur for cur in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]]
                        for bin_idx in range(num_bin):
                            cur_onset = event_onset + bin_size*step_size
                            cur_offset = event_onset + bin_size*(step_size+1)

                            # compute the averaged firing rate throughout the whole stim onset time
                            spike_count = len([1 for spike_time in neuron_spikes 
                                        if spike_time>=cur_onset and spike_time<=cur_offset])
                            firing_rate = spike_count/(stim_end-stim_start)
                            tmp_firing[bin_idx].append(firing_rate)                    
                    
                face_response[face_name].append(np.array(tmp_firing))
        
    return face_response
                
def get_replay_data(trial_data, trial_info, wall_layout, face2id, filter_rate,
                        tolerance=8, stim_start=0.05, stim_end=0.4, bin_size=None, step_size=None,
                        drop_invalid=True, cell_type='rML', subset=None, fsi_score=None, fsi_thres=0.2):
    """ Extract the neural data when looking at a face in the arena.

        Inputs:
            - trial_data: pre-processed trial data.
            - trial_info: trial info read by MinosData.
            - wall_layout: wall positions of the polyface environment.
            - face2id: face2id mapping
            - filter_rate: average firing rate for filtering the "replay" period
            - tolerance: tolerance angle between eye/object ray (to be changed to distance-based measurement).
            - stim_start/end: start/end time to extract the neural responses, aligned to onset.
            - step_size: step size for bining, None for not bining
            - bin_size: breaking the neural responses into bins for temporal analysis, None for not bining
            - drop_invalid: dropping the invalid data (shorter than specified stim_time) or not .
            - cell_type: type of cells for consideration
            - save_path: a pickle file to save the extracted data.
            - subset: If not None, only compute the stat based on the given subset of trial number.
            - fsi_score: if not None, only consider the top face selective cell.
            - fsi_thres: threshold for face cell selection.        
        Returns:
            - Neural responses for replaying
            - Target face associated with each replaying session
            - Last viewing face associated with each replaying session
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]

    if fsi_score is not None:
        cell_idx = [cell_idx[idx] for idx in range(len(cell_idx)) if fsi_score[idx]>fsi_thres]

    # get the locations of faces in different trials
    face_loc = get_face_pos(trial_info)
    id2face = {face2id[k]: k for k in face2id}

    # initialize face response structure
    replay_response = []
    replay_target_face = []

    # iterate through all trials
    for trial_idx in range(len(trial_data['Paradigm']['PolyFaceNavigator']['Number'])):
        # temporarily remove bad data
        if trial_idx == 158:
            break
        trial_number = trial_data['Paradigm']['PolyFaceNavigator']['Number'][trial_idx]

        cur_target_face = [k for k in face_loc[trial_number] if face_loc[trial_number][k]['isTarget']]
        cur_target_face = id2face[cur_target_face[0]]

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

        # collect the neural data when passing through different locations
        # reorganize the neuron responses (may lose tiny precision)
        time_window = int((trial_offset-trial_onset)*1000)
        processed_neuron_spikes = np.zeros([time_window, len(cell_idx)])
        for neuron_idx, neuron in enumerate(cell_idx):
            for cur_spike in trial_data['Paradigm']['PolyFaceNavigator']['Spike'][trial_idx][neuron]:
                if cur_spike<trial_onset:
                    continue
                spike_time = int((cur_spike-trial_onset)*1000)-1
                processed_neuron_spikes[spike_time, neuron_idx] += 1

        # first collect face interaction periods
        all_face_event = []
        for face in interaction_block:
            for event in interaction_block[face]:
                all_face_event.append((event[0], event[1]))
        all_face_event = sorted(all_face_event)
        
        # collect all non-face interaction periods
        buffer = 0.2 # ignore the period right before/after looking at a face
        non_face_event = []
        current_time = trial_onset
        for idx, (start, end) in enumerate(all_face_event):
            adjusted_start = max(trial_onset, start - buffer)
            adjusted_end = min(trial_offset, end + buffer)
            if adjusted_start > current_time:
                non_face_event.append((current_time, adjusted_start))
            current_time = max(current_time, adjusted_end)    

        if current_time < trial_offset:
            non_face_event.append((current_time, trial_offset))

        # scan through each event and locate sub-periods with potential replaying
        for idx, event in enumerate(non_face_event):
            if event[1]-event[0]<stim_end:
                continue
            
            start_idx = int((event[0]-trial_onset)*1000)-1
            end_idx = int((event[1]-trial_onset)*1000)-1
            search_size = 20
            while start_idx+int(stim_end*1000)<end_idx:
                cur_firing = (processed_neuron_spikes[start_idx:start_idx+int(stim_end*1000)].sum(0)/stim_end).mean()
                if cur_firing >= filter_rate:
                    num_bin = int((stim_end-stim_start-bin_size)//step_size) + 1
                    tmp_firing = []
                    for bin_idx in range(num_bin):
                        tmp_start, tmp_end = start_idx+bin_idx*int(step_size*1000), start_idx+bin_idx*int(step_size*1000)+int(bin_size*1000)
                        tmp_firing.append(
                            processed_neuron_spikes[tmp_start:tmp_end].sum(0)/bin_size)

                    replay_response.append(tmp_firing)
                    replay_target_face.append(cur_target_face)
                    break
                start_idx += search_size
    
    replay_response = np.array(replay_response)
    return replay_response, replay_target_face


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
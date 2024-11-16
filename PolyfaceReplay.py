from MinosData import MinosData
import json
import argparse
import os
import numpy as np
from PolyfaceUtil import align_trial, process_trial, get_valid_face, get_trial_json_mapping

parser = argparse.ArgumentParser('Script for processing Minos navigation data for Minos replay', add_help=False)
parser.add_argument('--record_path', default=None, type=str, help='Root folder storing the Minos recording')
parser.add_argument('--trial_path', default=None, type=str, help='Folder containing the trial json files')
parser.add_argument('--save_path', default=None, type=str, help='file for storing the processed data')
args = parser.parse_args()


# loading all necessary data
trial_info = MinosData(os.path.join(args.record_path, 'Poly Face Navigator', 'TrialInfo.bin')).Values
trial_data = MinosData(os.path.join(args.record_path, 'Poly Face Navigator', 'Trials.bin')).Values
player_data = MinosData(os.path.join(args.record_path, 'Player.bin')).Values
eye_data = MinosData(os.path.join(args.record_path, 'Eye.bin')).Values 
face2id = json.load(open(os.path.join(args.trial_path, 'face2id.json')))
id2face = dict()
for k in face2id:
    id2face[face2id[k]] = k

# preprocess the data for better structure
trial_data = process_trial(trial_data)
valid_face = get_valid_face(trial_info, id2face)
trial2json = get_trial_json_mapping(trial_info)

# looping through all trials and store the information for replay
replay_data = {'player':[], 'trialId': [], 'validFace': [], 'eye':[], 'eyeCue': []}
for trial_num in trial_data:
    json_idx = int(trial2json[trial_num])
    cur_face = valid_face[trial_num]

    # align the navigation data with trials based on time stamp
    start_idx, end_idx = align_trial(trial_num, player_data, trial_data, 'Start', 'End_Correct')
    player_x = [float(player_data['PosX'][idx]) for idx in range(start_idx, end_idx+1)]
    player_y = [float(0.5) for idx in range(start_idx, end_idx+1)]
    player_z = [float(player_data['PosZ'][idx]) for idx in range(start_idx, end_idx+1)]
    player_roty = [float(player_data['RotY'][idx]) for idx in range(start_idx, end_idx+1)]
    tmp_player = {'PosX': player_x, 'PosY': player_y, 'PosZ': player_z, 'RotY': player_roty}

    # align the eye data for navigation phase
    start_idx, end_idx = align_trial(trial_num, eye_data, trial_data, 'Start', 'End_Correct')
    eye_x = [float(eye_data['ConvergenceX'][idx]) for idx in range(start_idx, end_idx+1)]
    eye_y = [float(eye_data['ConvergenceY'][idx]) for idx in range(start_idx, end_idx+1)]
    eye_z = [float(eye_data['ConvergenceZ'][idx]) for idx in range(start_idx, end_idx+1)]
    tmp_eye = {'EyeX': eye_x, 'EyeY': eye_y, 'EyeZ': eye_z}

    # align the eye data for cue phase
    start_idx, end_idx = align_trial(trial_num, eye_data, trial_data, 'On', 'Off')
    cue_eye_x = [float(eye_data['ConvergenceX'][idx]) for idx in range(start_idx, end_idx+1)]
    cue_eye_y = [float(eye_data['ConvergenceY'][idx]) for idx in range(start_idx, end_idx+1)]
    cue_eye_z = [float(eye_data['ConvergenceZ'][idx]) for idx in range(start_idx, end_idx+1)]
    tmp_cue_eye = {'EyeX': cue_eye_x, 'EyeY': cue_eye_y, 'EyeZ': cue_eye_z}

    replay_data['player'].append(tmp_player)
    replay_data['trialId'].append(json_idx)
    replay_data['validFace'].append(cur_face)
    replay_data['eye'].append(tmp_eye)
    replay_data['eyeCue'].append(tmp_cue_eye)

with open(args.save_path, 'w') as f:
    json.dump(replay_data, f)



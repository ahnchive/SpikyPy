import json
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import argparse
import numpy as np
import copy
import os 
from minos.MinosData import MinosData
import imageio
from scipy.ndimage import gaussian_filter
from polyface.PolyfaceUtil import align_trial, process_trial, get_valid_face, get_trial_json_mapping

parser = argparse.ArgumentParser('Script for visualizing the navigation trajectory in top-down view', add_help=False)
parser.add_argument('--trial_setting', default=None, type=str, help='json file containing the trial settings')
parser.add_argument('--trial_record', default=None, type=str, help='folder containing the recorded Minos data')
parser.add_argument('--room_data', default=None, type=str, help='json file storing the spatial layout of the environment')
parser.add_argument('--save_path', default=None, type=str, help='folder for storing the visualization results')

args = parser.parse_args()

# define the location of furnitures
furnitures = {"office_table": [[-29.89, 33.14], [-13.795, 16.425], [14.72, 12.31], [-19.73, -12.13]],
             "table_light": [[-6.84, -13.1], [17.79, -9.22], [-5.63, 12.94], [-32.82, 42.73]],
             "scratching_post": [[-24.06, -19.12], [9.69, -5.4], [-18.27, 6.64], [-42.97, 25.14]],
             "table": [[-10.44, -29.95], [17.95, -0.03], [-5.97, 2.91], [-41.03, 33.97]]
            }

furniture_marker = {"office_table": 's', 'table': 'd', 'scratching_post': 'v', "table_light": 'h'}


def get_face_loc(trial_setting, valid_face):
    """ Function for reading and reorganizing face locations.
    """
    target_face = dict()
    distractor_face = dict()
    for room in ['triangle', 'circle', 'rectangle', 'diamond']:
        for face in trial_setting[room]:
            if face['stimuli'] not in valid_face:
                continue
            if 'target' not in face['stimuli']:
                distractor_face[face['stimuli']] = [face['x'], face['z']]
            else:
                target_face[face['stimuli']] = [face['x'], face['z']]
    return target_face, distractor_face

def plot_single_trial(room_data, target_face, distractor_face, 
                player_trajectory, save_name):
    """ Plotting the trajectory for a single trial (as a static image).

        Inputs:
            room_data: room layout information.
            target_face: location of target faces.
            distractor_face: location of distractor faces.
            player_trajectory: player locations and rotations overtime for the current trial.
            save_name: file name (and directory) for saving the visualization.
    """
    plt.close('all')
    fig, ax = plt.subplots()
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2))

    # plot the spatial layout of the rooms first
    all_x, all_y = [], []
    for wall in room_data['walls']:
        x = [wall['startPoint']['x'], wall['endPoint']['x']]
        y = [wall['startPoint']['y'], wall['endPoint']['y']]
        plt.plot(x, y, c='black')
        all_x.extend(x)
        all_y.extend(y)
   
    # plot the target and distractor faces with different markers/colors
    for idx, face in enumerate(distractor_face):
        if idx == 0:
            plt.scatter(distractor_face[face][0], distractor_face[face][1], 
                    marker="X", c='b', s=50, label='distractor face')
        else:
            plt.scatter(distractor_face[face][0], distractor_face[face][1], 
                    marker="X", c='b', s=50)

    for idx, face in enumerate(target_face):
        if idx == 0:
            plt.scatter(target_face[face][0], target_face[face][1], marker="*", 
                        c='r', s=50, label='target face')
        else:
            plt.scatter(target_face[face][0], target_face[face][1], marker="*", 
                        c='r', s=50)
    
    # plot the furnitures
    multiply_factor = 0.2
    show_once = True
    for furniture in furnitures:
        for single_loc in furnitures[furniture]:
            x, z = single_loc
            x *= multiply_factor
            z *= multiply_factor
            if show_once:
                plt.scatter(x, z, marker='h', c='k', s=50, label='object')
                show_once = False
            else:
                plt.scatter(x, z, marker='h', c='k', s=50)
    
    # plot the trajectory
    plt.plot(player_trajectory[:, 0], player_trajectory[:, 1], 'k--', linewidth=2)

    # draw an arrow to indicate direction
    arrow_start = (player_trajectory[-1, 0], player_trajectory[-1, 1])  # starting point of the arrow
    arrow_length = 0.5   
    arrow_angle = (90 - player_trajectory[-1, 2]) % 360
    # Calculate the end position of the arrow
    arrow_end = (arrow_start[0] + arrow_length * np.cos(np.radians(arrow_angle)),
                arrow_start[1] + arrow_length * np.sin(np.radians(arrow_angle)))

    arrow.set_position(arrow_start)
    arrow.xy = arrow_end

    plt.legend(loc='upper right', fontsize=15)

    plt.axis('off')
    fig.set_size_inches(8, 8)
    fig.savefig(save_name+'.png', bbox_inches='tight')

    # for heatmap visualization
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    array_size = 1008

    # Function to map the original coordinates to the new array coordinates
    def map_coordinates(x, y, x_min, x_max, y_min, y_max, array_size):
        x_scaled = (x - x_min) / (x_max - x_min) * (array_size - 1)
        y_scaled = (y - y_min) / (y_max - y_min) * (array_size - 1)
        return x_scaled.astype(int), y_scaled.astype(int)

    # Map the original coordinates to the new array coordinates
    x_mapped, y_mapped = map_coordinates(player_trajectory[:, 0], player_trajectory[:, 1], x_min, x_max, y_min, y_max, array_size)

    agg_heatmap = np.zeros([array_size, array_size])
    # Populate the array with the count of points
    for i in range(len(x_mapped)):
        agg_heatmap[y_mapped[i], x_mapped[i]] += 1

    return agg_heatmap

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# general data loading
room_layout = json.load(open(args.room_data))
trial_info = MinosData(os.path.join(args.trial_record, 'Poly Face Navigator', 'TrialInfo.bin')).Values
trial_data = MinosData(os.path.join(args.trial_record, 'Poly Face Navigator', 'Trials.bin')).Values
player_data = MinosData(os.path.join(args.trial_record, 'Player.bin')).Values
face2id = json.load(open(os.path.join(args.trial_setting, 'face2id.json')))

id2face = dict()
for face in face2id:
    id2face[face2id[face]] = face

# preprocess the data for better structure
trial_data = process_trial(trial_data)
valid_face = get_valid_face(trial_info, id2face)
trial2json = get_trial_json_mapping(trial_info)

# iterate through each trial for visualization
agg_heatmap = None

for trial_num in trial_data:
    json_idx = int(trial2json[trial_num])

    # retrieve trial setting
    cur_trial_setting = json.load(open(os.path.join(args.trial_setting, 'trial_'+str(json_idx)+'.json')))
    target_face, distractor_face = get_face_loc(cur_trial_setting, valid_face[trial_num])

    # retrieve player data    
    start_idx, end_idx = align_trial(trial_num, player_data, trial_data, 'Start', 'End_Correct')
    player_trial = [[player_data['PosX'][idx], player_data['PosZ'][idx], player_data['RotY'][idx]] for idx in range(start_idx, end_idx+1)]
    player_trial = np.array(player_trial)

    heat_map = plot_single_trial(room_layout, target_face, distractor_face, 
                    player_trial, os.path.join(args.save_path, 'per_trial_visualization', str(trial_num)))

    if agg_heatmap is None:
        agg_heatmap = heat_map
    else:
        agg_heatmap += heat_map

# draw the aggregated heatmap
fig, ax = plt.subplots()

# Plot the static elements once
all_x, all_y = [], []
for wall in room_layout['walls']:
    x = [wall['startPoint']['x'], wall['endPoint']['x']]
    y = [wall['startPoint']['y'], wall['endPoint']['y']]
    plt.plot(x, y, c='black')
    all_x.extend(x)
    all_y.extend(y)
plt.axis('off')
fig.set_size_inches(8, 8)

# plot the furnitures
multiply_factor = 0.2
for furniture in furnitures:
    for single_loc in furnitures[furniture]:
        x, z = single_loc
        x *= multiply_factor
        z *= multiply_factor
        plt.scatter(x, z, marker='h', c='k', s=50)

# Apply Gaussian blur to the heatmap
heatmap_blurred = gaussian_filter(agg_heatmap, sigma=10)

# Overlay the heatmap
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
extent = [x_min, x_max, y_min, y_max]
im = ax.imshow(heatmap_blurred, origin='lower', cmap='viridis', extent=extent, alpha=0.6, interpolation='nearest')
colormap = plt.cm.viridis
colormap.set_under(color='white', alpha=0)
fig.savefig(os.path.join(args.save_path, 'heatmap.png'), bbox_inches='tight')
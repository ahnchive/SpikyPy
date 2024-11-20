import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import json
from polyface.PolyfaceUtil import get_room_loc, get_face_pos, eye_face_interaction

# predefine color for different rooms, corridor has white color
room_color = {'circle': 'red', 'rectangle': 'aqua',
              'diamond': 'green', 'triangle': 'yellow',
              'corridor': 'purple'
              }


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
        if trial_idx == 217:
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

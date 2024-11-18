import numpy as np
from matplotlib import pyplot as plt
import os
import json
from PolyfaceUtil import get_room_loc

# predefine color for different rooms, corridor has white color
room_color = {'circle': 'red', 'rectangle': 'aqua',
              'diamond': 'green', 'triangle': 'yellow',
              'corridor': 'purple'
              }


def plot_raster(trial_data, paradigm, save_dir, save_format='pdf', 
                eye_interaction=False, spatial_overlay=False,
                continuous_only=False):
    """ Visualizing the raster plot for a single trial. The
    plot is divided into different blocks based on a set of discrete events.
    It also supports using background color to incorporate eye interactions
    within the visual scene (e.g., for Polyface).

    Inputs:
        trial_data: preprocessed dictionary.
        paradigm: paradigm for visualization.
        eye_interaction: adding eye interaction data or not.
        spatial_overlay: add spatial position as background color
        save_dir: directory for the saved visualization.
        continuous_only: only consider continuous trials (Polyface only).
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
        # plt.xlim(-offset, trial_len+offset)
        plt.ylim(0, len(trial_data['Neuron_type'])+10)

        for neuron in range(len(trial_data['Paradigm'][paradigm]['Spike'][trial_idx])):
            spike_time = [cur-trial_onset for cur in trial_data['Paradigm'][paradigm]['Spike'][trial_idx][neuron]]
            plt.scatter(spike_time, [neuron+5]*len(spike_time), c='black', marker='.', s=0.2)
        
        for event in discrete_events:
            if paradigm == 'PolyFaceNavigator' and event == 'End':
                event = cur_type
            plt.axvline(x = trial_data['Paradigm'][paradigm][event][trial_idx]-trial_onset, color = 'b', linewidth=2)

        # add spatial background
        if spatial_overlay:
            ax = plt.gca()
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

        # save the plot
        fig.set_size_inches(10, 10)
        fig.savefig(os.path.join(save_dir, type_, str(trial_number)+'.'+save_format), bbox_inches='tight')

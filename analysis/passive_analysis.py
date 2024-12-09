import numpy as np
from matplotlib import pyplot as plt
from analysis.util import normalize_response
from matplotlib.colors import LinearSegmentedColormap


def selectivity_visualization(trial_data, class_mapping, stim_start=0.1, stim_end=0.3, 
                              baseline_start=0.05, baseline_end=0.1, 
                              cell_type='rML', sort_cat=None, save_path=None):
    """ Drawing the classic "orange plot" for the class selectivity during passive
        fixation experiment. 

        Inputs:
            -trial_data: processed trial data.
            -class_mapping: mapping from each stimulus to its class label,
                        assuming that the mapping is in blocks, e.g., 0-k class 1,
                        k+1:2k+1 class 2.
            -stim_start/end: start/end time for computing the stimulus response.
            -baseline_start/end: start/end time for computing the baseline response.
            -sort_cat: sorting the neurons based on selectivity on the selected category.
            -save_path: file name for saving the plot.
    """

    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]
    
    avg_stim_response = np.zeros([len(cell_idx), len(class_mapping)])
    avg_baseline_response= np.zeros([len(cell_idx), len(class_mapping)])
    stim_count = np.zeros([len(class_mapping),])

    for trial_idx in range(len(trial_data['Paradigm']['PassiveFixation']['Number'])):
        # drop invalid trials
        if np.isnan(trial_data['Paradigm']['PassiveFixation']['End_Correct'][trial_idx]):
            continue

        stim_id = trial_data['Paradigm']['PassiveFixation']['Stimulus'][trial_idx]
        stim_count[stim_id] += 1
        trial_onset = trial_data['Paradigm']['PassiveFixation']['Start_Align'][trial_idx]

        # compute the stim/baseline average firing rate for each neuron/repetition
        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PassiveFixation']['Spike'][trial_idx][neuron]]

            stim_spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time-trial_onset>=stim_start and spike_time-trial_onset<=stim_end])
            stim_firing_rate = stim_spike_count/(stim_end-stim_start)
            avg_stim_response[neuron_idx, stim_id] += stim_firing_rate

            baseline_spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time-trial_onset>=baseline_start and spike_time-trial_onset<=baseline_end])
            baseline_firing_rate = baseline_spike_count/(baseline_end-baseline_start)
            avg_baseline_response[neuron_idx, stim_id] += baseline_firing_rate
        
    # normalize by repetitions
    for i in range(len(stim_count)):
        avg_stim_response[:, i] /= stim_count[i]
        avg_baseline_response[:, i] /= stim_count[i]

    norm_response = normalize_response(avg_stim_response, avg_baseline_response)

    # sort neurons
    if sort_cat is not None:
        select_stim = np.array([cur for cur in class_mapping if class_mapping[cur]==sort_cat])
        seletivity = []
        for i in range(len(norm_response)):
            seletivity.append(np.mean(norm_response[i, select_stim]))
        sort_idx = np.argsort(seletivity)[::-1]
        norm_response = np.array([norm_response[idx] for idx in sort_idx])

    cat_count = {}
    for k in class_mapping:
        if class_mapping[k] not in cat_count:
            cat_count[class_mapping[k]] = 0
        cat_count[class_mapping[k]] += 1
    xticks = np.cumsum([cat_count[cur] for cur in cat_count])
    tick_label = [cur for cur in cat_count]

    # Define a colormap from white to orange
    colors = [(1, 1, 1), (1, 0.6, 0)]  # White to Orange
    cmap = LinearSegmentedColormap.from_list("white_orange", colors, N=256)

    plt.close('all')
    fig = plt.figure(figsize=(5, 8))
    plt.gcf().patch.set_facecolor('white')
    plt.imshow(norm_response, aspect='auto', cmap=cmap, origin='upper')
    plt.title("Category selectivity")
    plt.xlabel("Stimulus")
    plt.ylabel("Neuron")
    plt.xticks(ticks=xticks, labels=tick_label)
    plt.yticks([])
    fig.savefig(save_path, bbox_inches='tight')
import numpy as np
from matplotlib import pyplot as plt
from analysis.util import normalize_response
from matplotlib.colors import LinearSegmentedColormap
from analysis.decoding import kfold_cross_validation
from analysis.metrics import FSI
import os
from sklearn.metrics.pairwise import cosine_similarity

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


def stim_decoding(trial_data, class_mapping, stim_start=0.1, stim_end=0.3, 
                    baseline_start=0.05, baseline_end=0.1, 
                    cell_type='rML', sort_cat=None, category_decoding=False, k_fold=8,
                    num_select_neuron=None, classifier='svc', reg_para=0.1, cat_subset=None, seed=888):
    """ Compute the decoding accuracy for each stimulus/category, by default it use
        k-fold cross-validation.

        Inputs:
            -trial_data: processed trial data.
            -class_mapping: mapping from each stimulus to its class label,
                        assuming that the mapping is in blocks, e.g., 0-k class 1,
                        k+1:2k+1 class 2.
            -stim_start/end: start/end time for computing the stimulus response.
            -baseline_start/end: start/end time for computing the baseline response.
            -sort_cat: sorting the neurons based on selectivity on the selected category.
            -category_decoding: decoding the category instead of stimulus.
            -k_fold: number of folds.
            -num_select_neuron: if not None, select the first N neuron with high selectivity.
            -classifier: classifer for decoding, e.g., logistic or svc.
            -reg_para: regularization parameter for the classifier.
            -cat_subset: if not None, only work on a subset of samples from the specified categories (e.g., face).
            -seed: seed for reproducibility.
    """
    np.random.seed(seed=seed)
    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]
    
    avg_stim_response = np.zeros([len(cell_idx), len(class_mapping)])
    avg_baseline_response= np.zeros([len(cell_idx), len(class_mapping)])
    decoding_response = dict()
    stim_count = np.zeros([len(class_mapping),])

    for trial_idx in range(len(trial_data['Paradigm']['PassiveFixation']['Number'])):
        # drop invalid trials
        if np.isnan(trial_data['Paradigm']['PassiveFixation']['End_Correct'][trial_idx]):
            continue

        stim_id = trial_data['Paradigm']['PassiveFixation']['Stimulus'][trial_idx]
        stim_count[stim_id] += 1
        trial_onset = trial_data['Paradigm']['PassiveFixation']['Start_Align'][trial_idx]

        # compute the stim/baseline average firing rate for each neuron/repetition
        tmp_neuron_response = []
        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PassiveFixation']['Spike'][trial_idx][neuron]]

            stim_spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time-trial_onset>=stim_start and spike_time-trial_onset<=stim_end])
            stim_firing_rate = stim_spike_count/(stim_end-stim_start)
            avg_stim_response[neuron_idx, stim_id] += stim_firing_rate
            tmp_neuron_response.append(stim_firing_rate)

            baseline_spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time-trial_onset>=baseline_start and spike_time-trial_onset<=baseline_end])
            baseline_firing_rate = baseline_spike_count/(baseline_end-baseline_start)
            avg_baseline_response[neuron_idx, stim_id] += baseline_firing_rate
        
        if stim_id not in decoding_response:
            decoding_response[stim_id] = []
        decoding_response[stim_id].append(np.array(tmp_neuron_response))

    # select the top-k most selective neurons
    if num_select_neuron is not None:
        # normalize by repetitions
        for i in range(len(stim_count)):
            avg_stim_response[:, i] /= stim_count[i]
            avg_baseline_response[:, i] /= stim_count[i]

        norm_response = normalize_response(avg_stim_response, avg_baseline_response)    

        select_stim = np.array([cur for cur in class_mapping if class_mapping[cur]==sort_cat])
        seletivity = []
        for i in range(len(norm_response)):
            seletivity.append(np.mean(norm_response[i, select_stim]))
        sort_idx = np.argsort(seletivity)[::-1]

        for k in decoding_response:
            for i in range(len(decoding_response[k])):
                # decoding_response[k][i] = decoding_response[k][i][sort_idx[:num_select_neuron]]
                decoding_response[k][i] = decoding_response[k][i][sort_idx[num_select_neuron:]]

    # collect feature/label for decoding
    feature, label = [], []
    category_mapping = {k: idx for (idx, k) in enumerate(np.unique(list(class_mapping.values())))}

    for k in decoding_response:
        if cat_subset is not None and class_mapping[k]!=cat_subset:
            continue
        for i in range(len(decoding_response[k])):
            feature.append(decoding_response[k][i])
            label.append(k if not category_decoding else category_mapping[class_mapping[k]])

    shuffle_idx = np.arange(len(feature))
    np.random.shuffle(shuffle_idx)
    feature = np.array(feature)[shuffle_idx]
    label = np.array(label)[shuffle_idx]

    if not cat_subset and not category_decoding:
        avg_acc = kfold_cross_validation(feature, label, k_fold, classifier, reg_para, 
                                        category_mapping=class_mapping)
    else:
        avg_acc = kfold_cross_validation(feature, label, k_fold, classifier, reg_para)

    return avg_acc

def template_matching(trial_data, class_mapping, stim_start=0.1, stim_end=0.3, 
                    cell_type='rML', num_select_neuron=None,
                    metric='euclidean', num_source=1, save_path=None,
                    vmin=0.4, vmax=0.9):
    """ Compute the distance between responses of different stimuli (trial-based)

        Inputs:
            -trial_data: processed trial data.
            -class_mapping: mapping from each stimulus to its class label,
                        assuming that the mapping is in blocks, e.g., 0-k class 1,
                        k+1:2k+1 class 2.
            -stim_start/end: start/end time for computing the stimulus response.
            -baseline_start/end: start/end time for computing the baseline response.
            -sort_cat: sorting the neurons based on selectivity on the selected category.
            -num_select_neuron: if not None, select the first N neuron with high selectivity.
            -metric: metric used to compute the distance.
            -num_source: number of samples to formulate the source vector (averaged)
            -save_path: path for saving the visualization
    """
    # get the index of selected cells
    cell_idx = [idx for idx in range(len(trial_data['Neuron_type'])) 
            if trial_data['Neuron_type'][idx]==cell_type]
    
    avg_stim_response = np.zeros([len(cell_idx), len(class_mapping)])
    stim_response = dict()
    stim_count = np.zeros([len(class_mapping),])

    for trial_idx in range(len(trial_data['Paradigm']['PassiveFixation']['Number'])):
        # drop invalid trials
        if np.isnan(trial_data['Paradigm']['PassiveFixation']['End_Correct'][trial_idx]):
            continue

        stim_id = trial_data['Paradigm']['PassiveFixation']['Stimulus'][trial_idx]
        stim_count[stim_id] += 1
        trial_onset = trial_data['Paradigm']['PassiveFixation']['Start_Align'][trial_idx]

        # compute the stim/baseline average firing rate for each neuron/repetition
        tmp_neuron_response = []
        for neuron_idx, neuron in enumerate(cell_idx):
            neuron_spikes = [cur for cur in trial_data['Paradigm']['PassiveFixation']['Spike'][trial_idx][neuron]]

            stim_spike_count = len([1 for spike_time in neuron_spikes 
                        if spike_time-trial_onset>=stim_start and spike_time-trial_onset<=stim_end])
            stim_firing_rate = stim_spike_count/(stim_end-stim_start)
            avg_stim_response[neuron_idx, stim_id] += stim_firing_rate
            tmp_neuron_response.append(stim_firing_rate)

        if stim_id not in stim_response:
            stim_response[stim_id] = []
        stim_response[stim_id].append(np.array(tmp_neuron_response))

    for i in range(len(stim_count)):
        avg_stim_response[:, i] /= stim_count[i]


    # compute the histogram of FSI
    plt.close('all')
    fig = plt.figure()
    avg_face = np.array([avg_stim_response[:, idx] for idx in range(len(class_mapping)) if class_mapping[idx]=='face'])
    avg_other = np.array([avg_stim_response[:, idx] for idx in range(len(class_mapping)) if class_mapping[idx]!='face'])
    fsi = FSI(avg_face.mean(0), avg_other.mean(0))
    if num_select_neuron is None:
        selected_fsi = fsi
    else:
        selected_fsi = fsi[np.argsort(fsi)[::-1][:num_select_neuron]]

    print('FSI mean %.3f, median %.3f, variance %.3f' %(np.mean(selected_fsi), np.median(selected_fsi), np.var(selected_fsi)))
    plt.hist(selected_fsi, bins=20, color='gray', edgecolor='black', alpha=0.7)
    plt.xlabel('FSI')
    plt.ylabel('Frequency')

    # Show the plot
    fig.savefig(os.path.join(save_path, 'fsi.pdf'), bbox_inches='tight')

    # select the top-k most selective neurons
    if num_select_neuron is not None:
        sort_idx = np.argsort(fsi)[::-1]

        for k in stim_response:
            for i in range(len(stim_response[k])):
                stim_response[k][i] = stim_response[k][i][sort_idx[:num_select_neuron]]

    # "trim" the data so that each stimulus has the same number of repetition
    trim_count = np.min([len(stim_response[k]) for k in stim_response])
    for k in stim_response:
        stim_response[k] = stim_response[k][:trim_count]

    # compute the stat for each cell for normalization
    cell_stat = []
    for k in stim_response:
        cell_stat.extend(stim_response[k])
    cell_stat = np.array(cell_stat)
    max_response, min_response = np.max(cell_stat, axis=0), np.min(cell_stat, axis=0)

    # compute the distance as a heatmap
    count = len(range(trim_count-num_source-1))
    heatmap = np.zeros([count, len(stim_response), len(stim_response)])

    for source_pointer in range(trim_count-num_source-1):
        source_idx = [source_pointer+idx for idx in range(num_source)]
        target_idx = [idx for idx in range(trim_count) if idx not in source_idx]
        for i in range(len(stim_response)):
            for j in range(len(stim_response)):
                source_vector = np.array([stim_response[i][idx] for idx in source_idx])
                source_vector = source_vector.mean(0) if num_source>1 else source_vector
                source_vector = (source_vector-min_response)/(max_response-min_response)
                target_vector = np.array([stim_response[j][idx] for idx in target_idx])
                target_vector = target_vector.mean(0) 
                target_vector = (target_vector-min_response)/(max_response-min_response)

                if metric == 'euclidean':
                    dist = np.linalg.norm(source_vector - target_vector)
                elif metric == 'cosine':
                    dist = 1-cosine_similarity(source_vector.reshape([1, -1]), 
                                                        target_vector.reshape([1, -1]))[0]
                else:
                    NotImplementedError('Selected distance not implemented')
                heatmap[source_pointer, i, j] = dist

    cat_count = {}
    for k in class_mapping:
        if class_mapping[k] not in cat_count:
            cat_count[class_mapping[k]] = 0
        cat_count[class_mapping[k]] += 1
    xticks = np.cumsum([cat_count[cur] for cur in cat_count])
    tick_label = [cur for cur in cat_count]

    plt.close('all')
    fig = plt.figure()
    if vmin is None and vmax is None:
        plt.imshow(heatmap.mean(0), cmap='viridis', origin='upper')
    else:
        plt.imshow(heatmap.mean(0), cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.xticks(ticks=xticks, labels=tick_label)
    plt.yticks(ticks=xticks, labels=tick_label)
    fig.savefig(os.path.join(save_path, 'template_matching.pdf'), bbox_inches='tight')

    # compute the decoding accuracy based on the distance
    stim_acc = {k: [] for k in range(len(class_mapping))} # stim decoding accuracy
    category_acc = {k: [] for k in range(len(class_mapping))} # category decoing accuracy
    for i in range(len(heatmap)):
        pred_cls = np.argmin(heatmap[i], axis=1)
        for idx in range(len(pred_cls)):
            stim_acc[idx].append(True if pred_cls[idx]==idx else False)
            category_acc[idx].append(True if class_mapping[pred_cls[idx]]==class_mapping[idx] else False)
    
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].bar(np.arange(len(stim_acc)), [np.mean(stim_acc[idx]) for idx in np.arange(len(stim_acc))], 
                color='navy')
    axes[0, 0].axhline(y=1/96, color='red', linestyle='-')
    axes[0, 0].set_title('Stimulus Decoding')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Stimulus ID')

    axes[0, 1].bar(np.arange(len(category_acc)), [np.mean(category_acc[idx]) for idx in np.arange(len(category_acc))], 
                color='navy')
    axes[0, 1].axhline(y=1/6, color='red', linestyle='-')
    axes[0, 1].set_title('Category Decoding')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xlabel('Stimulus ID')

    # average the results by category
    category = ['face', 'body', 'fruit', 'techno', 'hand', 'scrambled']

    stim_acc_avg = {k: [] for k in np.unique(list(class_mapping.values()))}
    for idx in stim_acc:
        stim_acc_avg[class_mapping[idx]].extend(stim_acc[idx])
    axes[1, 0].bar(category, [np.mean(stim_acc_avg[cat]) for cat in category], 
                color='navy', yerr=[np.std(stim_acc_avg[cat]) for cat in category])
    axes[1, 0].axhline(y=1/96, color='red', linestyle='-')
    axes[1, 0].set_title('Stimulus Decoding by Category')
    axes[1, 0].set_ylabel('Accuracy')

    category_acc_avg = {k: [] for k in np.unique(list(class_mapping.values()))}
    for idx in stim_acc:
        category_acc_avg[class_mapping[idx]].extend(category_acc[idx])
    axes[1, 1].bar(category, [np.mean(category_acc_avg[cat]) for cat in category], 
                color='navy', yerr=[np.std(category_acc_avg[cat]) for cat in category])
    axes[1, 1].axhline(y=1/6, color='red', linestyle='-')
    axes[1, 1].set_title('Category Decoding by Category')
    axes[1, 1].set_ylabel('Accuracy')
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            axes[i, j ].set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    fig.savefig(os.path.join(save_path, 'decoding.pdf'), bbox_inches='tight')



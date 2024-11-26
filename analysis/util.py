import numpy as np

def moving_average(data, k):
    """
    Perform moving average smoothing.

    Parameters:
        data (list or numpy array): The input data to be smoothed.
        k (int): Number of data points to include in the moving average (including the current point).

    Returns:
        numpy array: Smoothed data.
    """
    if k < 1:
        raise ValueError("k must be greater than or equal to 1.")
    
    data = np.array(data)  # Ensure input is a numpy array for easy manipulation
    window_size = k
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # To maintain the length of the original data, prepend with the original data
    # for the initial points where the window is incomplete
    padding = np.cumsum(data[:window_size-1]) / np.arange(1, window_size)
    return np.concatenate([padding, smoothed_data])

def compute_population_response(spike_time, start, end):
    """ Auxiliary function for computing the average firing rate
        for each neurons within the given period.

        Inputs:
            - spike_time: a nested list storing the time of each spike
                    for each neuron.
            - start: start of the selected time period.
            - end of the selected time period.
        
        Return:
            A (N, ) numpy array storing the average firing rate, where
                N is the number of neurons. 
    """

    num_neuron = len(spike_time)
    avg_firing = np.zeros([num_neuron, ])

    for neuron_idx in range(len(spike_time)):
        neuron_spikes = spike_time[neuron_idx]
        spike_count = len([1 for cur in neuron_spikes 
                    if cur>=start and cur<=end])
        avg_firing[neuron_idx] = spike_count/(end-start)
    
    return avg_firing

def normalize_response(stim_firing_rate, baseline_firing_rate):
    """ Normalize categorical responses for each neuron. The normalization process
        follows the lab convention.

        Inputs:
            - stim_firing_rate: firing rate for each unique stimulus/setting, 
                        assuming a NxC numpy array, where N/C corresponds to
                        neuron/stimulus.
            - baseline_firing_rate: baseline firing rate following the same convention.
    """
    norm_response = stim_firing_rate-baseline_firing_rate.mean(-1, keepdims=True)

    return (norm_response-norm_response.min(-1, keepdims=True))/(norm_response.max(-1, keepdims=True)-norm_response.min(-1, keepdims=True))

def check_eye_interaction(eye_interaction, period):
    """ Function for checking if the monkey is looking at any face during
        the given period.
    """

    for face in eye_interaction:
        for cur_block in eye_interaction[face]:
            if max(cur_block[0], period[0]) <= min(cur_block[1], period[1]):
                return True
    
    return False


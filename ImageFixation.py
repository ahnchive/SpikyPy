from MinosData import MinosData
import numpy as np
from matplotlib import pyplot as plt
import json
import argparse
import os
from PolyfaceUtil import align_trial, process_trial, get_stimulus_mapping, MinosEyeConversion
from FixationExtraction import extract_fixations
import cv2
from scipy.ndimage import gaussian_filter
import math

parser = argparse.ArgumentParser('Script for extracting eye fixation for passive fixation stimuli', add_help=False)
parser.add_argument('--trial_record', default=None, type=str, help='folder containing the recorded Minos data')
parser.add_argument('--save_path', default=None, type=str, help='folder for storing the visualization results')
parser.add_argument('--stim2id', default=None, type=str, help='Json file storing the mapping between Minos stimulus ininces and image names')
parser.add_argument('--save_dir', default=None, type=str, help='Directory for saving the results')
parser.add_argument('--vis_map', action='store_true', help='visualizing the saliency/fixation maps or not')
parser.add_argument('--img_dir', default=None, type=str, help='Directory for raw images')
parser.add_argument('--img_w', default=800, type=float, help='Image width')
parser.add_argument('--img_h', default=600, type=float, help='Image height')
parser.add_argument('--load_json', default=None, type=str, help='Pre-extracted data')

args = parser.parse_args()

def overlay_heatmap(img, att, cmap=plt.cm.jet):
    gamma = 1.0
    att = cv2.blur(att, (35, 35))
    colorized = cmap(np.uint8(att*255))
    alpha = 0.7
    overlaid = np.uint8(img*(1-alpha)+colorized[:, :, 2::-1]*255*alpha)
    return overlaid

if args.load_json is None:
    # create the mapping from ID to image name
    stim2id = json.load(open(args.stim2id))
    id2stim = dict()
    for k in stim2id:
        id2stim[stim2id[k]] = k.split('\\')[-1]

    trial_info = MinosData(os.path.join(args.trial_record, 'Passive Fixation', 'TrialInfo.bin')).Values
    trial_data = MinosData(os.path.join(args.trial_record, 'Passive Fixation', 'Trials.bin')).Values
    eye_data = MinosData(os.path.join(args.trial_record, 'Eye.bin')).Values

    # preprocess the data for better structure
    trial_data = process_trial(trial_data, 'Start_Align', 'End')
    trial2stim = get_stimulus_mapping(trial_info, id2stim)

    # iterate through all trial and extract the fixation 
    avg_num_fixation = []
    record_data = dict()
    for trial_num in trial_data:
        # align the eye data 
        start_idx, end_idx = align_trial(trial_num, eye_data, trial_data, 'Start_Align', 'End')
        eye_x = [float(eye_data['ConvergenceX'][idx]) for idx in range(start_idx, end_idx+1)]
        eye_y = [float(eye_data['ConvergenceY'][idx]) for idx in range(start_idx, end_idx+1)]
        eye_z = [float(eye_data['ConvergenceZ'][idx]) for idx in range(start_idx, end_idx+1)]
        eye_x, eye_y = MinosEyeConversion(eye_x, eye_y, eye_z, 50)

        # from 100 ns to ms
        eye_time = [float(eye_data['Timestamp'][idx])/1e+4 for idx in range(start_idx, end_idx+1)]
        tmp_eye = {'x': eye_x, 'y': eye_y, 'time': eye_time}
        fixation = extract_fixations(tmp_eye, 0.3, 0.12, 80) # originally 0.25, 0.1, 150
        record_data[trial2stim[trial_num]] = [[fix[0], fix[1]] for fix in fixation]
        avg_num_fixation.append(len(fixation))

    # basic stat
    print('Average number of fixations %.2f, STD: %.2f' %(np.mean(avg_num_fixation), np.std(avg_num_fixation)))

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'fixation.json'), 'w') as f:
        json.dump(record_data, f)
else:
    record_data = json.load(open(args.load_json))

# visualization
if args.vis_map:
    os.makedirs(os.path.join(args.save_dir, 'saliency_map'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'fixation_map'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'overlay_map'), exist_ok=True)
    for img in record_data:
        cur_img = cv2.imread(os.path.join(args.img_dir, img))
        cur_fix = record_data[img]
        fix_map = np.zeros([args.img_h, args.img_w])
        for (x, y) in cur_fix:
            if x<1 and x>=0 and y>=0 and y<1:
                y = int(y*args.img_h)
                x = int(x*args.img_w)
                fix_map[y, x] = 1
        
        cv2.imwrite(os.path.join(args.save_dir, 'fixation_map', img), fix_map*255)

        # convert fixation map to saliency maps
        sal_map = gaussian_filter(fix_map, sigma=50)
        sal_map /= sal_map.max()
        cv2.imwrite(os.path.join(args.save_dir, 'saliency_map', img), sal_map*255)

        # overlay map
        overlay_img = overlay_heatmap(cur_img, sal_map)
        cv2.imwrite(os.path.join(args.save_dir, 'overlay_map', img), overlay_img)



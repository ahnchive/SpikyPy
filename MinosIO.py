from MinosData import MinosData
import numpy as np
import os
from glob import glob
import json

class Loader():
    """ class object for loading all useful files
    from one Minos recording session. The session may
    include multiple experimental paradigms.
    """
    def __init__(self, session_root):
        """ Assuming the standardized structure
        of Minos recording data, where the information
        about different paradigms are stored in subfolders, with
        their names matching the paradigm names. All other files,
        e.g., eye data and player locations, are stored under the 
        root folder (i.e., session_root).
        """
        self.session_root = session_root
        self.paradigm = None

    def paradigms(self,):
        """ Show all available paradigms within the session.
        """
        self.all_paradigm = [os.path.basename(cur) for cur in 
                        glob(os.path.join(self.session_root, '*')) 
                        if os.path.isdir(cur)]
        return self.all_paradigm
    
    def useParadigm(self, paradigm_name):
        """ set the current paradigm.
        """
        if paradigm_name in self.all_paradigm:
            self.processTrialInfo(paradigm_name)
        else:
            raise ValueError("Invalid paradigm selected")
    
    def processTrialInfo(self, paradigm_name):
        """ Align the trial data with trial info, and save the
        processed data into a more structure dictionary. The dictionary
        is organized by trial numbers (as keys), note that all trials
        are included (except those incomplete ones) and corrrect/incorrect trials 
        can be differentiated based on the key "isCorrect". Each trial contains
        information about the timestamps for trial onset/offset, stimulus ID, 
        stimulus position, dot position, and stimulus size.
        """

        # organize the trial info as a dictionary
        trial_info = MinosData(os.path.join(self.session_root, 
                            paradigm_name, 'TrialInfo.bin'))
        processed_info = dict()
        info_type = [cur for cur in trial_info.Values 
                    if cur not in ['Timestamp', 'Number']]
        for i in range(len(trial_info.Values['Number'])):
            # only save a single copy for the target face for 
            # each polyface trail, the remaining information is available 
            # in the trial JSON file.
            if self.paradigm == 'Polyface Navigation' and trial_info.Values['Type'][i] != 'Retrieval_Correct':
                continue

            trial_num = trial_info.Values['Number'][i]
            processed_info[trial_num] = dict()
            for k in info_type:
                processed_info[trial_num][k] = trial_info.Values[k][i]    

        # iterate through the trial data to obtain trial onset/offset timestamp, and its validity
        trial_data = MinosData(os.path.join(self.session_root, 
                            paradigm_name, 'Trials.bin'))
        processed_trial = dict()
        for i in range(len(trial_data.Values['Number'])):
            trial_num = trial_data.Values['Number'][i]
            if trial_num not in processed_trial:
                processed_trial[trial_num] = dict()
            processed_trial[trial_num][trial_data.Values['Event'][i]] = trial_data.Values['Timestamp'][i]
        
        # merge the two dictionary into final trial data
        self.paradigm = dict()
        for trial_num in processed_info:
            # filter out incomplete trials
            all_state = list(processed_trial[trial_num].keys())
            isValid = (([True if ('Start' in cur or 'End' in cur) else False for cur in all_state]).sum())>=2
            if not isValid:
                continue

            self.paradigm[trial_num] = processed_info[trial_num]
            isCorrect = True if 'End_Correct' in processed_trial[trial_num] else False
            self.paradigm[trial_num]['isCorrect'] = isCorrect

            # somehow the timestamps in trialinfo and trials binary files do not align
            # use the timestamp in trials binary file for now
            # TODO: check with Jialiang about integration with photodiode
            onState = 'Start_Align' if self.paradigm != 'Polyface Navigation' else 'Start'
            if self.paradigm != 'Polyface Navigation':
                offState = 'End' 
            else:
                if isCorrect:
                    offState = 'End_Correct'
                else:
                    offState = 'End_Wrong' if 'End_Wrong' in processed_trial[trial_num] else 'End_Miss'

            self.paradigm[trial_num]['stimOn'] = processed_trial[trial_num][onState]
            self.paradigm[trial_num]['stimOff'] = processed_trial[trial_num][offState]


    def currentParadigm(self,):
        """ Show the currently selected paradigm.
        """
        assert self.paradigm is not None, "Paradigm not set"
        return self.paradigm
    
    def stimOnsetMinos(self, correctOnly=False):
        """ Find the stimulus onset timestamp for each
            trial. 
        """
        
        # TODO: double-check if we want to use the raw timestamp or photodiode
        if correctOnly:
            onSetTime = [self.paradigm[cur]['stimOn'] for cur in
                         self.paradigm if self.paradigm[cur]['isCorrect']]
        else:
            onSetTime = [self.paradigm[cur]['stimOn'] for cur in self.paradigm]

        return onSetTime
    
    def synctimesMinos(self,):
        """ Sync timestamp with photodiode.
        """
        # read the sync time from binary file and save it as a dictionary (from internal trial number to time)
        syncData = MinosData(os.path.join(self.session_root, 'Sync.bin'))
        syncTime = dict()
        for i in range(len(syncData.Values['Number'])):
            syncTime[syncData.Values['Number'][i]] = syncData.Values['Timestamp'][i]
        return syncTime

    def mapMinosTimeToSyncedTarget(self, timesMinos, synctimesTarget):
        # TODO add support for syncing only data for selected internal trail numbers
        synctimesMinos = self.synctimesMinos()
        return np.interp(timesMinos, synctimesMinos, synctimesTarget)
    
    def stimfilenames(self, ):
        """ Returning the stimulus used in different trials. 
        """
        
        if self.paradigm != 'Polyface Navigator':
            # for passive fixation, return the Stimulus ID
            return [self.paradigm[cur]['Stimulus'] for cur in self.paradigm]
        else:
            # return indices for trial json file for polyface
            return [self.paradigm[cur]['Index'] for cur in self.paradigm]
    
    def usableblocks(self, name, ):
        pass


class TimeMachine:
    """ Time Machine for syncing time between Minos and openephys.
    """
    BNC_MINOSSYNC = 6
    def __init__(self, minos, oe, expt=1, rec=1):
        self.minos = minos
        self.oe = oe
        self.expt = expt
        self.rec = rec
        self.tsync = self.oe.events(self.oe.nidaqstream(), self.expt, self.rec)[TimeMachine.BNC_MINOSSYNC][:, 0]

    def minosTimeToOpenEphys(self, tt_minos, targetstream):
        tt_nidaq = self.minos.mapMinosTimeToSyncedTarget(tt_minos, self.tsync)
        if targetstream == self.oe.nidaqstream():
            return tt_nidaq
        else:
            return self.oe.shifttime(tt_nidaq, self.oe.nidaqstream(), targetstream, self.expt, self.rec)

    def minosPointProcessToOpenEphys(self, ttyy_minos, targetstream):
        """The function assumes that the data from Minos has been reorganized into KOFIKO format
        Specifically, first selecting a group of data (e.g., all eye tracking data), 
        and then restructure it into a dictionary with keys TimeStamp and Buffer (i.e., all data).
        """
        tt = np.asarray(ttyy_minos['TimeStamp']).flatten()
        yy = np.array(ttyy_minos['Buffer'])
        if len(tt)==1:
            if len(yy.shape): # y is a xN beast
                yy = yy.reshape(1, len(yy))
            else:
                yy = yy.flatten()
        tt_oe = self.minosTimeToOpenEphys(tt, targetstream)
        return tt_oe, yy

    def resamplePointProcess(self, ttyy_minos, targetstream, target_tt, missing=np.nan):
        tt, yy = self.minosPointProcessToOpenEphys(ttyy_minos, targetstream)
        ii = np.searchsorted(tt, target_tt, 'right')
        jj = ii - 1
        jj[jj<0] = 0
        zz = yy[jj]
        idx = np.nonzero(ii<1)[0]
        if len(idx):
            zz[idx] = missing
        return zz

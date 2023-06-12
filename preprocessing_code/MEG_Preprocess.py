from Subject import Subject
import pandas as pd
import matplotlib as mpl
import mne

'''#################################################'''
'''          INITIALIZE SUBJECT INSTANCES           '''
'''#################################################'''
subject_ID = 'R1873'
path = '../data_preprocessed/'
date = '6.23.22'
clip = '3'
subject = Subject(subject_ID, date, path, clip)

'''#################################################'''
'''                  INITIALIZE INDEX               '''
'''#################################################'''
low_pass = 40
high_pass = 1.0
filter_type = 'bandpass'
'''#################################################'''
'''                   Preprocess                    '''
'''#################################################'''
subject.load_raw()
subject.load_trial_info()
subject.find_events(stim_channels=['MISC 003', 'MISC 004', 'MISC 005'])
subject.remove_bad_channels()
subject.lsd()
subject.filter(filter_type, high_pass, low_pass)
# subject.resample()
subject.apply_ica()
subject.epoch()
# epoch_data = epoch.get_data()

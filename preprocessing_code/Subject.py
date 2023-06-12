'''#################################################'''
'''                     Imports                     '''
'''#################################################'''

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import mne
import autoreject
from PyQt5 import QtWidgets
import numpy as np
import os
import sys
import pickle
from mykit import least_square_reference
from scipy import io


class Dialog(QtWidgets.QDialog):
    def __init__(self, dinput):
        super(Dialog, self).__init__()
        self._output = None
        self.createFormGroupBox(dinput)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QtWidgets.QVBoxLayout(self)
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)

        self.setWindowTitle("Input Box")

    def createFormGroupBox(self, dinput):
        layout = QtWidgets.QFormLayout()
        self.linedit1 = QtWidgets.QLineEdit('')
        self.combox1 = QtWidgets.QComboBox()
        self.combox1.setToolTip('Choose Your Status')
        self.combox1.addItems(['0'])
        self.spinbox1 = QtWidgets.QSpinBox()

        for text, w in zip(dinput, (self.linedit1, self.combox1, self.spinbox1)):
            layout.addRow(text, w)

        self.formGroupBox = QtWidgets.QGroupBox("")
        self.formGroupBox.setLayout(layout)

    def accept(self):
        self._output = self.linedit1.text()  # , self.combox1.currentText(), self.spinbox1.value()
        super(Dialog, self).accept()

    def get_output(self):
        return self._output


def call_msg_box(title='ICA Component'):
    app = QtWidgets.QApplication(sys.argv)
    # Staring Functions for Execution
    dinput = [title]
    # Call the UI and get the inputs
    dialog = Dialog(dinput)
    if dialog.exec_() == Dialog.Accepted:
        ICAs = dialog.get_output()
        return ICAs


class Subject:

    def __init__(self, Subject_ID, Date, DIR=None, Clip = None):
        self.ica = None
        self.raw_resample = None
        self.raw_ica = None
        self.raw_filtered = None
        self.raw_lsd = None
        self.raw_ER = None
        self.raw_inter = None
        self.events = None
        self.trial_info = None
        self.raw = None
        self.save_plot = True
        self.dpi = 360
        # save plot into figure directory, default True
        '''
        Controls the number of PCA components from the pre-ICA PCA entering the ICA 
        decomposition in the ICA.fit() method. If None (default), all PCA components 
        will be used (== max_pca_components). If int, must be <= max_pca_components.
        If float, value between 0 and 1 to select n_components based on the percentage 
        of variance explained by the PCA components. 
        '''
        self.ica_method = 'fastica'
        # ica method to use
        self.ica_components = 20
        self.ica_decim = 3
        # save time to set decim bigger than 1
        self.n_max_ecg, n_max_eog = 3, 1
        # maximum number of components to reject
        self.low_pass = 40
        self.high_pass = 0.05
        self.filter_type = 'bandpass'
        self.down_rs = 100
        # choose filter type: either 'bandpass' for direct bandpass filter or 'hl' for highpass and lowpass filter in
        # order
        self.bad_channels = [f'MEG 0{m}' for m in [15, 79]]
        # some sensors that are always noisy
        self.tmin = -1
        self.tmax = 60
        # set epoch duration
        self.aud = 0
        # if using StimTracker sending pulses for actual auditory onsets
        self.baseline = (self.tmin, 0)
        # set baseline
        self.event_id = {'sound': 4}
        # set up event id

        if DIR is None:
            print("You must provide the data directory path.")
            return None

        '''
            Set up path and file names
        '''
        self.ID = Subject_ID
        self.DIR = DIR
        self.Date = Date
        self.Clip = Clip
        self.Name = f'{Subject_ID}_{Date}_{Clip}'
        self.DATA_DIR = f'{DIR}{Subject_ID}_{Date}/{Subject_ID}_data/'
        self.FIGURE_DIR = f'{DIR}{Subject_ID}_{Date}/{Subject_ID}_fig/'
        self.SAVE_DIR = f'{DIR}{Subject_ID}_{Date}/{Subject_ID}_save/'
        self.raw_file = f'{self.DATA_DIR}{self.Name}.sqd'
        self.empty_room_file = f'{self.DATA_DIR}{self.ID}_EmptyRoom_{Date}.sqd'
        self.ER_file = f'{self.SAVE_DIR}{self.Name}_em_raw.fif'
        self.trial_info_file = f'{self.SAVE_DIR}{self.Name}_trial_info.p'
        # saved trial info file
        self.event_file = f'{self.SAVE_DIR}{self.Name}_eve.fif'
        # self.event_mat_file = f'{self.SAVE_DIR}{self.Name}_eve.mat'
        self.event_path = f'{self.DATA_DIR}{self.Name}_trial_data.json'
        # event file
        self.resample_event_file = f'{self.SAVE_DIR}{self.Name}_resample_eve.fif'
        self.resample_event_mat_file = f'{self.SAVE_DIR}{self.Name}_resample_eve.mat'
        # event file
        self.raw_lsd_file = f'{self.SAVE_DIR}{self.Name}_lsd.fif'
        # saved LSD raw file
        self.raw_filt_file = f'{self.SAVE_DIR}{self.Name}_filt_{self.high_pass}_{self.low_pass}.fif'
        # saved filtered raw dataset
        self.raw_resample_file = f'{self.SAVE_DIR}{self.Name}_resample.fif'
        self.raw_ica_file = f'{self.SAVE_DIR}{self.Name}_ica.fif'
        self.raw_ica_mat = f'{self.SAVE_DIR}{self.Name}_ica.mat'
        # saved ica raw dataset
        self.raw_interpolate_file = f'{self.SAVE_DIR}{self.Name}_interpolate_raw.fif'
        # saved interpolated raw dataset
        self.epoch_file = f'{self.SAVE_DIR}{self.Name}_epo.fif'
        self.epoch_path = f'{self.SAVE_DIR}{self.Name}_epo.npy'
        # saved epoch dataset
        self.epoch_mat_path = f'{self.SAVE_DIR}{self.Name}_epoch_data.mat'

        ## for headshape infos:
        self.headshape1_file = f'{self.DATA_DIR}{Subject_ID}_{Date}.hsp'
        self.headshape2_file = f'{self.DATA_DIR}{Subject_ID}_{Date}.elp'
        self.markers1_file = f'{self.DATA_DIR}{Subject_ID}_{Date}-1.mrk'
        self.markers2_file = f'{self.DATA_DIR}{Subject_ID}_{Date}-2.mrk'

        # KIT files
        self.KIT_layout = mne.channels.layout.read_layout(f'KIT-157_new3.lout')
        self.KIT_selection = f'KIT.sel'

    def construct_event_dic(self):

        return

    def load_raw(self):
        self.raw = mne.io.read_raw_kit(self.raw_file, preload=True, hsp=self.headshape1_file, elp=self.headshape2_file,
                                       mrk=self.markers2_file)
        return self.raw

    def load_trial_info(self):
        if os.path.isfile(self.trial_info_file):
            self.trial_info = pickle.load(open(self.trial_info_file, 'rb'))
        # load trial info
        else:
            self.trial_info = {}
        return self.trial_info

    def find_events(self, stim_channels=None):
        if os.path.isfile(self.event_file):
            self.events = mne.read_events(self.event_file)

        # read events from event file
        else:
            self.events = [None] * len(stim_channels)
            i = 0
            for channel in stim_channels:
                self.events[i] = mne.find_events(self.raw, stim_channel=channel, min_duration=0.005)
                i += 1
            self.events = np.concatenate(self.events)
            self.events = np.sort(self.events, 0)
            # io.savemat(self.event_mat_file, {'events': self.events})

            pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))
            # save events into trial info
            mne.write_events(self.event_file, self.events)
        return self.events

    def remove_bad_channels(self):
        if os.path.isfile(self.raw_interpolate_file):
            self.raw_inter = mne.io.read_raw_fif(self.raw_interpolate_file, preload=True)
        # read interpolated data
        else:
            self.raw_inter = self.raw.copy()
            # raw_ER = mne.io.read_raw_kit(empty_room_file, preload=True)
            # load empty room data for interpolation
            self.raw_ER = mne.io.read_raw_kit(self.empty_room_file, preload=True, hsp=self.headshape1_file,
                                              elp=self.headshape2_file,
                                              mrk=self.markers2_file)

            self.raw_inter.pick_types(meg=True, ref_meg=True)
            self.raw_ER.pick_types(meg=True, ref_meg=True)
            # pick channel type

            # bad_channels = list(set(bad_channels + ransac.bad_chs_))
            # #bad channels = predetermined set + autoreject detected set

            self.raw_inter.info['bads'] = self.bad_channels
            self.raw_ER.info['bads'] = self.bad_channels
            # set bad channels
            print(f'Bad channels in total: {self.bad_channels}, number = {len(self.bad_channels)}')
            # check bad channels
            self.trial_info['bads'] = self.raw_inter.info['bads']
            # save the bad channel info
            self.raw_ER.info['bads'] = self.raw_inter.info['bads']
            # set bad channels consistent
            print(f'Bad channels final: {self.trial_info["bads"]}, number = {len(self.trial_info["bads"])}')
            # show the final decision of bad channels
            # raw_inter.plot_sensors(ch_type='mag',show_names=1, block=1)
            # show the bad channel in 2D topomap
            pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))

            # save interpolated channels
            self.raw_inter.interpolate_bads(reset_bads=True)
            self.raw_ER.interpolate_bads(reset_bads=True)

            # interpolate bads and reset so that we have same number of channels for all blocks/subjects
            self.raw_inter.save(self.raw_interpolate_file, overwrite=True)
            self.raw_ER.save(self.ER_file, overwrite=True)
        return None

    def lsd(self):
        if os.path.isfile(self.raw_lsd_file):
            self.raw_lsd = mne.io.read_raw_fif(self.raw_lsd_file, preload=True)
        # read denoised data
        else:

            self.raw_lsd = least_square_reference(self.raw_interpolate_file, self.ER_file)
            blocks = []

            print("Applying Least Square Denoising. New data stored in self.raw")
            # apply LSD
            self.raw_lsd.pick_types(meg=True, misc=False)
            # get meg channel only since information in MISC channel is lost after LSD
            self.raw_lsd.save(self.raw_lsd_file, overwrite=True)
            # save denoised raw data
        return self.raw_lsd

    def filter(self, filter_type=None, high_pass=None, low_pass=None):
        if os.path.isfile(self.raw_filt_file):
            self.raw_filtered = mne.io.read_raw_fif(self.raw_filt_file, preload=True)
        # read pre-filtered data
        else:
            if self.filter_type == 'bandpass':
                self.raw_filtered = self.raw_lsd.filter(l_freq=high_pass, h_freq=low_pass, n_jobs=2)
            # apply band pass filter
            else:
                # apply high & low pass filter in order
                self.raw_filtered = self.raw_lsd.filter(l_freq=high_pass, h_freq=None, n_jobs=2)
                # highpass filter
                self.raw_filtered = self.raw_lsd.filter(l_freq=None, h_freq=low_pass, n_jobs=2)
            # lowpass filter
            psd_before_filter = self.raw_lsd.plot_psd(area_mode='range', average=False, show=False)
            psd_after_filter = self.raw_filtered.plot_psd(area_mode='range', average=False, fmax=60.)
            # plot psd before&after filter to check
            self.trial_info['filter'] = [high_pass, low_pass, filter_type]
            pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))
            # save filter
            # raw_filtered.add_channels([sac_channel,blk_channel])
            # raw_filtered.add_channels([eog_channel])
            # add back eye tracking channel

            self.raw_filtered.save(self.raw_filt_file, overwrite=True)
            # save filtered raw dataset
            if self.save_plot:
                psd_before_filter.savefig(f'{self.FIGURE_DIR}psd_before_filter_{self.Clip}.svg', dpi=self.dpi)
                psd_after_filter.savefig(f'{self.FIGURE_DIR}psd_after_filter_{self.Clip}.svg', dpi=self.dpi)
            # save psd plots
            else:
                pass
        return None

    def resample(self):
        if os.path.isfile(self.raw_resample_file):
            self.raw_resample = mne.io.read_raw_fif(self.raw_resample_file, preload=True)
        # read pre-ica data
        else:
            self.raw_resample = self.raw_filtered.resample(self.down_rs, npad="auto", events=self.events)
            self.raw_resample[0].save(self.raw_resample_file, overwrite=True)

            self.events = self.raw_resample[1]
            io.savemat(self.resample_event_mat_file, {'events': self.events})

            pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))
            # save events into trial info
            mne.write_events(self.resample_event_file, self.events)
            self.raw_resample = self.raw_resample[0]

    def apply_ica(self):
        if os.path.isfile(self.raw_ica_file):
            self.raw_ica = mne.io.read_raw_fif(self.raw_ica_file, preload=True)
        # read pre-ica data
        else:
            if os.path.isfile(self.raw_resample_file):
                ica_base = self.raw_resample
            else:
                ica_base = self.raw_filtered
            self.ica = mne.preprocessing.ICA(n_components=self.ica_components, method=self.ica_method, random_state=0)
            # generate ica
            self.ica.fit(ica_base, picks='meg', decim=self.ica_decim)
            # fit ica to dataset

            ica_components_plot = self.ica.plot_components(show=True)
            # plot all components
            self.ica.exclude = [int(i) for i in call_msg_box().split(',')]

            # exclude components
            ica_source_plot = self.ica.plot_sources(ica_base)
            # plot continuous ICA component data
            ica_porperty = self.ica.plot_properties(ica_base, picks=self.ica.exclude)
            # plot all components with property, take couple minutes, use with patience

            self.trial_info['ica'] = self.ica.exclude
            pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))
            # save ica component into trial info
            ica_overlay = self.ica.plot_overlay(ica_base)
            # check ica result before&after artifacts rejection

            self.raw_ica = ica_base.copy()
            self.ica.apply(self.raw_ica)
            # apply ica to raw data
            self.raw_ica.pick_types(meg=True, eog=False)
            # get rid of eog channel
            self.raw_ica.save(self.raw_ica_file, overwrite=True)
            # save raw dataset after ICA

            # io.savemat(self.raw_ica_mat, dict(data=self.raw_ica.get_data()), oned_as='row')
            if self.save_plot:
                for ind, fig in enumerate(ica_components_plot): fig.savefig(f'{self.FIGURE_DIR}ica_components{ind}_{self.Clip}.svg',
                                                                            dpi=self.dpi)
                ica_overlay.savefig(f'{self.FIGURE_DIR}ica_overlay_{self.Clip}.png', dpi=self.dpi)
            # save ica plots
            else:
                pass

        return self.raw_ica

    def epoch(self):
        if os.path.isfile(self.epoch_file):

            epoch = mne.read_epochs(self.epoch_file)
            np.save(self.epoch_path, epoch.get_data())

        # read pre-existed epoch data

        else:
            # mne.viz.plot_raw(self.raw_ica, block=True)
            # epoch = mne.Epochs(self.raw_ica, self.events, event_id=self.event_id, tmin=self.tmin, tmax=self.tmax,
            #                    proj=False, baseline=None, preload=True, reject=None, flat=None)

            epoch = mne.Epochs(self.raw_ica, self.events, event_id=None, tmin=- 0.2, tmax=95, baseline=(None, 0),
                               picks=None, preload=True,
                               reject=None, flat=None, proj=False, decim=1, reject_tmin=None, reject_tmax=None,
                               detrend=None,
                               on_missing='raise', reject_by_annotation=None, metadata=None, event_repeated='merge',
                               verbose=None)


            '''

            epoch = mne.Epochs(self.raw_ica, self.events, event_id=self.event_id, tmin=self.tmin, tmax=self.tmax,
                   proj=False, baseline=None, preload=True, reject=dict(mag=4000), reject_tmin=0.1, reject_tmax=0.2, flat=None)

            raw: Raw object
            events: array of int, shape (n_events, 3)
            event_id: The id of the event to consider.
            tmin: Start time before event.
            tmax: End time after event.
            baseline: The time interval to apply baseline correction.
            preload: Load all epochs from disk when creating the object
                     or wait before accessing each epoch (more memory efficient
                     but can be slower).
            reject: Rejection parameters based on peak-to-peak amplitude.
            '''
            # picks = mne.pick_types(epoch.info, meg=True, eog=False)
            # # pick channel types
            # # ar = autoreject.AutoReject(picks=picks, n_jobs=4)
            # # # generate autoreject object
            # # reject = autoreject.get_rejection_threshold(epoch)
            # # # get autoreject threshold
            # # print(reject)
            # # epoch.drop_bad(reject=reject)
            # # reject accroding to the threshold
            # epoch.plot(block=False, n_channels=40)
            # # manul drop bad epoch by clicking
            # self.trial_info['drop_log'] = [i for i in epoch.drop_log if i != ['IGNORED'] and i != []]
            # pickle.dump(self.trial_info, open(self.trial_info_file, 'wb'))
            # save reject info into trial info
            epoch = epoch.resample(self.down_rs,npad="auto")
            epoch.save(self.epoch_file, overwrite=True)
            epoch.plot_drop_log(ignore=[])
        # save epoch dataset

        # epoch_data = epoch.get_data()
        # io.savemat(self.epoch_mat_path, dict(epoch_data=epoch_data), oned_as='row')

        # # plots each condition butterfly and M1 topography
        # for condition in self.event_id.keys():
        #     evoked = epoch[condition].average()
        #     # average each condition epoch
        #     selection = mne.read_vectorview_selection('temporal', fname=self.KIT_selection, info=epoch.info)
        #     # select temporal lobe to pick up auditory cortex response
        #     butterfly = evoked.plot(picks=selection, gfp=True, time_unit='ms', window_title=f'Condition:{condition}')
        #     # plot butterfly&GFP
        #     topomap = evoked.plot_topomap(times=[0, 0.05, 0.1, 0.17], time_unit='ms', ch_type='mag',
        #                                   average=0.025, colorbar=True)
        #     # plot topography
        #     joint = evoked.plot_joint(times=[0, 0.08, 0.15])
        #     # plot joint graph
        #     if self.save_plot:
        #         butterfly.savefig(f'{self.FIGURE_DIR}{condition}_butterfly.svg', dpi=self.dpi)
        #         topomap.savefig(f'{self.FIGURE_DIR}{condition}_topomap.svg', dpi=self.dpi)
        #         joint.savefig(f'{self.FIGURE_DIR}{condition}_joint.svg', dpi=self.dpi)
        #     # save evoked plots
        #     else:
        #         pass

        return epoch

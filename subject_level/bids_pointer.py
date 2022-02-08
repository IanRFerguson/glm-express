#!/bin/python3

"""
About this Class

BIDSPointer is the foundation of the subject-level
analysis pipeline. The subsequent Subject object
inherits this class.

Ian Richard Ferguson | Stanford University
"""

"""
Usage Notes

* The TEMPLATE SPACE attribute is hard-coded ... override this with set_template_space()
"""

# ---- Imports
import os, json, pathlib, sys, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
from build_task_info import build_task_info


class BIDSPointer:
      
      # ---- Class Functions
      def __init__(self, sub_id, task, bids_root=os.path.join('./bids'), suppress=False):

            self.sub_id = sub_id                                                    #
            self.task = task                                                        #
            self.template_space = "MNILin6"                                         #
            self.conditions = []                                                    #

            #
            if os.path.exists(bids_root):
                  self.bids_root = bids_root
            else:
                  raise OSError(f'{bids_root} is invalid path ... your current directory: {os.getcwd()}')


            self.bids_container = self._set_container()                             #

            # -- BIDS Paths   
            self.raw_bold, self.events = self._split_raw_data()                     #
            self.preprocessed_bold, self.confounds = self._split_derived_data()     #
            self.first_level_ouptut = self._output_tree()                           #
            self.functional_runs = self._derive_functional_runs()                   #


            # -- User-defined task information
            task_file = self._validate_task_file()                                  #

            self.t_r = 1.                                                           #
            self.dummy_scans = 2                                                    #

            self.conditions = []                                                    #
            self.condition_variable = task_file['condition_identifier']             #
            self.confound_regressors = task_file['confound_regressors']             #
            self.auxilary_regressors = task_file['auxilary_regressors']             #
            self.contrasts = task_file['contrasts']                                 #

            #
            if not suppress:
                  print(str(self))


      def __str__(self):
            container = {'Subject ID': self.sub_id,
                        'Task': self.task,
                        "# of Functional Runs": len(self.derived_data),
                        "Output Directory": self.first_level_output,
                        "Defined Contrasts": self.contrasts}

            return json.dumps(container, indent=4)


      # ---- Setter helpers
      def set_tr(self, incoming):
            self.t_r = incoming


      def set_dummy_scans(self, incoming):
            self.dummy_scans = incoming


      def set_conditions(self, incoming):
            self.conditions = incoming


      def set_template_space(self, incoming):
            self.template_space = incoming


      # ---- Ecosystem helpers
      def _validate_task_file(self):
            """
            
            """

            target = os.path.join(self.bids_root, 'task_information.json')

            if not os.path.exists(target):
                  build_task_info(self.bids_root)

            with open(target) as incoming:
                  return json.load(incoming)[self.task]


      def _output_tree(self):
            """
            
            """

            primary = os.path.join(self.bids_root, 
                                  'derivatives/first-level-output/',
                                  f'sub-{self.sub_id}/task-{self.task}')

            for secondary in ['models', 'plots']:
                  for tertiary in ['condition-maps', 'contrast-maps']:
                        temp = os.path.join(primary, secondary, tertiary)

                        if not os.path.exists(temp):
                              pathlib.Path(temp).mkdir(parents=True, exist_ok=True)

            return primary


      # -- 
      def _all_raw_data(self):
            """
            TODO: Can we make this even more abstract? Such that the subject ID
            can be recursively located in the main bids root?
            """

            raw_file_path = [path for path in os.listdir(self.bids_root) if self.sub_id in path]
            raw_file_path = os.path.join(self.bids_root, raw_file_path, "func")

            target_extensions = ['.nii.gz', '.tsv']

            iso_files = [x for x in os.listdir(raw_file_path) if x in target_extensions if self.task in x]
            
            return [os.path.join(raw_file_path, x) for x in iso_files]


      def _split_raw_data(self):
            """
            
            """

            raw_data = self._all_raw_data()
            bold = [x for x in raw_data if '.nii.gz' in x]
            events = [x for x in raw_data if '.tsv' in x]

            return bold, events


      # --      
      def _all_derived_data(self):
            """
            TODO: Same shit as above
            """

            derivatives = os.path.join(self.bids_root, 'derivatives/fmriprep')
            derived_file_path = [path for path in os.listdir(derivatives) if self.sub_id in path]
            derived_file_path = os.path.join(derived_file_path, 'func')

            target_extensions = ['.nii.gz', '.tsv']

            iso_files = [x for x in os.listdir(derived_file_path) if x in target_extensions if self.task in x]
            
            return [os.path.join(derived_file_path, x) for x in iso_files]


      def _split_derived_data(self):
            """
            
            """

            derived = self._all_derived_data()
            bold = [x for x in derived if '.nii.gz' in x if self.template_space in x]
            confounds = [x for x in derived if '.tsv' in x]

            return bold, confounds


      # --
      def _iso_anatomical(self):
            pass


      def _iso_preprocessed(self):
            pass


      def _derive_functional_runs(self):
            return len(self.raw_bold)


      # ---- Utility helpers
      def _init_container(self):
            """
            Initializes container to store run-specific bold / event / confound info
            """
            
            # Empty dictionary to append into
            container = {}

            # E.g., 0, 1, 2
            for run in range(len(self.functional_runs)):

                  # 0 => run-1
                  key = f'run-{run+1}'

                  for filetype in ['func', 'event', 'confound']:
                        container[key][filetype] = ''

            return container


      def _build_container(self):
            """
            Populates container object with information
            """

            container = self._init_container()



            return container


      def _iso_events(self):
            pass


      def _iso_confounds(self):
            pass


      def load_events(self, run='ALL'):
            pass


      def load_confounds(self, run='ALL'):
            pass


      def _iso_run_number(string):
            temp = string.split('run-')[1].split('_')[0]

            return int(temp)
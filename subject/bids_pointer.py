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
* TR is hard-coded at 1. ... override this with set_tr()
* DUMMY SCANS are hard-coded at 2 ... override this with set_dummy_scans()
"""

# ---- Imports
from .build_task_info import build_task_info
import os, json, pathlib, sys, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
from bids import BIDSLayout


# ---- Object definition
class BIDSPointer:
      def __init__(self, sub_id, task, bids_root='./bids', suppress=False, template_space="MNI152NLin6",
                  dummy_scans=0, repetition_time=1.):
            """
            BIDSPointer constructor
            
            Parameters
                  sub_id: str or int | Subject ID corresponding to label in BIDS project
                  task: str | Task name corresponding to label in BIDS project
                  bids_root: str | Relative path to top of BIDS project directory tree
                  suppress: Boolean | if True, dictionary of attributes printed at init
                  template_space: str | Anatomical space derived from fmriprep (defaults to MNI152NLin6)
                  dummy_scans: int | Number of non-steady state volumes preceding functional runs
                  repetition_time: int | TR value derived from fMRI scanner
            """

            self.sub_id = sub_id                                                    # Unique subject ID from BIDS project
            self.task = task                                                        # Task name from BIDS project
            self.template_space = template_space                                    # Preprocessed template space
            self.bids_root = bids_root


            # === Validate BIDS input ===
            bids = BIDSLayout(self.bids_root)                                            

            if self.sub_id not in bids.get_subjects():
                  raise OSError(f'{self.sub_id} not found in BIDS project ... valid: {bids.get_subjects()}')

            if self.task not in bids.get_tasks():
                  raise OSError(f'{self.task} not found in BIDS project ... valid: {bids.get_tasks()}')


            # === BIDS Paths ===  
            self.raw_bold, self.events = self._split_raw_data()                     # Raw (unrprocessed NifTi and Events files)
            self.preprocessed_bold, self.confounds = self._split_derived_data()     # Preprocessed NifTi and confound regressor files
            self.first_level_output = self._output_tree()                           # Relative path for analysis output
            self.functional_runs = self._derive_functional_runs()                   # Length of functional runs for this task
            self.bids_container = self._build_container()                           # Container to organize all NifTi / events / confounds


            # === Object attributes ===
            task_file = self._validate_task_file()                                  # Build task_information.json file if it doesn't exist                             

            self.t_r = repetition_time                                              # User-defined repetition time
            self.dummy_scans = dummy_scans                                          # User-defined dummy scans

            self.conditions = []                                                    # Conditions list (empty @ __init__)
            self.condition_variable = task_file['condition_identifier']             # 
            self.confound_regressors = task_file['confound_regressors']             # List of confound regressors to include
            self.contrasts = task_file['design_contrasts']                          # Dictionary of contrasts to compute

            # Print subject information at __init__
            if not suppress:
                  print(str(self))


      def __str__(self):
            container = {'Subject ID': self.sub_id,
                        'Task': self.task,
                        "# of Functional Runs": self.functional_runs,
                        "Output Directory": self.first_level_output,
                        "Defined Contrasts": self.contrasts,
                        "Confound Regressors": self.confound_regressors}

            return json.dumps(container, indent=4)


      # ---- Setter methods
      def set_tr(self, incoming):
            self.t_r = incoming


      def set_dummy_scans(self, incoming):
            self.dummy_scans = incoming


      def set_conditions(self, incoming):
            self.conditions = incoming


      def set_template_space(self, incoming):
            self.template_space = incoming


      def set_confound_regressors(self, incoming):
            self.confound_regressors = incoming


      # ---- Ecosystem helpers
      def _validate_task_file(self):
            """
            Builds task_information.json if it doesn't already exist

            Returns
                  Task-specific parameters from JSON file
            """

            # Relative path to task_information file
            target = os.path.join('./task_information.json')

            # Build task file if it doesn't exist
            if not os.path.exists(target):
                  build_task_info(self.bids_root)

            # Open task file and return as dictionary
            with open(target) as incoming:
                  return json.load(incoming)[self.task]



      def _output_tree(self):
            """
            Builds out subject's directory hierarchy for first-level modeling

            Returns
                  Relative path to first-level output
            """

            # Base output directory
            primary = os.path.join(self.bids_root, 
                                  'derivatives/first-level-output/',
                                  f'sub-{self.sub_id}/task-{self.task}')

            # Build out subdirectories
            for secondary in ['models', 'plots']:
                  for tertiary in ['condition-maps', 'contrast-maps']:
                        temp = os.path.join(primary, secondary, tertiary)

                        if not os.path.exists(temp):
                              pathlib.Path(temp).mkdir(parents=True, exist_ok=True)

            return primary


      # ---- Isolate data paths
      def _split_raw_data(self):
            """
            Isolates relative paths to unprocessed BOLD and EVENTS files

            Returns
                  Two lists (path to BOLD files, path to Events files)
            """

            # Isolate path to subject data in BIDS project (not preprocessed)
            iso_path = [path for path in os.listdir(self.bids_root) if self.sub_id in path][0]
            
            # Path to functional subdirectory
            rel_path = os.path.join(self.bids_root, iso_path, 'func')

            # All files in subject's functional subdirectory
            raw_data = [os.path.join(rel_path, x) for x in os.listdir(rel_path) if self.task in x]
            
            
            # All NifTi files for the current task
            bold = [x for x in raw_data if '.nii.gz' in x]
            
            # All events files for the current task
            events = [x for x in raw_data if '.tsv' in x]

            return bold, events



      def _split_derived_data(self):
            """
            Isolates relative paths to preprocessed BOLD and CONFOUNDS files

            Returns
                  Two lists (path to BOLD files, path to Confounds files)
            """

            # Path to fmriprep derivatvies
            deriv_path = os.path.join(self.bids_root, 'derivatives/fmriprep')
            
            # Isolate path to subject's preprocessed data (excluding the useful HTML files created by fmriprep!)
            iso_path = [path for path in os.listdir(deriv_path) if self.sub_id in path if ".html" not in path][0]
            
            # Path to functional subdirectory in preprocessed directory
            rel_path = os.path.join(deriv_path, iso_path, 'func')

            
            # Preprocessed NifTi's must match template space and given naming conventions
            bold = [os.path.join(rel_path, x) for x in os.listdir(rel_path) if self.task in x 
                    if self.template_space in x 
                    if '.nii.gz' in x
                    if 'preproc' in x]

            # Confounds derived for each functional run
            confounds = [os.path.join(rel_path, x) for x in os.listdir(rel_path) if self.task in x if '.tsv' in x]

            return bold, confounds



      def _derive_functional_runs(self):
            return len(self.raw_bold)


      # ---- Utility helpers
      def _init_container(self):
            """
            Initializes container to store run-specific bold / event / confound info

            Returns
                  Empty BIDS Container, this is filled in the next step
            """
            
            # Empty dictionary to append into
            container = {}

            # E.g., 0, 1, 2
            for run in range(self.functional_runs):

                  # 0 => run-1
                  key = f'run-{run+1}'

                  container[key] = {}

                  for filetype in ['func', 'event', 'confound']:
                        container[key][filetype] = ''

            # These are empty lists at __init__
            container['all_func'] = []
            container['all_events'] = []
            container['all_confounds'] = []

            return container



      def _build_container(self):
            """
            Populates BIDS Container object with information

            Returns
                  BIDS Container ready for analysis
            """

            # Initialize container
            container = self._init_container()

            # Build runwise and aggregated lists 
            for run in range(self.functional_runs):
                  
                  key = f'run-{run+1}'

                  # Subject has multiple functional runs for the current task
                  if len(self.preprocessed_bold) > 1:
                        current_bold = [x for x in self.preprocessed_bold if key in x][0]
                        current_event = [x for x in self.events if key in x][0]
                        current_confound = [x for x in self.confounds if key in x][0]

                  # Subject has a single functional run for the current task
                  else:
                        try:
                              current_bold = [x for x in self.preprocessed_bold][0]

                        # Raise error if no functional runs are found - we assume this to be erroneous
                        except:
                              raise ValueError(f'No functional runs identified for task-{self.task} + template-space-{self.template_space}')
                        
                        
                        #
                        current_event = [x for x in self.events][0]
                        current_confound = [x for x in self.confounds][0]

                  """
                  Philosophy of this step ...

                  * We'll have run-wise keys (e.g., run-1, run-2, etc.) that will store relative paths to preprocessed BOLD maps, events, and confounds
                  * We'll additionally have format-wise keys (e.g., all_functional, all_events, all_confounds)
                  """

                  # Populate run-wise keys
                  container[key]['func'] = current_bold
                  container[key]['event'] = current_event
                  container[key]['confound'] = current_confound

                  # Append to format-wise keys
                  container['all_func'].append(current_bold)
                  container['all_events'].append(current_event)
                  container['all_confounds'].append(current_confound)

            # Save local BIDS container in subject's first-level-output directory
            with open(f'{self.first_level_output}/sub-{self.sub_id}_task-{self.task}_bids-container.json', 'w') as outgoing:
                  json.dump(container, outgoing, indent=5)

            return container



      def load_events(self, run='ALL'):
            """
            Loads events file (or files!) in a Pandas DataFrame

            Parameters
                  run: int or str | Defaults to ALL, or else enter a single run number

            Returns
                  Pandas DataFrame
            """
            
            if run != 'ALL':
                  # Isolate event file from all events
                  iso_event = [x for x in self.bids_container['all_events'] if f'run-{run}' in x]

                  # Read in event as Pandas DataFrame
                  temp_event = pd.read_csv(iso_event[0], sep='\t')

                  # Add run column if it doesn't exist
                  if 'run' not in temp_event.columns:
                        temp_event['run'] = [int(run)] * len(temp_event)

                  return temp_event

            else:
                  # Empty list to append into
                  all_events = []

                  for k in range(self.functional_runs):
                        
                        # Index 0 == run-1
                        run = k+1

                        # Read in event file as Pandas DataFrame
                        temp = self.bids_container[f'run-{run}']['event']
                        temp = pd.read_csv(temp, sep='\t')

                        # Add run column if it doesn't exist
                        if 'run' not in temp.columns:
                              temp['run'] = [run] * len(temp)

                        # Add temporary DataFrame to list
                        all_events.append(temp)

                  # Concatenate temporary frames into one DataFrame
                  agg_events = pd.concat(all_events).reset_index(drop=True)

                  return agg_events



      def load_confounds(self, run='ALL'):
            """
            Loads confounds into a Pandas DataFrame object

            Parameters
                  run: int or str | Defaults to ALL, or else enter a single run number

            Returns
                  Pandas DataFrame
            """

            if run != 'ALL':
                  # Isolate confounds TSV file
                  iso_confound = [x for x in self.bids_container['all_confounds'] if f'run-{run}' in x]

                  # Read file as Pandas DataFrame
                  temp_confound = pd.read_csv(iso_confound[0], sep='\t')

                  # Add run column if it doesn't exist
                  if run not in temp_confound.columns:
                        temp_confound['run'] = [int(run)] * len(temp_confound)

                  return temp_confound

            else:
                  # Empty list to append into
                  all_confounds = []

                  for k in range(self.functional_runs):
                        
                        # Index 0 == run-1
                        run = k+1

                        # Read in confound file as Pandas DataFrame
                        temp = self.bids_container[f'run-{run}']['confound']
                        temp = pd.read_csv(temp, sep='\t')

                        # Add run column if it doesn't exist
                        if 'run' not in temp.columns:
                              temp['run'] = [run] * len(temp)

                        # Add temporary DataFrame to list
                        all_confounds.append(temp)

                  # Concatenate temporary frames into one DataFrame
                  agg_confounds = pd.concat(all_confounds).reset_index(drop=True)

                  return agg_confounds
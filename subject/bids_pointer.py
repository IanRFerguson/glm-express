#!/bin/python3

"""
Usage Notes

* The TEMPLATE SPACE attribute is hard-coded ... override this with set_template_space()
* TR is hard-coded at 1. ... override this with set_tr()
* DUMMY SCANS are hard-coded at 2 ... override this with set_dummy_scans()
"""

# ---- Imports
from glm_express.build_info.build_task_info import build_task_info, build_dataset_description
import os, json, pathlib, glob
import pandas as pd
from bids import BIDSLayout


# ---- Object definition
class Build_Subject:
      def __init__(self, sub_id, task, bids_root='./bids', suppress=False, template_space="MNI152NLin2009c",
                  dummy_scans=0, repetition_time=1.):
            """
            BIDSPointer is the foundation of the subject-level analysis pipeline. 
            The subsequent Subject object inherits this class.
            
            Parameters
                  sub_id: str or int | Subject ID corresponding to label in BIDS project
                  task: str | Task name corresponding to label in BIDS project
                  bids_root: str | Relative path to top of BIDS project directory tree
                  suppress: Boolean | if True, dictionary of attributes printed at init
                  template_space: str | Anatomical space derived from fmriprep (defaults to MNI152NLin6)
                  dummy_scans: int | Number of non-steady state volumes preceding functional runs
                  repetition_time: int | TR value derived from fMRI scanner
            """

            self.sub_id = str(sub_id)                                               # Unique subject ID from BIDS project
            self.task = task                                                        # Task name from BIDS project
            self.template_space = template_space                                    # Preprocessed template space
            self.bids_root = bids_root

            # Build dataset description if it doesn't exist
            if not os.path.exists(os.path.join(self.bids_root, 'dataset_description.json')):
                  build_dataset_description(self.bids_root)

            # === Validate BIDS input ===
            bids = BIDSLayout(self.bids_root)

            if self.sub_id not in bids.get_subjects():
                  raise OSError(f'{self.sub_id} not found in BIDS project ... valid: {bids.get_subjects()}')

            if self.task not in bids.get_tasks():
                  raise OSError(f'{self.task} not found in BIDS project ... valid: {bids.get_tasks()}')


            # === BIDS Paths === 
            self.raw_bold = self._isolate_raw_data()                                # Paths to raw NifTi files
            self.events = self._isolate_events_files()                              # Paths to event files
            self.preprocessed_bold = self._isolate_preprocessed_data                # Paths to preprocessed NifTi files
            self.confounds = self._isolate_confounds_files()                        # Paths to confound regressor TSV files
            
            self.first_level_output = self._output_tree()                           # Relative path for analysis output
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
            if not type(incoming) == list:
                  raise TypeError(
                        'Provide this function with a list of condition trial types from your events file'
                  )
            
            self.conditions = incoming


      def set_template_space(self, incoming):
            self.template_space = incoming


      def set_confound_regressors(self, incoming):
            if not type(incoming) == list:
                  raise TypeError(
                        'Provide this function with a list of regressors derived from fmriprep'
                  )
            
            self.confound_regressors = incoming


      def set_design_contrasts(self, incoming):
            if not type(incoming) == dict:
                  raise TypeError(
                        'Design contrasts need to be a dictionary in {"labeled_name": "column1 - column2" format'
                  )

            self.contrasts = incoming


      # ---- Ecosystem helpers
      def _validate_task_file(self):
            """
            Builds task_information.json if it doesn't already exist

            Returns
                  Task-specific parameters from JSON file
            """

            path = pathlib.Path(self.bids_root).parents[0]

            # Relative path to task_information file
            target = os.path.join(path, "task_information.json")

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
      def _isolate_raw_data(self):
            """
            Returns list of NifTi files directly off the scanner
            """

            functional_path = os.path.join(self.bids_root,
                                           f"sub-{self.sub_id}",
                                           "func")

            pattern = os.path.join(functional_path, "**/*.nii.gz")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x]



      def _isolate_events_files(self):
            """
            Returns events files for current task
            """

            functional_path = os.path.join(self.bids_root,
                                           f"sub-{self.sub_id}",
                                           "func")

            pattern = os.path.join(functional_path, "**/*.tsv")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x]



      def _isolate_preprocessed_data(self):
            """
            Returns list of NifTi files derived from fmriprep
            """

            # BIDS data path (preprocessed)
            funtional_path = os.path.join(self.bids_root, 
                                          "derivatives/fmriprep",
                                          f"sub-{self.sub_id}", 
                                          "func")

            # Pattern to feed into glob generator
            pattern = os.path.join(funtional_path, "**/*.nii.gz")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x
                                                                  if self.template_space in x
                                                                  if "preproc_bold" in x]



      def _isolate_confounds_files(self):
            """
            Isolate relative paths to TSV files derived from fmriprep
            """

            # BIDS data path (preprocessed)
            functional_path = os.path.join(self.bids_root, 
                                          "derivatives/fmriprep",
                                          f"sub-{self.sub_id}", 
                                          "func")

            # Pattern to feed into glob generator
            pattern = os.path.join(functional_path, "**/*.tsv")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x
                                                                  if "confounds" in x]



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


            # Isolate files from BIDS project
            raw = self._isolate_raw_data()
            events = self._isolate_events_files()
            preprocessed = self._isolate_preprocessed_data()
            confounds = self._isolate_confounds_files

            # Loop through runs from BIDS project
            for ix in range(len(raw)):

                  # Index 0 == run-1
                  run_value = f"run-{ix + 1}"


                  # Isolate run-wise files
                  current_raw = [x for x in raw if run_value in x][0]
                  current_event = [x for x in events if run_value in x][0]
                  current_prep = [x for x in preprocessed if run_value in x][0]
                  current_confound = [x for x in confounds if run_value in x][0]


                  # Build out run-wise dictionary
                  container[run_value] = {
                        "raw_bold": current_raw,
                        "event": current_event,
                        "preprocessed_bold": current_prep,
                        "confounds": current_confound
                  }


                  # Append to dictionary (or create key if it doesn't exist)
                  try:
                        container["all_raw_bold"].append(current_raw)
                  except:
                        container["all_raw_bold"] = [current_raw]

                  try:
                        container["all_events"].append(current_event)
                  except:
                      container["all_events"] = [current_event]

                  try:
                      container["all_preprocessed_bold"].append(current_prep)
                  except:
                      container["all_preprocessed_bold"] = [current_prep]

                  try:
                        container["all_confounds"].append(current_confound)
                  except:
                        container["all_confounds"] = [current_confound]
                  
                  
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

                  # Isolate event file from all events if there are multiple runs
                  iso_event = [x for x in self.bids_container['all_events'] if f'run-{run}' in x]
                  
                  if len(iso_event) == 0:
                        # Isolate event file from all events if there is one run
                        iso_event = [x for x in self.bids_container['all_events']]
                        
                  # Read in Event file as Pandas DataFrame      
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

                  if len(iso_confound) == 0:
                        iso_confound = [x for x in self.bids_container['all_confounds']]

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
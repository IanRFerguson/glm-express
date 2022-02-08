#!/bin/python3

"""

"""


# ---- Imports + Ecoystem
import os, json, pathlib, sys, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
from nilearn.glm import first_level
import nilearn.plotting as nip
import matplotlib.pyplot as plt
from build_task_info import build_task_info


# ---- Init
class Subject:
      def __init__(self, sub_id, task, bids_root=os.path.join('./bids'), suppress=False):
            """
            
            """

            self.sub_id = sub_id                                              #
            self.task = task                                                  #
            
            if os.path.exists(bids_root):
                  self.bids_root = bids_root                                  #
            else:
                  raise OSError(f'{bids_root} does not exist')

            # --- Raw data
            self._all_raw_data = self._all_raw_data()                         #
            self.anat = self._iso_anatomical()                                #
            self.events = self._iso_events()                                  #

            # --- Preprocessed data
            self._all_preprocessed_data = self._all_preprocessed_data()       #
            self.preprocssed_bold_data = self._iso_preprocessed_functional_data()

            #
            self.first_level_output = os.path.join(bids_root, 
                                                  f"derivatives/first-level/sub-{self.sub_id}/task-{self.task}")

            #
            if not os.path.exists(self.first_level_output):
                  pathlib.Path(self.first_level_output).mkdir(parents=True, exist_ok=True)

            task_file = self._taskfile_validator()                            #
            output_paths = self._output_directories()                         #

            # --- 
            self.design_type = task_file['design_type']
            self.tr = task_file['tr']
            self.condition_variable = task_file['condition_identifier']
            self.confound_regressors = task_file['confound_regressors']
            self.auxilary_regressors = task_file['auxilary_regressors']

            self.model_output = output_paths['models']                        #
            self.plotting_output = output_paths['plotting']                   #

            if not suppress:
                  print(str(self))


      def __str__(self):
            """
            
            """

            container = {'Subject ID': self.sub_id,
                         'Task': self.task,
                         "# of Functional Runs": len(self.preprocssed_bold_data),
                         "Output Directory": self.first_level_output,
                         "Defined Contrasts": self.contrasts}

            return json.dumps(container, indent=4)

      # ----- Validators

      def _taskfile_validator(self):
            """
            
            """

            val = os.path.join('./task_information.json')

            if not os.path.exists(val):

                  build_task_info(self._bids_root)

                  sleep(0.5)
                  print("""
                  \ntask_information.json created! 
                  Fill in the field data in the JSON file 
                  and you're ready to run FANG!
                  """)

                  sys.exit(0)

            else:
                  with open('./task_information.json') as incoming:
                        return json.load(incoming)[self.task]


      def _output_directories(self):
            """
            
            """

            base = self.first_level_output

            output = {'models':[os.path.join(base, "model-maps")],
                     'plotting':[os.path.join(base, "plotting")]}

            for path in output['models']:
                  for reference in ['contrasts', 'conditions']:
                        pathlib.Path(os.path.join(path, reference)).mkdir(exist_ok=True, parents=True)

            for path in output['plotting']:
                  for reference in ['contrasts', 'conditions']:
                        for plot_type in ['glass', 'stat']:
                              pathlib.Path(os.path.join(path, reference, plot_type)).mkdir(exist_ok=True, parents=True)


      def _all_raw_data(self):
            """
            
            """

            container = []

            for roots, dirs, files in os.walk(os.path.join(self._bids_root, f"sub-{self.sub_id}")):
                  for name in files:
                        container.append(os.path.join(roots, name))

                  for name in dirs:
                        container.append(os.path.join(roots, name))

            return container


      def _iso_anatomical(self):
            """
            
            """

            return [x for x in self._all_raw_data if "T1W" in x if ".nii.gz" in x]


      def _all_preprocessed_data(self):
            """
            
            """

            continaer = []

            for roots, dirs, files in os.walk(os.path.join(self._bids_root, f"derivatives/fmriprep/sub-{self.sub_id}")):
                  for name in files:
                        continaer.append(os.path.join(roots, name))

                  for name in dirs:
                        continaer.append(os.path.join(roots, name))

            return continaer


      def _iso_preprocessed_functional_data(self):
            """
            
            """

            bold_data = [x for x in self._all_preprocessed_data if self.task in x if '.nii.gz' in x]

            if len(bold_data) == 0:
                  warnings.warn(f"""
                  Warning: No preprocessed NifTi files for {self.task} identified ... 
                  Raw subject data will be used by default
                  """)

                  bold_data = [x for x in self._all_raw_data if self.task in x if '.nii.gz' in x]

            return bold_data


      # ----- Utility functions


      def isolate_run(X):
            """
            
            """

            for node in X.split('_'):
                  if 'run-' in node:
                        y = node.split('-')[1]
                        return int(y)


      def plot_anatomical(self, dim=1.65, threshold=5., inline=True):
            """
            
            """

            k = nip.plot_anat(self.anat[0], 
                              threshold=threshold,
                              dim=dim, 
                              title=f"{self.sub_id}_anatomical",
                              draw_cross=False)

            if inline:
                  plt.show()
            else:
                  plt.savefig(os.path.join(self.first_level_output, f"sub-{self.sub_id}_anatomical.png"))


      def _iso_events(self):
            """
            
            """

            events = [x for x in self._all_raw_data if self.task in x if ".tsv" in x]

            if len(events) == 0:
                  raise OSError(f"No {self.task} events TSVs found in raw subject data")

            return events


      def load_events(self, run='ALL'):
            pass


      def _iso_confounds(self):
            """
            
            """

            confounds = [x for x in self._all_preprocessed_data if self.task in x if "confounds" in x if ".tsv" in x]

            if len(confounds) == 0:
                  warnings.warn(f"""
                  Warning: No confound timeseries for {self.task} identified ... 
                  Double-check your fmriprep output. Please note that all defined confound regressors will be ignored
                  """)

            return confounds



      def load_confounds(self, run='All'):
            pass


      # ----- Analysis functions

#!/bin/python3

"""
About this Class

RestingState is optimized to model functional and structural connectivity
in a preprocessed resting state fMRI scan.

Ian Richard Ferguson | Stanford University
"""


# --- Imports
import os, glob, json, pathlib
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout


# --- Object definition
class RestingState:

      # -- Constructor
      def __init__(self, sub_id, task="rest", bids_root="./bids", suppress=False, template_space="MNI152NLin2009"):
            """
            RestingState Constructor
            """

            # -- Object attributes
            self.sub_id = str(sub_id)
            self.task = task
            self.bids_root = self.set_bids_root(bids_root)
            self.template_space = self.set_template_space(template_space)
            self.output_path = self._build_output_path()


            # Verify SubID
            if not self.sub_id in list(BIDSLayout(self.bids_root).get_subjects()):
                  raise ValueError(f"ERROR: sub-{self.sub_id} not found in BIDS project")


            # -- RS data paths
            self.raw_data = self._get_raw_data()
            self.preprocessed_data = self._get_preprocessed_data()
            self.confounds = self._get_confound_regressors()

            self.bids_container = self._build_bids_container()



      # -- Ecosystem helpers
      def _path_to_raw(self):
            return os.path.join(self.bids_root, f"sub-{self.sub_id}", "func")


      def _path_to_derivatives(self):
            return os.path.join(self.bids_root, "derivatives/fmriprep", f"sub-{self.sub_id}", "func")


      def _get_raw_data(self):
            path = self._path_to_raw()

            return [x for x in glob.glob(path, recursive=True) if f"task-{self.task}" in x
                                                               if "nii.gz" in x]


      def _get_preprocessed_data(self):
            path = self._path_to_derivatives()

            return [x for x in glob.glob(path, recursive=True) if f"task-{self.task}" in x
                                                               if ".nii.gz" in x
                                                               if self.template_space in x]


      def _get_confound_regressors(self):
            path = self._path_to_derivatives()

            return [x for x in glob.glob(path, recursive=True) if f"task-{self.task}" in x
                                                               if "confounds" in x
                                                               if ".tsv" in x]


      def _build_bids_container(self):

            def isolate_run(x):
                  return int(x.split("run-")[1].split("_")[0])

            
            output = {"all_raw": self.raw_data,
                     "all_preprocessed": self.preprocessed_data,
                     "all_confounds": self.confounds}


            for k in [output["all_preprocessed"], output["all_confounds"]]:
                  for m in k:
                        run_val = f"run-{isolate_run(m)}"

                        if not run_val in output.keys():
                              output[run_val] = [m]
                        else:
                              output[run_val].append(m)


            filename = f"sub-{self.sub_id}_task-{self.task}_bids-container.json"

            with open(os.path.join(self.output_path, filename), "w") as log:
                  json.dump(output, log, indent=5)

            
            return output


      def _build_output_path(self):
            path = os.path.join(self.bids_root, "derivatives/first-level-output", 
                                f"sub-{self.sub_id}", f"task-{self.task}")

            for subdir in ["models", "plots"]:
                  temp = os.path.join(path, subdir)

                  if not os.path.exists(temp):
                        pathlib.Path(temp).mkdir(parents=True, exist_ok=True)


      # -- Set methods
      def set_bids_root(self, incoming):

            #
            if not os.path.exists(incoming):
                  print(f"\nERROR: {incoming} is an invalid path\n")
           
            #
            else:
                  self.bids_root = incoming


      def set_template_space(self, incoming):

            #
            valid_spaces = ["MNI152NLin2009", "MNI152NLin6"]

            #
            if not incoming in valid_spaces:
                  print(f"\nERROR: {incoming} is an invalid template space ... {valid_spaces} are valid\n")
            
            #
            else:
                  self.template_space = incoming

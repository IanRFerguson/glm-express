#!/bin/python3

"""
"""

# --- Imports
from glm_express.build_info.build_task_info import build_task_info, build_dataset_description
import os, json, glob, pathlib
import pandas as pd
from bids import BIDSLayout

# --- Object definition
class Build_RS:

      def __init__(self, sub_id, task="rest", bids_root="./bids", suppress=False, 
                   template_space="MNI152NLin2009c"):
            
            self.sub_id = sub_id
            self.task = task
            self.bids_root = bids_root
            self.template_space = template_space

            # Build dataset description if it doesn't exist
            if not os.path.exists(os.path.join(self.bids_root, "dataset_description.json")):
                  build_dataset_description(self.bids_root)


            # === Validate BIDS input ===
            bids = BIDSLayout(self.bids_root)

            if self.sub_id not in bids.get_subjects():
                  raise OSError(f"{self.sub_id} not found in BIDS project")

            if self.task not in bids.get_tasks():
                  raise OSError(f"{self.task} not found in BIDS project")


            # === BIDS Paths ===
            self.raw_bold = self._isolate_raw_data()
            self.preprocessed_runs = self._isolate_preprocessed_data()
            self.first_level_output = self._output_tree()
            self.bids_container = self._build_bids_container()

            
            # === Modeling attributes ===
            task_file = self._validate_taskfile()
            self.confound_regressor_names = task_file["confound_regressors"]

            if not suppress:
                  print(str(self))


      def __str__(self):
            container = {
                  "Subject ID": self.sub_id,
                  "Task": self.task,
                  "# of Resting State Runs": len(self.preprocessed_runs),
                  "Confound Regressors": self.confound_regressor_names
            }

            return json.dumps(container, indent=4)


      # -- Setters



      # -- Getters


      # --- Ecosystem helpers
      def _validate_taskfile(self):
            """
            
            """

            path = pathlib.Path(self.bids_root).parents[0]
            target = os.path.join(path, "task_information.json")

            if not os.path.exists(target):
                  build_task_info(self.bids_root)

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
                  temp = os.path.join(primary, secondary)

                  if not os.path.exists(temp):
                        pathlib.Path(temp).mkdir(parents=True, exist_ok=True)

            return primary


      def _isolate_raw_data(self):
            """
            Isolates relative paths to unprocessed BOLD files

            Returns
                  List of relative paths
            """

            functional_path = os.path.join(self.bids_root, 
                                           f"sub-{self.sub_id}", 
                                           "func")

            pattern = os.path.join(functional_path, "**/*.nii.gz")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x]


      def _isolate_preprocessed_data(self):
            """
            Isolates relative paths to preprocessed BOLD files

            Returns
                  List of relative paths
            """

            funtional_path = os.path.join(self.bids_root, 
                                          "derivatives/fmriprep",
                                          f"sub-{self.sub_id}", 
                                          "func")

            pattern = os.path.join(funtional_path, "**/*.nii.gz")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x
                                                                  if self.template_space in x
                                                                  if "preproc_bold" in x]


      def _isolate_confound_regressors(self):
            """
            Isolates relative paths to TSV files derived from fmriprep

            Returns
                  List of relative paths
            """

            functional_path = os.path.join(self.bids_root, 
                                          "derivatives/fmriprep",
                                          f"sub-{self.sub_id}", 
                                          "func")

            pattern = os.path.join(functional_path, "**/*.tsv")

            return [x for x in glob.glob(pattern, recursive=True) if self.task in x
                                                                  if "confounds" in x]


      def _build_bids_container(self):
            """
            
            """

            container = {}

            raw = self._isolate_raw_data()
            preprocessed = self._isolate_preprocessed_data()
            confounds = self._isolate_confound_regressors()

            if len(raw) != len(preprocessed):
                  raise OSError(
                        f"Length mismatch ... raw: {len(raw)} preprocessed: {len(preprocessed)}"
                  )
            
            elif len(raw) != len(confounds):
                  raise OSError(
                        f"Length mismatch ... raw: {len(raw)} confounds: {len(confounds)}"
                  )
            
            elif len(preprocessed) != len(confounds):
                  raise OSError(
                        f"Length mismatch ... preprocessed: {len(preprocessed)} confounds: {len(confounds)}"
                  )

            for ix in range(len(raw)):

                  run_value = f"run-{ix + 1}"

                  current_raw = [k for k in raw if run_value in k][0]
                  current_prep = [k for k in preprocessed if run_value in k][0]
                  current_confounds = [k for k in confounds if run_value in k][0]

                  container[run_value] = {
                        "raw_bold": current_raw,
                        "preprocessed_bold": current_prep,
                        "confounds": current_confounds
                  }

                  try:
                        container["all_raw_bold"].append(current_raw)
                  except:
                        container["all_raw_bold"] = [current_raw]

                  
                  try:
                        container["all_preprocessed_bold"].append(current_prep)
                  except:
                      container["all_preprocessed_bold"] = [current_prep]


                  try:
                      container["all_confounds"].append(current_confounds)
                  except:
                      container["all_confounds"] = [current_confounds]

                  
            container_filename = f"sub-{self.sub_id}_task-{self.task}_bids-container.json"

            with open(os.path.join(self.first_level_output, container_filename), "w") as outgoing:
                  json.dump(container, outgoing, indent=5)

            return container


      # --- Functional Helpers
      def load_confounds(self, run="ALL"):
            
            if run == "ALL":

                  output = pd.DataFrame()

                  for ix, file in enumerate(self.bids_container["all_confounds"]):

                        temp = pd.read_csv(file, sep="\t")

                        temp["run"] = [ix + 1] * len(temp)

                        output = output.append(temp, ignore_index=True)

                  return output.reset_index(drop=True)


            else:

                  key = f"run-{run}"

                  file = self.bids_container[key]["confounds"]

                  return pd.read_csv(file, sep="\t")
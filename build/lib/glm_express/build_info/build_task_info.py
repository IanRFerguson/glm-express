#!/bin/python3

"""
About this Script

The `build_task_info()` function creates the relevant `task_information.json` file
that is critical to running first- and second-level models using GLM-Express

Ian Richard Ferguson | Stanford University
"""

import os, json, sys
import pathlib
from bids import BIDSLayout

def derive_tasks(bids_root):
      """
      Feeds user-supplied BIDS root into BIDSLayout object to obtain list of functional tasks

      Parameters
            bids_root: str | Relative path to the top of your BIDS project

      Returns
            List of functional tasks
      """
      
      # Feed bids_root into BIDSLayout object
      # NOTE: This will throw an error if your project is not BIDS-compliant
      bids = BIDSLayout(bids_root)
      
      # Return functional tasks in list
      return bids.get_tasks()


def build_task_info(bids_root):
      """
      This function constructs the `task_information.json` file that is required
      for the Subject and GroupLevel objects to run successfully

      Parameters
            bids_root: str | Relative path to the top of your BIDS project
      """

      # Derive list of tasks in your BIDS project
      functional_tasks = derive_tasks(bids_root)
      
      path = pathlib.Path(bids_root).parents[0]
      output_path = os.path.join(path, "task_information.json")

      # Empty dictionary to append into
      output = {}

      # Loop through tasks and create dictionaries for each
      for task in functional_tasks:

            output[task] = {}                                           # Empty dictionary to append into

            output[task]['block_identifier'] = 'block_type'             # Column in events file corresponding to Block identifier (if applicable) 
            output[task]['condition_identifier'] = 'trial_type'         # Column in events file corresponding to Trial Type identifier
            output[task]['confound_regressors'] = []                    # Regressors to include from fmriprep confounds file
            output[task]['design_contrasts'] = []                       # Contrasts of interest to run in first-level design
            output[task]['excludes'] = []                               # List of subject IDs to exclude (for batching / SLURM scheduling)
            output[task]['group_level_regressors'] = []                 # Regressors to include at second-level (NOTE: In development)
            output[task]['tr'] = 1.                                     # Repetition time for the given first-level task (you can update this in the Subject object too)

      # Save task file locally
      with open(output_path, 'w') as outgoing:
            json.dump(output, outgoing, indent=6)


def build_dataset_description(bids_root):
      """
      For whatever reason, sample datasets are often missing a dataset
      description file. We'll create an empty file at __init__ if it doesn't exist

      Parameters
            bids_root: str | Relative path to the top of your BIDS project
      """

      output = {
            "Acknowledgements": "",
            "Authors": [],
            "BIDSVersion": "",
            "DatasetDOI": "",
            "Funding": "",
            "HowToAcknowledge": "",
            "License": "",
            "Name": "",
            "ReferencesAndLinks": [],
            "template": "project"
      }

      with open(os.path.join(bids_root, 'dataset_description.json'), 'w') as outgoing:
            json.dump(output, outgoing, indent=6)


if __name__ == "__main__":
      try:
            root = sys.argv[1]
      except:
            raise OSError(f'No command line argument supplied')

      build_task_info(root)

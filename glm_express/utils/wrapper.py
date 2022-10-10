#!/bin/python3
import pathlib, json, os
from .general_utils import derive_tasks


##########


def build_task_info(bids_root: os.path, 
                    return_path: bool=False,
                    verbose: bool=False):
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

      # Create file if it doesn't exist
      if not os.path.exists(output_path):

            if verbose:
                  print("\n** Writing task_information.json **\n")

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


      else:
            if verbose:
                  print("\n** Task file exists - see parent directory of BIDS project for details **\n")


      if return_path:
            return output_path
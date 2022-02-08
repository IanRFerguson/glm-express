#!/bin/python3

"""

"""

import os, json, sys
from bids import BIDSLayout

def derive_tasks(bids_root):
      bids = BIDSLayout(bids_root)
      return bids.get_tasks()


def build_task_info(bids_root):
      functional_tasks = derive_tasks(bids_root)

      output = {}

      for task in functional_tasks:

            output[task] = {}

            output[task]['design_type'] = 'event'
            output[task]['tr'] = 1.
            output[task]['excludes'] = []
            
            output[task]['condition_identifier'] = 'trial_type'
            output[task]['conditions'] = []
            output[task]['confound_regressors'] = []
            output[task]['auxilary_regressors'] = []
            output[task]['modulators'] = []

            output[task]['block_identifier'] = None
            
            output[task]['design-contrasts'] = 'default'
            output[task]['group_level_regressors'] = []


      output = {k:v for k, v in sorted(output.items())}

      with open(os.path.join(bids_root, 'task_information.json'), 'w') as outgoing:
            json.dump(output, outgoing, indent=5)


if __name__ == "__main__":

      try:
            root = sys.argv[1]
      except:
            raise OSError(f'No command line argument supplied')

      build_task_info(root)
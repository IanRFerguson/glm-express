#!/bin/python3
import os, json, sys, pathlib
from bids import BIDSLayout


##########


def derive_tasks(bids_root: os.path):
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



def build_dataset_description(bids_root: os.path):
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
#!/bin/python3

"""
About this Class

RestingState is optimized to model functional and structural connectivity
in a preprocessed resting state fMRI scan.

Ian Richard Ferguson | Stanford University
"""


# --- Imports
from glm_express.bids_pointer.bids_pointer import BIDSPointer
import os, glob, json, pathlib
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout


# --- Object definition
class RestingState:

      # -- Constructor
      def __init__(self, sub_id, task="rest", bids_root="./bids", suppress=False, template_space="MNI152NLin2009",
                  repetition_time=1., dummy_scans=2):

            # Inherits constructor from BIDSPointer
            BIDSPointer.__init__(self, sub_id=sub_id, task=task, bids_root=bids_root, suppress=suppress,
                                 template_space=template_space, repetition_time=repetition_time,
                                 dummy_scans=dummy_scans)


            self.mask_path = os.path.join(self.bids_root, "derivatives/nilearn_masks")

            if not os.path.exists(self.mask_path):
                  pathlib.Path(self.mask_path).mkdir(exist_ok=True, parents=True)


      # -- Nilearn helpers
      def load_mask(self, mask):
            """
            This function pulls down masks and atlases from the Nilearn API. On the first run,
            a NifTi file is saved locally in the mask directory.

            Parameters
                  mask: str | Validated name of mask to pull down

            Returns
                  NifTi file
            """

            valid = ["harvard_oxford", "msdl", "power", "setizman"]

            if mask.lower() not in valid:
                  raise ValueError(f"{mask} is invalid input ... valid: {valid}")

            #
            if mask == "harvard_oxford":
                  pass
            elif mask == "msdl":
                  pass
            elif mask == "power":
                  pass
            elif mask == "seitzman":
                  pass

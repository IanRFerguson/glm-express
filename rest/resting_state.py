#!/bin/python3

"""
About this Class

RestingState is optimized to model functional and structural connectivity
in a preprocessed resting state fMRI scan.

Ian Richard Ferguson | Stanford University
"""


# --- Imports
from multiprocessing.sharedctypes import Value
from .build import Build
import os, glob, json, pathlib
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nilearn import datasets
import nilearn.plotting as nip


# --- Object definition
class RestingState(Build):

      # -- Constructor
      def __init__(self, sub_id, task="rest", bids_root="./bids", suppress=False, template_space="MNI152NLin2009"):

            # Inherits constructor from BIDSPointer
            super(RestingState, self).__init__(sub_id=sub_id, task=task, 
                                               bids_root=bids_root, suppress=suppress, 
                                               template_space=template_space)


            self.mask_path = os.path.join(self.bids_root, "derivatives/nilearn_masks")

            if not os.path.exists(self.mask_path):
                  pathlib.Path(self.mask_path).mkdir(exist_ok=True, parents=True)



      # -- Nilearn helpers
      def load_atlas(self, mask, harvard_atlas_name="cort-prob-2mm"):
            """
            This function pulls down masks and atlases from the Nilearn API. On the first run,
            a NifTi file is saved locally in the mask directory.

            Parameters
                  mask: str | Validated name of mask to pull down

            Returns
                  NifTi file
            """

            mask = mask.lower().strip()
            valid = ["harvard_oxford", "msdl"]

            if mask.lower() not in valid:
                  raise ValueError(f"{mask} is invalid input ... valid: {valid}")

            #
            if mask == "harvard_oxford":

                  atlas_names = ["cort-prob-2mm", "cort-prob-1mm", "cortl-prob-2mm",
                                "sub-prob-2mm", "sub-prob-1mm"]

                  if harvard_atlas_name not in atlas_names:
                        raise ValueError(f"{harvard_atlas_name} invalid input ... valid: {atlas_names}") 

                  pull = datasets.fetch_atlas_harvard_oxford(atlas_name=harvard_atlas_name,
                                                             data_dir=self.mask_path)

                  return pull.filename, pull.labels
            
            #
            elif mask == "msdl":
                  pull = datasets.fetch_atlas_msdl(data_dir=self.mask_path)

                  return pull.maps, pull.labels



      def load_coords(self, coords):
            """
            
            """

            coords = coords.lower().strip()
            valid = ["power", "seitzman"]

            if coords not in valid:
                  raise ValueError(f"{coords} is invalid input ... valid: {valid}")

            #
            if coords == "power":
                  raise ValueError("power IS valid input but this function is currently in development ... use 'seitzman' for now")
            
            #
            elif coords == "seitzman":
                  return datasets.fetch_coords_seitzman_2018()



      # -- Wrappers
      def time_series_from_parcellation(self, run):
            pass



      def time_series_from_atlas(self, run, atlas="msdl", save_matrix_output=False, save_plots=False,
                                 show_plots=True, standardize=True):
            """
            Describe

            Parameters
                  run
                  atlas
                  save_matrix_output
                  save_plots

            Returns
                  Correlation matrix
            """

            from nilearn.maskers import NiftiMapsMasker
            from nilearn.connectome import ConnectivityMeasure

            atlas_maps, atlas_labels = self.load_atlas(mask=atlas)
            bold_run = self.bids_container[f"run-{run}"]["preprocessed_bold"]
            confound = self.load_confounds(run=run)

            masker = NiftiMapsMasker(maps_img=atlas_maps, standardize=standardize)

            time_series = masker.fit_transform(bold_run, confounds=confound)

            correlation_transformer = ConnectivityMeasure(kind="correlation")
            correlation_matrix = correlation_transformer.fit_transform(time_series)

            if show_plots:
                  self.plot_correlation_matrix(correlation_matrix, labels=atlas_labels, 
                                               run=run, save_local=save_plots)

                  self.plot_connectomes()



      def plot_correlation_matrix(self, matrix, labels, run, vmin=-1., vmax=1., 
                                  save_local=True):
            """
            Custom correlation matrix function

            Parameters
                  matrix: np.array | Correlation matrix derived from time series data
                  labels: list | Region or connectome labels from atlas
                  run: str or int | Run value from BIDS project
                  vmin: numeric | Minimum value to plot in correlation matrix
                  vmax: numeric | Maximum value to plot in correlation matrix
                  save_local: Boolean | if True, matrix is saved to subject output
            """
            
            # Instantiate matplotlib canvas
            plt.figure(figsize=(12,10))
            
            # Fill diagonal with zeroes
            np.fill_diagonal(matrix, 0)

            # Create heatmap of regional correlation values
            sns.heatmap(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        xticklabels=labels, yticklabels=labels)

            # Title will double as output filename
            title = f"sub-{self.sub_id}_task-{self.task}_run-{run}_correlation-matrix"
            plt.title(title)

            # Save locally
            if save_local:
                  output_name = os.path.join(self.first_level_output, 
                                             "plots",
                                            f"{title}.jpg")

                  plt.savefig(output_name)

            # Print to local environment
            else:
                  plt.show()


      def plot_connectomes(self, matrix, run, atlas_map, save_local=False):
            """
            
            """

            coordinates = nip.find_probabilistic_atlas_cut_coords(maps_img=atlas_map)

            title = f"sub-{self.sub_id}_run-{run}"

            if save_local:
                  output_name = f"sub-{self.sub_id}_task-{self.task}_connectome.jpg"
                  output_path = os.path.join(self.first_level_output, "plots", output_name)

                  nip.plot_connectome(matrix, coordinates, title=title,
                                      output_file=output_path)

            else:
                  nip.plot_connectome(matrix, coordinates, title=title,
                                      output_file=output_path)

                  nip.show()
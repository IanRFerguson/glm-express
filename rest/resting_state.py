#!/bin/python3

"""
About this Class

RestingState is optimized to model functional and structural connectivity
in a preprocessed resting state fMRI scan.

Ian Richard Ferguson | Stanford University
"""


# --- Imports
from glm_express.rest.build import Build_RS
import os, pathlib
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nilearn import datasets
import nilearn.plotting as nip


# --- Object definition
class RestingState(Build_RS):

      # -- Constructor
      def __init__(self, sub_id, task="rest", bids_root="./bids", suppress=False, 
                   template_space="MNI152NLin2009"):

            # Inherits constructor from Build_RS object
            Build_RS.__init__(self, sub_id=sub_id, task=task, 
                              bids_root=bids_root, suppress=suppress, 
                              template_space=template_space)
            
            self.mask_path = os.path.join(self.bids_root, "derivatives/nilearn_masks")

            if not os.path.exists(self.mask_path):
                  pathlib.Path(self.mask_path).mkdir(exist_ok=True, parents=True)



      # -- Nilearn dataset helpers
      def pull_msdl_atlas(self):
            """
            Wraps nilearn.datasets.fetch_atlas_msdl function
            """

            pull = datasets.fetch_atlas_msdl(data_dir=self.mask_path)

            return pull.maps, pull.labels



      def pull_harvard_oxford_atlas(self, atlas_name=None):
            """
            Wraps nilearn.datasets.fetch_atlas_harvard_oxford
            """

            if atlas_name is None:
                  print("NOTE: Defaulting to Cortical 2mm atlas")
                  atlas_name = "cort-prop-2mm"

            pull = datasets.fetch_atlas_harvard_oxford(data_dir=self.mask_path, atlas_name=atlas_name)

            return pull.maps, pull.labels



      def pull_seitzman_coords(self):
            """
            Wraps nilearn.datasets.fetch_coords_seitzman function
            """

            return datasets.fetch_coords_seitzman_2018()



      def pull_power_coords(self):
            """
            Wraps nilearn.datasets.fetch_coords_power function
            """

            return datasets.fetch_coords_power_2011()



      # -- Time serires extraction
      def matrix_from_labels_masker(self, run, atlas_to_use=None, labels_to_use=None, standardize=True,
                                    save_matrix_output=False, save_plots=False, show_plots=True,
                                    return_array=True, verbose=True):
            """
            This function applies when data from non-overlapping volumes should be extracted

            Parameters
                  run:  str or int | Functional run from BIDS project
                  atlas_to_use:  Relative path to NifTi mask (defaults to native MSDL atlas)
                  labels_to_use:  List of labels to feed to the correlation matrix
                  standardize:  Boolean | if True, the extracted signal is z-transformed
                  save_matrix_output:  Boolean | if True, correlation matrix is saved to subject folder
                  save_plots:  Boolean | if True, correlation matrix and connectome is saved locally
                  show_plots:  Boolean | if True, plots are displayed to the console
                  return_array:  Boolean | if True, array is returned and can be assigned to variable
                  verbose:  Boolean | if True, method mechanics are printed to the console

            Returns
                  If return_array == True, returns correlation matrix
            """

            from nilearn.maskers import NiftiLabelsMasker
            from nilearn.connectome import ConnectivityMeasure

            # ==== Validate method attributes ====
            # -- Check reference atlas validity
            if atlas_to_use is None:
                  atlas_maps, atlas_labels = self.pull_msdl_atlas()

            else:
                  if labels_to_use is None:
                        raise ValueError(
                              "NOTE: If supplying your own atlas, you MUST provide list of labels"
                        )

                  atlas_maps = atlas_to_use
                  atlas_labels = labels_to_use

            # -- Load subject BOLD and confound data
            bold_run = self.bids_container[f"run-{run}"]["preprocessed_bold"]
            confound = self.load_confounds(run=run).loc[:, self.confound_regressor_names]

            method = f"""
Running sub-{self.sub_id}_task-{self.task}_run-{run}...\n\n
Reference atlas:\t{atlas_maps}\n
BOLD run:\t\t{bold_run}\n
Confounds:\t\t{confound}
            """

            # -- If verbose, we'll print to the console as the method runs
            if verbose:
                  value = 5
                  print(method)
                  sleep(2)
            else:
                  value = 0

            # ==== Run labels masker ====
            masker = NiftiLabelsMasker(labels_img=atlas_maps, standardize=standardize,
                                       verbose=value)

            time_series = masker.fit_transform(bold_run, confounds=confound)

            # ==== Run correlation transformer ====
            correlation_transformer = ConnectivityMeasure(kind="correlation")
            correlation_matrix = correlation_transformer.fit_transform([time_series])[0]

            # ==== Apply both plotting functions ====
            self.plot_correlation_matrix(correlation_matrix, labels=atlas_labels,
                                         run=run, save_local=save_plots, suppress_plot=show_plots)

            self.plot_connectomes(matrix=correlation_matrix, run=run, atlas_map=atlas_maps,
                                  save_local=save_plots, show_plot=show_plots)


            # Save matrix locally if user desires
            if save_matrix_output:

                  filename = f"sub-{self.sub_id}_task-{self.task}_run-{run}_labels-masker-matrix.npy"

                  correlation_matrix.tofile(os.path.join(self.first_level_output,
                                                         "models",
                                                         filename))
                  

            if return_array:
                  return correlation_matrix



      def matrix_from_maps_masker(self, run, atlas_to_use=None, labels_to_use=None, standardize=True,
                                  save_matrix_output=False, save_plots=False, show_plots=True, 
                                  return_array=True, verbose=True):
            """
            This function applies when data from overlapping volumes should be extracted

            Parameters
                  run:  str or int | Functional run from BIDS project
                  atlas_to_use:  Relative path to NifTi mask (defaults to native MSDL atlas)
                  labels_to_use:  List of labels to feed to the correlation matrix
                  standardize:  Boolean | if True, the extracted signal is z-transformed
                  save_matrix_output:  Boolean | if True, correlation matrix is saved to subject folder
                  save_plots:  Boolean | if True, correlation matrix and connectome is saved locally
                  show_plots:  Boolean | if True, plots are displayed to the console
                  return_array:  Boolean | if True, array is returned and can be assigned to variable
                  verbose:  Boolean | if True, method mechanics are printed to the console

            Returns
                  If return_array == True, returns correlation matrix
            """

            from nilearn.maskers import NiftiMapsMasker
            from nilearn.connectome import ConnectivityMeasure

            # ==== Validate method attributes ====
            # -- Check reference atlas validity
            if atlas_to_use is None:
                  print("NOTE: Defaulting to MSDL Atlas")
                  atlas_maps, atlas_labels = self.pull_msdl_atlas()

            else:
                  if labels_to_use is None:
                        raise ValueError(
                              "NOTE: If supplying your own atlas, you MUST provide list of labels"
                        )

                  atlas_maps = atlas_to_use
                  atlas_labels = labels_to_use

            # -- Load subject BOLD and confound data
            bold_run = self.bids_container[f"run-{run}"]["preprocessed_bold"]
            confound = self.load_confounds(run=run).loc[:, self.confound_regressor_names]

            method = f"""
Running sub-{self.sub_id}_task-{self.task}_run-{run}...\n\n
Reference atlas\n{atlas_maps}\n\n
BOLD run\n{bold_run}\n\n
Confounds\n{self.bids_container[f"run-{run}"]["confounds"]}\n\n
            """

            # -- If verbose, we'll print to the console as the method runs
            if verbose:
                  value = 5
                  print(method)
                  sleep(2)
            
            else:
                  value = 0


            # ==== Run maps masker ====
            masker = NiftiMapsMasker(maps_img=atlas_maps, standardize=standardize,
                                     verbose=value)

            time_series = masker.fit_transform(bold_run, confounds=confound)


            # ==== Run correlation transformer ====
            correlation_transformer = ConnectivityMeasure(kind="correlation")
            correlation_matrix = correlation_transformer.fit_transform([time_series])[0]

            # ==== Apply both plotting functions ====
            self.plot_correlation_matrix(correlation_matrix, labels=atlas_labels, 
                                         run=run, save_local=save_plots, 
                                         show_plot=show_plots)

            self.plot_connectomes(correlation_matrix, run=run, 
                                  atlas_map=atlas_maps, save_local=save_plots, 
                                  show_plot=show_plots)

            # Save matrix locally if user desires
            if save_matrix_output:
                  
                  filename = f"sub-{self.sub_id}_task-{self.task}_run-{run}_maps-masker-matrix.npy"

                  correlation_matrix.tofile(os.path.join(self.first_level_output,
                                                         "models",
                                                         filename))

            if return_array:
                  return correlation_matrix



      def load_correlation_matrix(self, run, masker_method="maps"):
            """
            Loads a correlation matrix that has been run and saved

            Parameters
                  run:  str or int | Function run derived from BIDS project
                  masker_method:  str | Method that time series was extracted

            Returns
                  2x2 correlation matrix as numpy.ndarray object
            """

            # File name of saved correlation matrix
            target_file_name = f"sub-{self.sub_id}_task-{self.task}_run-{run}_{masker_method}-masker-matrix.npy"
            
            # Relative path to file, if it exists
            target_path = os.path.join(self.first_level_output, "models", target_file_name)

            # Return array if it exists
            if os.path.exists(target_path):
                  return np.load(target_path)

            else:
                  raise OSError(f"File doesn't exist: {target_file_name}")



      # -- Plotting functions
      def plot_correlation_matrix(self, matrix, labels, run, vmin=-1., vmax=1., 
                                  save_local=True, show_plot=False, override_range=False):
            """
            Custom correlation matrix function

            Parameters
                  matrix: np.array | Correlation matrix derived from time series data
                  labels: list | Region or connectome labels from atlas
                  run: str or int | Run value from BIDS project
                  vmin: numeric | Minimum value to plot in correlation matrix
                  vmax: numeric | Maximum value to plot in correlation matrix
                  save_local: Boolean | if True, matrix is saved to subject output
                  show_plot:  Boolean | if True, plot is printed to the console
                  override_range:  Boolean | if True, vmin and vmax are ignored
            """
            
            # Instantiate matplotlib canvas
            plt.figure(figsize=(12,10))
            
            # Fill diagonal with zeroes
            np.fill_diagonal(matrix, 0)


            # Include vmin and vmax
            if not override_range:
                  sns.heatmap(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                              xticklabels=labels, yticklabels=labels)
            
            # Ignore vmin and vmax
            else:
                  sns.heatmap(matrix, cmap="RdBu_r",
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

            # Print plot to console
            if show_plot:
                  plt.show()
            else:
                  plt.close()



      def plot_connectomes(self, matrix, run, atlas_map, save_local=False, 
                           show_plot=False, display_mode="lyrz"):
            """
            Plots connectome map in a glass brain

            Parameters
                  matrix:  np.ndarray | Correlation matrix derived from helper function
                  run:  str or int | Functional run derived from BIDS project
                  atlas_map:  NifTi image or path to NifTi image
                  save_local:  Boolean | if True, connectome plot is saved locally
                  show_plot:  Boolean | if True, plot is printed to the console
            """

            # Gets coordinates from 4D probabilistic atlas
            coordinates = nip.find_probabilistic_atlas_cut_coords(maps_img=atlas_map)

            # Format plot title
            title = f"sub-{self.sub_id}_run-{run}"

            # Save plot to output directory
            if save_local:
                  
                  # Formatted output name
                  output_name = f"sub-{self.sub_id}_task-{self.task}_connectome.jpg"
                  
                  # Relative path to output filename
                  output_path = os.path.join(self.first_level_output, "plots", output_name)

                  nip.plot_connectome(matrix, 
                                      coordinates, 
                                      title=title,
                                      output_file=output_path,
                                      display_mode=display_mode)

                  plt.close()


            if show_plot:
                  nip.plot_connectome(matrix, 
                                      coordinates, 
                                      title=title,
                                      display_mode=display_mode)

                  nip.show()

            else:
                  # This is hacky but effectively suppresses output
                  try:
                        nip.plot_connectome(matrix, coordinates, display_mode=None)
                        plt.close()
                  except:
                        pass

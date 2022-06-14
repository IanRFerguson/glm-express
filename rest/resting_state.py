#!/bin/python3

"""
About this Class

RestingState is optimized to model functional and structural connectivity
in a preprocessed resting state fMRI scan.

Ian Richard Ferguson | Stanford University
"""


# --- Imports
from cProfile import label
from multiprocessing.sharedctypes import Value
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
      """
      NOTE: Keeping this function around but out of production
      """
      def __matrix_from_labels_masker(self, run="ALL", atlas_to_use=None, labels_to_use=None, standardize=True,
                                    save_matrix_output=False, save_plots=False, show_plots=True,
                                    return_array=True, verbose=True):
            """
            This function applies when data from non-overlapping volumes should be extracted

            Parameters
                  run:  str or int | Functional run from BIDS project; defaults to ALL runs (concatenated matrix)
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
            if run == "ALL":
                  bold_run = [x for x in self.bids_container["all_preprocessed_bold"]]

            else:
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



      """
      NOTE: Keeping this function around but out of production
      """

      def __matrix_from_maps_masker(self, run, atlas_to_use=None, labels_to_use=None, standardize=True,
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



      def matrix_from_masker(self, run="ALL", method="maps", atlas_to_use=None, 
                             labels_to_use=None, standardize=True, save_matrix_output=False, 
                             save_plots=False, show_plots=True, return_array=True, verbose=True):
            """
            This function applies when data from overlapping volumes should be extracted

            Parameters
                  run:  str or int | Functional run from BIDS project; if ALL, functional runs are concatenated
                  method: str | Currently, 'maps' or 'labels' ... determines which masker to use
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

            from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
            from nilearn.connectome import ConnectivityMeasure


            # ==== Validate user input ====
            if method not in ['maps', 'labels']:
                  raise ValueError(
                        f"ERROR: {method} is invalid input ... valid: ['maps', 'labels']"
                  )

            # -- Default to MSDL atlas if none is provided
            if atlas_to_use is None:
                  print("NOTE: Defaulting to MSDL atlas")
                  atlas_to_use, labels_to_use = self.pull_msdl_atlas()

            # -- If atlas IS provided and labels ARE NOT
            elif atlas_to_use is not None and labels_to_use is None:
                  raise ValueError(
                        "ERROR: If providing a reference atlas you MUST provide a list of corresponding labels"
                  )

            # -- User desires all runs to be concatenated
            # TODO: This currently isn't working            
            if run == "ALL":
                  # List of relative paths to preprocessed BOLD runs
                  bold = [x for x in self.bids_container["all_preprocessed_bold"]]

                  # List of loaded and isolated confound regressors
                  confounds = [self.load_confounds(run=x).loc[:, self.confound_regressor_names] 
                               for x in range(1, len(self.preprocessed_runs)+1)]

                  # This will print if verbose
                  header = f"sub-{self.sub_id}_aggregated-BOLD-runs"

            # -- Single run
            else:
                  # Isolate BIDS run
                  bold = self.bids_container[f"run-{run}"]["preprocessed_bold"]

                  # Isolate confound regressors
                  confounds = self.load_confounds(run=run).loc[:, self.confound_regressor_names]

                  # This will print if verbose
                  header = f"sub-{self.sub_id}_run-{run}"

            # This will print if verbsose
            message = f"""
Running {header}\n\n
Reference Atlas\n{atlas_to_use}\n\n
BOLD Run(s)\n{bold}\n\n
Confounds\n{confounds}\n\n
            """

            if verbose:
                  print(message)
                  value = 5
                  sleep(2)

            else:
                  value = 0


            # ==== Run desired masker ====
            if method == "maps":
                  masker = NiftiMapsMasker(maps_img=atlas_to_use, 
                                           standardize=standardize,
                                           verbose=value)


            elif method == "labels":
                  masker = NiftiLabelsMasker(labels_img=atlas_to_use, 
                                             standardize=standardize,
                                             verbose=value)

            # Fit bold run(s) to masker object
            time_series = masker.fit_transform(bold, confounds=confounds)

            # ==== Run correlation transformer ====
            correlation_transformer = ConnectivityMeasure(kind="correlation")
            correlation_matrix = correlation_transformer.fit_transform([time_series])[0]

            # ==== Apply both plotting functions ====
            self.plot_correlation_matrix(correlation_matrix, labels=labels_to_use, 
                                         run=run, save_local=save_plots, 
                                         show_plot=show_plots)

            self.plot_connectomes(correlation_matrix, run=run, 
                                  atlas_map=atlas_to_use, save_local=save_plots, 
                                  show_plot=show_plots)

            # Save matrix locally if user desires
            if save_matrix_output:
                  
                  if run == "ALL":
                        filename = f"sub-{self.sub_id}_task-{self.task}_aggregated.npy"

                  else:
                        filename = f"sub-{self.sub_id}_task-{self.task}_run-{run}_maps-masker-matrix.npy"

                  correlation_matrix.tofile(os.path.join(self.first_level_output, "models", filename))

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


            if run == "ALL":
                  #
                  title = f"sub-{self.sub_id}_aggregated-matrix"

            else:
                  # Format plot title
                  title = f"sub-{self.sub_id}_run-{run}"

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

            if run == "ALL":
                  #
                  title = f"sub-{self.sub_id}_aggregated-matrix"

            else:
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



      # --- Time series wrappers
      def _extract_all_time_series(self, method="maps", atlas_to_use=None, labels_to_use=None,
                                   standardize=True, show_plots=False, save_plots=True,
                                   save_each_array=True, verbose=False):
            """
            DEVELOPMENTAL FUNCTION

            This function loops through all resting state BOLD
            runs and creates a connectivity matrix for each run

            Parameters
                  method: str | "maps" or "labels", determines masker type
                  atlas_to_use: NifTi file or path to NifTi file, reference atlas
                  labels_to_use: list | Functional areas corresponding to reference atlas
                  standardize: Boolean | if True, time series are z-transformed
                  show_plots: Boolean | if True, plots are printed to the console
                  save_plots: Boolean | if True, run-wise plots are saved locally
                  save_each_array: Boolean | if True, run-wise matricies are saved locally
                  verbose: Boolean | if True, processes are printed to the console

            Returns
                  List of fun-wise connectivity matrices
            """

            method = method.lower().strip()

            if method not in ["maps", "labels"]:
                  raise ValueError(
                        f"ERROR: {method} is invalid input ... valid: ['maps', 'labels']"
                  )

            # ==== Run Masker helper method ====
            container = []

            for ix in range(len(self.preprocessed_runs)):
                        
                  run = ix + 1

                  temp_matrix = self.matrix_from_masker(run=run, atlas_to_use=atlas_to_use, method=method,
                                                                  labels_to_use=labels_to_use, standardize=standardize, 
                                                                  show_plots=show_plots, save_plots=save_plots, 
                                                                  verbose=verbose, save_matrix_output=save_each_array)

                  container.append(temp_matrix)


            return container



      def _weigh_average_timeseries(self, extracted_timeseries=[], atlas_to_use=None, 
                                   labels_to_use=None, show_plots=False, save_plots=True, 
                                   save_matrix=True):
            """
            DEVELOPMENTAL FUNCTION

            This function aggregates all BOLD connectivity matrices into a single
            index of a subject's functional connectivity at reast

            Parameters
                  extracted_timeseries: list | List of connectivity matricies from each BOLD run
                  atlas_to_use: NifTi file or path to NifTi file to use as reference atlas
                  labels_to_use: list | Functional areas corresponding to reference atlas
                  show_plots: Boolean | if True, plots are displayed to the console
                  save_plots: Boolean | if True, plots are saved to participant output directory
                  save_matrix: Boolean | if True, each BOLD connectivity matrix is saved locally

            Returns
                  Average functional connectivity matrix
            """

            # ==== Validate user input ====
            if atlas_to_use is None:
                  print("Defaulting to MSDL atlas")
                  atlas_to_use, labels_to_use = self.pull_msdl_atlas()

            # Weigh average functional connectivity matrix
            average_matrix = np.mean(np.array(extracted_timeseries), axis=0)

            # Run plotting functions 
            self.plot_correlation_matrix(average_matrix, labels=labels_to_use, run="ALL",
                                         save_local=save_plots, show_plot=show_plots)

            self.plot_connectomes(average_matrix, run="ALL", atlas_map=atlas_to_use,
                                  save_local=save_plots, show_plot=show_plots)

            # Save matrix locally
            if save_matrix:
                  filename = f"sub-{self.sub_id}_task-{self.task}_averaged-matrix.npy"

                  average_matrix.tofile(os.path.join(self.first_level_output,
                                                     "models",
                                                     filename))

            return average_matrix



      def _run_weighted_timeseries(self, method="maps", atlas_to_use=None, labels_to_use=None,
                                  standardize=True, show_plots=False, save_plots=True,
                                  save_each_array=True, verbose=False, return_matrix=True):
            """
            DEVELOPMENTAL FUNCTION

            This function combines the wrapping functions above. In practice, it extacts a
            time series for each BOLD run, creates a connectivity matrix for each run, and
            aggregates a weighted average as an index of the subject's overall functional
            connectivity at rest

            Parameters
                  method: str | maps or labels, determines which masker will be used
                  atlas_to_use: NifTi file or path to NifTi file to use as reference atlas
                  labels_to_use: list | Functional areas corresponding to reference atlas
                  standardize: Boolean | if True, time series is z-transformed
                  show_plots: Boolean | if True, plots are displayed to the console
                  save_plots: Boolean | if True, plots are saved to participant output directory
                  save_each_array: Boolean | if True, a matrix is saved per run
                  verbose: Boolean | if True, processes are printed to the console
                  return_matrix: Boolean | if True, the weighted matrix returned

            Returns
                  If return_matrix, the weighted matrix is returned
            """

            # ==== Validate user input ====
            if atlas_to_use is None:
                  print("Defaulting to MSDL atlas...")
                  atlas_to_use, labels_to_use = self.pull_msdl_atlas()


            if method not in ["maps", "labels"]:
                  raise ValueError(f"{method} is invalid input ... valid: ['maps', 'labels']")


            # ==== Extract run-wise timeseries ====
            if verbose:
                  print("Extracting run-wise timeseries...")
                  sleep(1)

            aggregate_timeseries = self.extract_all_time_series(method=method, atlas_to_use=atlas_to_use,
                                                                labels_to_use=labels_to_use, standardize=standardize,
                                                                show_plots=show_plots, save_plots=save_plots,
                                                                save_each_array=save_each_array, verbose=verbose)

            # ==== Weigh individual timeseries ====
            if verbose:
                  print("Aggregating connectivity matrices...")
                  sleep(1)

            weighted_connectomes = self.weigh_average_timeseries(aggregate_timeseries, atlas_to_use=atlas_to_use,
                                                                 labels_to_use=labels_to_use, show_plots=show_plots,
                                                                 save_plots=save_plots, save_matrix=save_each_array)

            if return_matrix:
                  return weighted_connectomes
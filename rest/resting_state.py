#!/bin/python3

# --- Imports
from glm_express.rest.build import Build_RS
import os, pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nilearn import datasets
import nilearn.plotting as nip
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import GraphicalLassoCV


# --- Object definition
class RestingState(Build_RS):
      """
      RestingState is optimized to model functional and structural connectivity
      in a preprocessed resting state fMRI scan.
      """

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



      # -- Plotting functions
      def plot_correlation_matrix(self, matrix, labels, run=None, vmin=-1., vmax=1.,
                                  save_local=True, show_plot=False, override_range=False,
                                  custom_title=None, lower_triangle=False):
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
                  custom_title: str (optional)
            """

            if run is None and custom_title is None:
                 raise ValueError(
                     "You must provide a run number OR custom title")

            if custom_title:
                 title = custom_title

            else:
                  if run == "ALL":
                       # Format plot title
                       title = f"sub-{self.sub_id}_aggregated-matrix"

                  else:
                       # Format plot title
                       title = f"sub-{self.sub_id}_run-{run}"

            if lower_triangle:
                 mask = np.triu(np.ones_like(matrix))
            else:
                 mask = None

            # Instantiate matplotlib canvas
            plt.figure(figsize=(12, 10))

            # Fill diagonal with zeroes
            np.fill_diagonal(matrix, 0)

            # Include vmin and vmax
            if not override_range:
                  sns.heatmap(matrix,
                              cmap="RdBu_r",
                              vmin=vmin,
                              vmax=vmax,
                              xticklabels=labels,
                              yticklabels=labels,
                              mask=mask)

            # Ignore vmin and vmax
            else:
                  sns.heatmap(matrix,
                              cmap="RdBu_r",
                              xticklabels=labels,
                              yticklabels=labels,
                              mask=mask)

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


      def plot_connectomes(self, matrix, atlas_map, run=None, save_local=False,
                           show_plot=False, display_mode="lyrz",
                           custom_title=None, custom_output_name=None):
            """
            Plots connectome map in a glass brain

            Parameters
                  matrix:  np.ndarray | Correlation matrix derived from helper function
                  run:  str or int | Functional run derived from BIDS project
                  atlas_map:  NifTi image or path to NifTi image
                  save_local:  Boolean | if True, connectome plot is saved locally
                  show_plot:  Boolean | if True, plot is printed to the console
            """

            if run is None and custom_title is None:
                 raise ValueError()

            if run is None and custom_output_name is None:
                 raise ValueError()

            # -- Title definition
            if custom_title is not None:
                 title = custom_title

            else:
                  if run == "ALL":
                       title = f"sub-{self.sub_id}_aggregated-matrix"

                  else:
                       title = f"sub-{self.sub_id}_run-{run}"

            # -- Output name definition
            if custom_output_name is not None:
                 output_name = f"{custom_output_name}.jpg"

            else:
                  if run == "ALL":
                       output_name = f"sub-{self.sub_id}_aggregated-matrix.jpg"

                  else:
                       output_name = f"sub-{self.sub_id}_run-{run}_connectome.jpg"

            # Gets coordinates from 4D probabilistic atlas
            coordinates = nip.find_probabilistic_atlas_cut_coords(
                  maps_img=atlas_map)

            # Save plot to output directory
            if save_local:

                  # Relative path to output filename
                  output_path = os.path.join(self.first_level_output, "plots", output_name)

                  nip.plot_connectome(matrix, coordinates, title=title,
                                      output_file=output_path, display_mode=display_mode)

                  plt.close()


            # Display plot
            if show_plot:
                  nip.plot_connectome(matrix, coordinates, title=title,
                                      display_mode=display_mode)

                  nip.show()

            else:
                  # This is hacky but effectively suppresses output
                  try:
                        nip.plot_connectome(matrix, coordinates, display_mode=None)
                        plt.close()
                  except:
                       pass



      # -- Masking functions
      def _compile_single_image(self):
            """
            This helper streamlines NifTi image concatenation, so
            analyses do not have be weighted or averaged

            Returns
                  One NifTi image, One long DataFrame of confound regressors
            """

            from nibabel.funcs import concat_images
            import pandas as pd

            images = []

            for run in range(1, len(self.preprocessed_runs)+1):
                  images.append(self.bids_container[f"run-{run}"]["preprocessed_bold"])

            single_run = concat_images(images, axis=3)
            confounds = self.load_confounds(run="ALL")

            return single_run, confounds



      def extract_time_series(self, run, method="maps", atlas_to_use=None, labels_to_use=None, 
                              standardize=True, verbose=True, regress_motion_outliers=True,
                              a_comp_cor=True, t_comp_cor=True):
            """
            TODO: Add docstring
            """

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


            all_regressors = [self.confound_regressor_names]


            # -- Check modeling parameters
            if regress_motion_outliers:
                  all_regressors = all_regressors + \
                        [x for x in self.load_confounds(run=run).columns 
                        if "motion_outlier" in x]

            if a_comp_cor:
                  all_regressors += [x for x in self.load_confounds(run=run).columns if "a_comp_cor" in x]

            if t_comp_cor:
                  all_regressors += [x for x in self.load_confounds(run=run).columns if "t_comp_cor" in x]

            if verbose:
                  print(f"Regressing out the following: {all_regressors}")


            # -- User defined concatenated runs
            if run == "ALL":

                  bold, confounds = self._compile_single_image()

                  confounds.fillna(0, inplace=True)

                  header = f"sub-{self.sub_id}_concatenated-BOLD-runs"


            # -- Single run
            elif run != "ALL":
                  # Isolate BIDS run
                  bold = self.bids_container[f"run-{run}"]["preprocessed_bold"]

                  # Isolate confound regressors
                  confounds = self.load_confounds(run=run).loc[:, self.confound_regressor_names].fillna(0)

                  # This will print if verbose
                  header = f"sub-{self.sub_id}_run-{run}"


            # Define verbosity
            if verbose:
                  value = 5

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

            # -- Fit bold run(s) to masker object
            time_series = masker.fit_transform(bold, confounds=confounds)

            return time_series



      def matrix_from_masker(self, run="ALL", method="maps", atlas_to_use=None, 
                             labels_to_use=None, standardize=True, save_matrix_output=False, 
                             save_plots=False, show_plots=True, return_array=True, verbose=True,
                             a_comp_cor=True, t_comp_cor=True, lower_triangle=False):
            """
            Creates a correlation matrix derived from time series

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
                  a_comp_cor: Boolean | if True, anatomical noise components are regressed out
                  t_comp_cor: Boolean | if True, temporal noise components are regressed out
                  lower_triangle: Boolean | if True, only lower triangle of correlation matrix is rendered

            Returns
                  If return_array == True, returns correlation matrix
            """

            # -- Default to MSDL atlas if none is provided
            if atlas_to_use is None:
                  print("NOTE: Defaulting to MSDL atlas")
                  atlas_to_use, labels_to_use = self.pull_msdl_atlas()

            # -- If atlas IS provided and labels ARE NOT
            elif atlas_to_use is not None and labels_to_use is None:
                  raise ValueError(
                        "ERROR: If providing a reference atlas you MUST provide a list of corresponding labels"
                  )


            time_series = self.extract_time_series(run=run, method=method,
                                                   atlas_to_use=atlas_to_use,
                                                   labels_to_use=labels_to_use,
                                                   standardize=standardize,
                                                   verbose=verbose,
                                                   a_comp_cor=a_comp_cor,
                                                   t_comp_cor=t_comp_cor,
                                                   lower_triangle=lower_triangle)

            

            # -- Fit bold run(s) to masker object
            correlation_transformer = ConnectivityMeasure(kind="correlation")
            correlation_matrix = correlation_transformer.fit_transform([time_series])[0]


            # ==== Apply both plotting functions ====
            self.plot_correlation_matrix(correlation_matrix, 
                                         labels=labels_to_use, 
                                         run=run, 
                                         save_local=save_plots, 
                                         show_plot=show_plots,
                                         lower_triangle=lower_triangle)

            self.plot_connectomes(correlation_matrix, 
                                  run=run, 
                                  atlas_map=atlas_to_use, 
                                  save_local=save_plots, 
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



      def connectome_covariance(self, run="ALL", method="maps", atlas_to_use=None, labels_to_use=None,
                                standardize=True, save_matrix_output=False, save_plots=False,
                                show_plots=True, return_array=True, verbose=True,
                                sparse_inverse=True, a_comp_cor=True, t_comp_cor=True,
                                lower_triangle=False):
            """
            Maps direct connections between regions using sparse inverse covariance estimator

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
                  sprase_inverse: Boolean | if True, estimator precision is used (else, estimator covariance is used)
                  a_comp_cor: Boolean | if True, anatomical noise components are regressed out
                  t_comp_cor: Boolean | if True, temporal noise components are regressed out
                  lower_triangle: Boolean | if True, only lower triangle of heatmap is rendered

            Returns
                  if save_matrix_output, matrix is saved to output directory
            """

            # -- Default to MSDL atlas if none is provided
            if atlas_to_use is None:
                  print("NOTE: Defaulting to MSDL atlas")
                  atlas_to_use, labels_to_use = self.pull_msdl_atlas()


            time_series = self.extract_time_series(run=run, method=method, 
                                                   atlas_to_use=atlas_to_use,
                                                   labels_to_use=labels_to_use, 
                                                   standardize=standardize,
                                                   verbose=verbose,
                                                   a_comp_cor=a_comp_cor,
                                                   t_comp_cor=t_comp_cor)

            estimator = GraphicalLassoCV()
            estimator.fit(time_series)

            if run == "ALL":
                  formatted_run = "aggregated"
            else:
                  formatted_run = f"run-{run}"

            if sparse_inverse:
                  matrix_values = -estimator.precision_
                  c_title = f"sub-{self.sub_id}_{formatted_run}_sparse-inverse-covariance"

            else:
                  matrix_values = estimator.covariance_
                  c_title = f"sub-{self.sub_id}_{formatted_run}_covariance"

            # Plotting function
            self.plot_correlation_matrix(matrix_values, 
                                         labels=labels_to_use, 
                                         custom_title=c_title,
                                         show_plot=show_plots, 
                                         save_local=save_plots,
                                         lower_triangle=lower_triangle)

            self.plot_connectomes(matrix_values, 
                                  atlas_map=atlas_to_use, 
                                  save_local=save_plots,
                                  show_plot=show_plots, 
                                  custom_title=c_title, 
                                  custom_output_name=c_title)


            if save_matrix_output:
                  
                  # Custome output name
                  output_name = f"{c_title}.npy"

                  # Save file locally
                  matrix_values.tofile(os.path.join(self.first_level_output, "models", output_name))


            if return_array:
                  return matrix_values



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



      def matrix_to_dataframe(self, incoming_matrix=None, labels=None, a_comp_cor=True,
                              t_comp_cor=True):
            """
            Renders a one-dimensional Pandas DataFrame with all functional connection correlations mapped.
            You can supply your own matrix and labels, or use the default MSDL atlas values

            Parameters
                  incoming_matrix: np.ndarray | Defaults to None, in which case covariance matrix is produced
                  label_indices: List of functional regions to map onto correlation DF
                  a_comp_cor: Boolean | if True, anatomical noise components are included
                  t_comp_cor: Boolean | if True, temporal noise components are included

            Returns
                  Pandas DataFrame
            """

            if incoming_matrix is None:
                  incoming_matrix = self.connectome_covariance(verbose=False,
                                                               a_comp_cor=a_comp_cor,
                                                               t_comp_cor=t_comp_cor,
                                                               show_plots=False,
                                                               save_plots=False,
                                                               save_matrix_output=False,
                                                               return_array=True)

            if labels is None:
                  _, labels = self.pull_msdl_atlas()


            wide_df =  pd.DataFrame(incoming_matrix, 
                                    index=labels, 
                                    columns=labels)

            output = {}

            for region_l in labels:
                  
                  for region_r in labels:
                        
                        if region_l != region_r:

                              if f"{region_r}_{region_l}" not in list(output.keys()):

                                    key = f"{region_l}_{region_r}"

                                    value = wide_df.loc[region_l, region_r]

                                    output[key] = value


            return pd.DataFrame(output, index=[0])
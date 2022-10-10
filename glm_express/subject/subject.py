#!/bin/python3
from .bids_pointer import Build_Subject
import os, json
from time import sleep
import pandas as pd
import numpy as np
from tqdm import tqdm
import nilearn.plotting as nip
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report


##########


class Subject(Build_Subject):
      """
      Subject inherits all functions and attributes from the BIDSPointer class defined in this package. 
      Building on this framework, Subject is optimized to run basic first-level GLMs out of the box, 
      and users have the option to define their own design matrices as well.
      """

      def __init__(self, sub_id: str, task: str, bids_root: os.path, suppress: bool=False, 
                   template_space: str="MNI152NLin2009", repetition_time: float=1., 
                   dummy_scans: int=0):

            # Inherits constructor from BIDSPointer
            Build_Subject.__init__(
                  self, sub_id=sub_id, 
                  task=task, bids_root=bids_root, 
                  suppress=suppress,
                  template_space=template_space, 
                  repetition_time=repetition_time,
                  dummy_scans=dummy_scans
            )



      def generate_design_matrix(self, run: int, non_steady_state: bool=False, 
                                 auto_block_regressors: bool=False, motion_outliers: bool=True, 
                                 drop_fixation: bool=True, a_comp_cor: bool=True, 
                                 t_comp_cor: bool=True) -> pd.DataFrame:
            """
            Builds a first level design matrix

            Parameters
                  run: int | Functional run number, lines up with events file
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from event files

            Returns
                  Pandas DataFrame object
            """

            def aggregate_non_steady_state(length: int, dummy_value: int) -> list:
                  """
                  Creates a binary vector to serve as non_steady_state regressor

                  Parameters
                        length: int | Number of scans in the present run
                        dummy_value: int | Number of non-steady state regressors 

                  Returns
                        A binary vector of length `length`
                  """

                  # Number of scans - Number of dummy scans
                  real_length = length - int(dummy_value)

                  # Binary vector
                  return [1] * dummy_value + [0] * real_length



            def block_regressor(DF: pd.DataFrame) -> str:
                  """
                  Aggregates block and trial regressors (e.g., high_trust_perspective)

                  Parameters
                        DF: Pandas DataFrame | Both block and trial type must be present for this function to work!

                  Returns
                        Formatted string per row
                  """

                  # Block regressor
                  block = DF['block_type']
                  
                  # Trial type regressor
                  trial = DF['trial_type']

                  # Return formatted string
                  return f'{block}_{trial}'



            def reorder_design_columns(DM: pd.DataFrame):
                  """
                  Moves Nilearn modeling regressors to the end of the design matrix

                  Parameters
                        DM: Pandas DataFrame |

                  Returns:
                        Pandas DataFrame
                  """

                  # Drop index column if it exists
                  try:
                        DM.drop(columns=['index'], inplace=True)
                  except:
                        DM = DM

                  # Nilearn modeling regressors (drift and constant)
                  tail = [x for x in DM.columns if 'drift' in x] + ['constant']
                  
                  # All other column names
                  head = [x for x in DM.columns if x not in tail]

                  # Head first, then tails
                  clean = head + tail

                  # Return reordered Design Matrix
                  return DM.loc[:, clean]


            ##########

            
            iso_container = self.bids_container[f'run-{run}']                                   # BIDS container for the current run
            events = pd.read_csv(iso_container['event'], sep='\t')                              # Load events file
            confounds = pd.read_csv(iso_container['confounds'], sep='\t')                       # Load fmriprep regressors
            voi = ['onset', 'duration', 'trial_type']                                           # Starting variables of interest

            if auto_block_regressors:
                  if 'block_type' in events.columns:
                        voi += ['block_type']                                                   # Add block type to VOI if True

            events = events.loc[:, voi]                                                         # Reduce 
            

            ##########

            
            n_scans = len(confounds)                                                            # Number of scans in the current run
            t_r = self.t_r                                                                      # Repitition time
            frame_times = np.arange(n_scans) * t_r                                              # Volume acquisition sequence


            # Drop explicit fixation trials if True
            if drop_fixation:
                  events = events[events['trial_type'] != 'fixation'].reset_index(drop=True)

            # Aggregate block_type and trial_type regressors if True
            if auto_block_regressors:
                  events['trial_type'] = events.apply(block_regressor, axis=1)
                  events.drop(columns=['block_type'], inplace=True)

            # Set object conditions to match trial types
            clean_conditions = list(events['trial_type'].dropna().unique())
            self.set_conditions(clean_conditions)

            # Create baseline Design Matrix with defined parameters
            events = first_level.make_first_level_design_matrix(frame_times, events, hrf_model='spm').reset_index()

            # We assume there are no dummy scans unless otherwise specified
            try:
                  dummy_value = iso_container['dummy']
            except:
                  dummy_value = 0

            # Base list of confound regressors
            confound_vars = self.confound_regressors.copy()

            # Add non_steady_state regressor to confounds if True
            if non_steady_state:
                  confounds['non_steady_state'] = aggregate_non_steady_state(length=n_scans, dummy_value=dummy_value)
                  confound_vars += ['non_steady_state']

            # Pull in motion outliers to confounds if True
            if motion_outliers:
                  confound_vars += [x for x in list(confounds.columns) if 'motion_outlier' in x]

            # Pull in anatomical noise component derivatives if True
            if a_comp_cor:
                  confound_vars += [x for x in list(confounds.columns) if "a_comp_cor" in x]

            # Pull in temporal noise component derivatives if True
            if t_comp_cor:
                  confound_vars += [x for x in list(confounds.columns) if "t_comp_cor" in x]

            # Isolate confound regressor
            confounds = confounds.loc[:, confound_vars].reset_index()

            # Merge events and confounds into one DataFrame
            events = events.merge(confounds, on='index', how='left')

            # We'll ONLY mean impute missing whole-brain motion regressors (FD and DVARS)            
            for var in events.columns:

                  if 'dvars' in var.lower():
                        try:
                              events[var].fillna(np.mean(events[var]), inplace=True)
                        except:
                              continue
                  elif 'framewise' in var.lower():
                        try:
                             events[var].fillna(np.mean(events[var]), inplace=True)
                        except:
                             continue
                        

            # Rearrange Design Matrix columns
            events = reorder_design_columns(events)

            return events



      def first_level_design(self, non_steady_state: bool=False, auto_block_regressors: bool=False,
                            motion_outliers: bool=True, drop_fixation: bool=True, verbose: bool=True, 
                            a_comp_cor: bool=True, t_comp_cor: bool=True) -> list:
            """
            Compiles one design matrix per functional run

            Parameters
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from event files

            Returns
                  List of Pandas DataFrame objects (one per functional run)
            """

            model_specs = f"""\n\nRunning first-level designs for {self.task.upper()} with the following parameters:\n\n
Non-steady state regressors:\t\t{non_steady_state}\n
Auto-block regressors:\t\t\t{auto_block_regressors}\n
Motion outliers:\t\t\t{motion_outliers}\n
Fixation trials:\t\t\t{drop_fixation}\n
Anatomical noise components:\t\t{a_comp_cor}\n
Temporal noise components:\t\t{t_comp_cor}
\n\n
            """
            
            if verbose:
                  print(model_specs)
            sleep(1)

            # Empty list to append into
            matrices = []

            # Loop through functional runs
            for k in range(self.functional_runs):
                  
                  # k = 0 => run-1
                  run = k+1

                  # Generate design matrix given the input parameters
                  matrix = self.generate_design_matrix(
                        run, non_steady_state=non_steady_state,
                        auto_block_regressors=auto_block_regressors, 
                        motion_outliers=motion_outliers, 
                        drop_fixation=drop_fixation,
                        a_comp_cor=a_comp_cor,
                        t_comp_cor=t_comp_cor
                  )

                  # Output filename
                  filename = f'sub-{self.sub_id}_task-{self.task}_run-{run}_design-matrix.jpg'

                  # Aggregate output file path
                  output_path = os.path.join(self.first_level_output, 'plots', filename)

                  # Save design matrix plot to output directory
                  nip.plot_design_matrix(matrix, output_file=output_path)

                  # Add Design Matrix to matrices list
                  matrices.append(matrix)

            return matrices



      def _default_contrasts(self) -> dict:
            """
            Creates trial-wise contrast map to feed into first-level GLM

            Returns
                  Dictionary (keys and values == contrast to compute)
            """

            # Empty dictionary to append into
            contrasts = {}

            # Match trial types as 1-1 contrasts
            for outer in self.conditions:
                  for inner in self.conditions:
                        if outer != inner:

                              # Avoid duplicates - these can be modeled using directionality
                              if (f'{inner}-{outer}') not in list(contrasts.keys()):
                                    k = f'{outer}-{inner}'
                                    contrasts[k] = k

            return contrasts



      def _run_contrast(self, glm: first_level.FirstLevelModel, contrast: str, 
                        title: str, output_type: str, smoothing: float, 
                        plot_brains: bool=False, plot_type: str="stat"):
            """
            Linear contrast based on first-level GLM. NifTi output is always saved, stat maps are conditional

            Parameters
                  glm: FirstLevelModel object | Model is fit outside of this function
                  contrast: str | Mathematical contrast to compute (e.g., "high_trust_perspective - self_perspective")
                  title: str | Nominal contrast to feed into title of plot (e.g., "perspective_contrast")
                  output_type: str | Must be in ['condition', 'contrast']
                  smoothing: float | Smoothing kernel from GLM
                  plot_brains: boolean | if True, stat maps and contrast reports are saved locally
                  plot_type: str | 'stat' or 'glass'
            """

            # Remove any white space from contrast
            contrast = str(contrast).replace(' ', '').strip()
            
            # Cast kernel to string
            kernel = str(int(smoothing))

            if output_type == 'condition':
                  model_base = os.path.join(self.first_level_output, 'models', 'condition-maps')
                  plot_base = os.path.join(self.first_level_output, 'plots', 'condition-maps')

            elif output_type == 'contrast':
                  model_base = os.path.join(self.first_level_output, 'models', 'contrast-maps')
                  plot_base = os.path.join(self.first_level_output, 'plots', 'contrast-maps')

            else:
                  raise ValueError(f'Output type {output_type} invalid - must be condition or contrast')

            
            # Output name and path for computed contrast
            nifti_output = f'sub-{self.sub_id}_{output_type}-{title}_smoothing-{kernel}mm_z-map.nii.gz'
            nifti_output = os.path.join(model_base, nifti_output)

            # Run the contrast itself
            z_map = glm.compute_contrast(contrast)

            # Save contrast to output directory
            z_map.to_filename(nifti_output)

            # Plot stat map and reports if True
            if plot_brains:

                  # Formatted output paths
                  base_output = f"{plot_base}/sub-{self.sub_id}_{output_type}-{title}_smoothing-{kernel}"

                  stat_output = f'{base_output}mm_plot-stat-map.png'
                  glass_output = f'{base_output}mm_plot-glass-map.png'
                  report_output = f'{base_output}mm_contrast-summary.html'

                  # Plot stat map
                  if plot_type == "stat":
                        nip.plot_stat_map(
                              z_map, 
                              threshold=2.3, 
                              colorbar=False, 
                              draw_cross=False,
                              display_mode='ortho', 
                              title=title,
                              output_file=stat_output)

                  elif plot_type == "glass":
                        nip.plot_glass_brain(
                              z_map, 
                              threshold=2.3, 
                              plot_abs=False,
                              display_mode="lyrz", 
                              title=title,
                              output_file=glass_output)

                  # Make GLM report
                  make_glm_report(
                        model=glm, 
                        contrasts=contrast,
                        plot_type='glass').save_as_html(report_output)



      def run_first_level_glm(self, conditions: bool=True, contrasts: bool=True, smoothing: float=8., 
                              plot_brains: bool=True, user_design_matrices: list=None,
                              non_steady_state: bool=False, auto_block_regressors: bool=False,
                              motion_outliers: bool=True, drop_fixation: bool=True, 
                              verbose: bool=True, plot_type: str="stat", a_comp_cor: bool=True, 
                              t_comp_cor: bool=True):
            """
            Instantiates and fits a FirstLevelModel GLM object, compiles condition and contrast z-maps

            Parameters
                  conditions: boolean | if True, condition-wise contrast maps are computed
                  contrasts: boolean | if True, contrast-wise contrast maps are computed
                  smoothing: float | Kernel size to apply to contrast maps, default = 8.
                  plot_brains: boolean | if True, local contrast images are saved to the first level output directory
                  user_design_matrices: Defaults to None | if supplied, these replace the Object-generated design matrices
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from events file
                  plot_type: str | stat or glass
            """

            # Users have the option of specifying their own design matrices if they like
            if user_design_matrices is not None:
                  matrices = user_design_matrices

                  print("\n== PLEASE NOTE: We assume you have provided deisgn matrices in order (i.e,., run-1 == first matrix)\n\n")

                  for index, matrix in enumerate(matrices):
                        filename = f"sub-{self.sub_id}_task-{self.task}_user-defined-matrix_run-{index+1}.jpg"
                        output_path = os.path.join(self.first_level_output, "plots", filename)

                        nip.plot_design_matrix(matrix, output_file=output_path)

            else:
                  matrices = self.first_level_design(non_steady_state=non_steady_state,
                                                     auto_block_regressors=auto_block_regressors, 
                                                     motion_outliers=motion_outliers, 
                                                     drop_fixation=drop_fixation, 
                                                     verbose=verbose, 
                                                     a_comp_cor=a_comp_cor, 
                                                     t_comp_cor=t_comp_cor)
            
            # If user has not provided updated contrast map, the default pairwsie contrasts will run
            if self.contrasts == 'default':
                  contrasts_to_map = self._default_contrasts()

            else:
                  contrasts_to_map = self.contrasts


            # === Data type checks ===
            if isinstance(matrices, list):
                  if not isinstance(matrices[0], pd.DataFrame):
                        raise TypeError('Design matrices should be Pandas DataFrame objects!')

            elif not isinstance(matrices, pd.DataFrame):
                  raise TypeError('Design matrices should be Pandas DataFrame objects!')

            
            # === Build first-level GLM ===
            glm = first_level.FirstLevelModel(t_r=self.t_r, smoothing_fwhm=smoothing, hrf_model='spm', minimize_memory=False)

            if verbose:
                  print('\n=== Fitting GLM ===')

            # Fit to functional runs and design matrices
            model = glm.fit(self.bids_container['all_preprocessed_bold'], 
                            design_matrices=matrices)


            # === Conditionally run condition and contrast maps ===
            if verbose:
                  disable = False
            else:
                  disable = True


            if conditions:
                  if verbose:
                        print('\n=== Mapping condition z-scores ===\n')
                  
                  for condition in tqdm(self.conditions, disable=disable):
                        self._run_contrast(glm=model, contrast=condition, title=condition, 
                                           output_type='condition', smoothing=smoothing, 
                                           plot_brains=plot_brains,
                                           plot_type=plot_type)

            if (contrasts) and (len(self.contrasts)) > 0:
                  if verbose:
                        print('\n=== Mapping contrast z-scores ===\n')

                  for k in tqdm(list(contrasts_to_map.keys()), disable=disable):
                        self._run_contrast(glm=model,
                                           contrast=contrasts_to_map[k], 
                                           title=k,
                                           output_type='contrast', 
                                           smoothing=smoothing, 
                                           plot_brains=plot_brains,
                                           plot_type=plot_type)

            if verbose:
                  print(f'\n\n=== {self.task.upper()} contrasts computed! Subject {self.sub_id} has been mapped ===\n\n') 
#!/bin/python3

"""
About this Class

Subject inherits all functions and attributes from the BIDSPointer class defined in this package. 
Building on this framework, Subject is optimized to run basic first-level GLMs out of the box, 
and users have the option to define their own design matrices as well.

Ian Richard Ferguson | Stanford University
"""

# ---- Imports
from glm_express.subject.bids_pointer import Build_Subject
import os, json
from time import sleep
import pandas as pd
import numpy as np
from tqdm import tqdm
import nilearn.plotting as nip
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report

# ---- Object definition
class Subject(Build_Subject):

      # ---- Class functions
      def __init__(self, sub_id, task, bids_root, suppress=False, template_space="MNI152NLin2009",
                  repetition_time=1., dummy_scans=0):

            # Inherits constructor from BIDSPointer
            Build_Subject.__init__(self, sub_id=sub_id, task=task, bids_root=bids_root, suppress=suppress,
                                   template_space=template_space, repetition_time=repetition_time,
                                   dummy_scans=dummy_scans)

            # Boolean - are modulators provided or not?
            self.has_modulators = self._has_modulators()

            if self.has_modulators:

                  # Read in modulators from task_information.json file
                  self.modulators = self._parse_modulators()
                  
                  # Get mean values for each modulator
                  self._center_modulators()


      # ---- Modeling helpers
      def _parse_modulators(self):
            """
            Reads in modulators from task_information.json

            Returns
                  List of user provided modulators
            """

            with open('./task_information.json') as incoming:
                  data = json.load(incoming)[self.task]

            return list(data['modulators'])


      def _has_modulators(self):
            """
            Returns
                  Boolean | if True, modulators were provided
            """

            with open('./task_information.json') as incoming:
                  data = json.load(incoming)[self.task]

            return len(data['modulators']) > 1


      def _center_modulators(self):
            """
            Calculates average values from modulators, to use for centering in design stage
            """

            # Read in all events
            all_events = self.load_events()

            # Modulators key = empty dictionary to append into
            self.bids_container['modulators'] = {}

            # Loop through user-provided modulators
            for mod in self.modulators:
                  
                  """
                  NOTE: We assume that all modulators are contained in your events file!

                  If your modulators are not in the events file, they'll be removed
                  from the Subject object
                  """
                  
                  if mod not in all_events.columns:
                        print(f'\n** ERROR: {mod.upper()} not in Events File\n')
                        self.modulators.remove(mod)
                        continue

                  # Mean value from modulator column
                  avg = all_events[mod].mean()

                  # Assign to bids_container object
                  self.bids_container['modulators'][mod] = avg



      # ---- Modeling functions
      def generate_design_matrix(self, run, non_steady_state=False, include_modulators=False, 
                                 auto_block_regressors=False, motion_outliers=True, drop_fixation=True):
            """
            Builds a first level design matrix

            Parameters
                  run: int | Functional run number, lines up with events file
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  include_modulators: boolean | if True, Parametric modulators are calculated and included in design matrix
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from event files

            Returns
                  Pandas DataFrame object
            """

            # -- Helpers
            def aggregate_non_steady_state(length, dummy_value):
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


            def block_regressor(DF):
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


            def create_temp_dataframe(events, mod_name, frame_times):
                  """
                  Creates temporary DataFrame to merge into events DF (optimized to handle modulators)

                  Parameters
                        events: Pandas DataFrame |
                        mod_name: str |
                        frame_times: int |

                  Returns
                        Reduced Pandas DataFrame
                  """

                  # Nilearn implicitly requires a modulation column - we'll rename the column of interest here
                  temp = events.rename(columns={mod_name: 'modulation'})

                  # Calculate distance from the mean for the current modulation
                  temp['modulation'] = temp['modulation'] - self.bids_container['modulators'][mod_name]

                  # Rename trial_type => trial_type_x_{modulation}
                  temp['trial_type'] = temp['trial_type'].apply(lambda x: f'{x}_x_{mod_name}')

                  # These will be the only columns we output
                  output_columns = list(temp['trial_type'].unique())

                  # Translate DataFrame to Design Matrix
                  dm = first_level.make_first_level_design_matrix(frame_times,
                                                                  temp,
                                                                  hrf_model='spm')

                  # Return isolated trial x modulation columns AND index
                  return dm.loc[:, output_columns].reset_index()


            def reorder_design_columns(DM):
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


            # === Model foundations ===
            iso_container = self.bids_container[f'run-{run}']                                   # BIDS container for the current run
            events = pd.read_csv(iso_container['event'], sep='\t')                              # Load events file
            confounds = pd.read_csv(iso_container['confound'], sep='\t')                        # Load fmriprep regressors
            voi = ['onset', 'duration', 'trial_type']                                           # Starting variables of interest

            if auto_block_regressors:
                  if 'block_type' in events.columns:
                        voi += ['block_type']                                                   # Add block type to VOI if True

            if include_modulators:
                  voi += self.modulators                                                        # Add modulators to VOI if True

            events = events.loc[:, voi]                                                         # Reduce 
            

            # === Model parameters ===
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

            # Create mod-wise Design Matrices if True
            if include_modulators:
                  mod_matrices = []

                  for novel_modulator in self.modulators:
                        mod_matrices.append(create_temp_dataframe(events, novel_modulator, frame_times))


            # Set object conditions to match trial types
            clean_conditions = list(events['trial_type'].dropna().unique())
            self.set_conditions(clean_conditions)

            # Create baseline Design Matrix with defined parameters
            events = first_level.make_first_level_design_matrix(frame_times, events, hrf_model='spm').reset_index()

            # Merge modulated DataFrames into Design Matrix if True
            if include_modulators:
                  for matrix in mod_matrices:
                        events = events.merge(matrix, on='index', how='left')

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



      def first_level_design(self, non_steady_state=False, include_modulators=False, auto_block_regressors=False,
                            motion_outliers=True, drop_fixation=True, verbose=True):
            """
            Compiles one design matrix per functional run

            Parameters
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  include_modulators: boolean | if True, Parametric modulators are calculated and included in design matrix
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from event files

            Returns
                  List of Pandas DataFrame objects (one per functional run)
            """

            model_specs = f"""\n\nRunning first-level designs for {self.task.upper()} with the following parameters:\n\n
Non-steady state regressors:\t\t{non_steady_state}\n
Modulators:\t\t\t\t{include_modulators}\n
Auto-block regressors:\t\t\t{auto_block_regressors}\n
Motion outliers:\t\t\t{motion_outliers}\n
Fixation trials:\t\t\t{drop_fixation}
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
                  matrix = self.generate_design_matrix(run, non_steady_state, include_modulators,
                                                      auto_block_regressors, motion_outliers, drop_fixation)

                  # Output filename
                  filename = f'sub-{self.sub_id}_task-{self.task}_run-{run}_design-matrix.jpg'

                  # Aggregate output file path
                  output_path = os.path.join(self.first_level_output, 'plots', filename)

                  # Save design matrix plot to output directory
                  nip.plot_design_matrix(matrix, output_file=output_path)

                  # Add Design Matrix to matrices list
                  matrices.append(matrix)

            return matrices


      def _default_contrasts(self):
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



      def _run_contrast(self, glm, contrast, title, output_type, smoothing, plot_brains=False):
            """
            Linear contrast based on first-level GLM. NifTi output is always saved, stat maps are conditional

            Parameters
                  glm: FirstLevelModel object | Model is fit outside of this function
                  contrast: str | Mathematical contrast to compute (e.g., "high_trust_perspective - self_perspective")
                  title: str | Nominal contrast to feed into title of plot (e.g., "perspective_contrast")
                  output_type: str | Must be in ['condition', 'contrast']
                  smoothing: float | Smoothing kernel from GLM
                  plot_brains: boolean | if True, stat maps and contrast reports are saved locally
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
                  stat_output = f'{plot_base}/sub-{self.sub_id}_{output_type}-{title}_smoothing-{kernel}mm_plot-stat-map.png'
                  report_output = f'{plot_base}/sub-{self.sub_id}_{output_type}-{title}_smoothing-{kernel}mm_contrast-summary.html'

                  # Plot stat map
                  nip.plot_stat_map(z_map, threshold=2.3, colorbar=False, draw_cross=False,
                                    display_mode='ortho', title=title,
                                    output_file=stat_output)

                  # Make GLM report
                  make_glm_report(model=glm, contrasts=contrast,
                                plot_type='glass').save_as_html(report_output)



      def run_first_level_glm(self, conditions=True, contrasts=True, smoothing=8., plot_brains=True, user_design_matrices=None,
                              non_steady_state=False, include_modulators=False, auto_block_regressors=False,
                              motion_outliers=True, drop_fixation=True, verbose=True):

            """
            Instantiates and fits a FirstLevelModel GLM object, compiles condition and contrast z-maps

            Parameters
                  conditions: boolean | if True, condition-wise contrast maps are computed
                  contrasts: boolean | if True, contrast-wise contrast maps are computed
                  smoothing: float | Kernel size to apply to contrast maps, default = 8.
                  plot_brains: boolean | if True, local contrast images are saved to the first level output directory
                  user_design_matrices: Defaults to None | if supplied, these replace the Object-generated design matrices
                  non_steady_state: boolean | if True, all non steady state regressors are aggregated as one regressor
                  include_modulators: boolean | if True, Parametric modulators are calculated and included in design matrix
                  auto_block_regressors: boolean | if True, `block_type` and `trial_type` columns are implicitly merged
                  motion_outliers: boolean | if True, all motion outliers are included in design matrix
                  drop_fixation: boolean | if True, fixation trials are dropped from events file
            """

            # === Define DMs and contrasts ===
            # Users have the option of specifying their own design matrices if they like
            if user_design_matrices is not None:
                  matrices = user_design_matrices

            else:
                  matrices = self.first_level_design(non_steady_state=non_steady_state, include_modulators=include_modulators,
                                                   auto_block_regressors=auto_block_regressors, motion_outliers=motion_outliers,
                                                   drop_fixation=drop_fixation, verbose=verbose)
            
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
            model = glm.fit(self.bids_container['all_func'], design_matrices=matrices)

            # === Conditionally run condition and contrast maps ===
            if verbose:
                  disable = False
            else:
                  disable = True


            if conditions:
                  if verbose:
                        print('\n=== Mapping condition z-scores ===\n')
                  
                  for condition in tqdm(self.conditions, disable=disable):
                        self._run_contrast(glm=model, contrast=condition, title=condition, output_type='condition',
                                          smoothing=smoothing, plot_brains=plot_brains)

            if contrasts:
                  if verbose:
                        print('\n=== Mapping contrast z-scores ===\n')

                  for k in tqdm(list(contrasts_to_map.keys()), disable=disable):
                        self._run_contrast(glm=model, contrast=contrasts_to_map[k], title=k,
                                           output_type='contrast', smoothing=smoothing, plot_brains=plot_brains)

            if verbose:
                  print(f'\n\n=== {self.task.upper()} contrasts computed! Subject {self.sub_id} has been mapped ===\n\n') 

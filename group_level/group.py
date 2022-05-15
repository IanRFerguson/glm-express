#!/bin/python3

"""
About this Class

GroupLevel aggregates first-level contrast maps derived from the Subject class.

Ian Richard Ferguson | Stanford University
"""

# ---- Imports
import math, os, json, random, pathlib, glob
import pandas as pd
from nilearn.glm import second_level, threshold_stats_img
import nilearn.plotting as nip
from nilearn.reporting import get_clusters_table
import matplotlib.pyplot as plt
from bids.layout import BIDSLayout


# ---- Object definition
class GroupLevel:
      def __init__(self, task, bids_root='./bids'):
            """
            Parameters
                  task: str | Corresponds to functional task in your BIDS project
                  bids_root: str | Relative path to BIDS project
            """

            # === BIDS paths ===
            self.bids_root = bids_root
            self._layout = BIDSLayout(self.bids_root)
            self._all_subjects = self._layout.get_subjects()
            self.subjects = self._iso_BIDSsubjects()
            self._all_tasks = self._layout.get_tasks()

            # === Object attributes ===
            self.task = task
            self.output_path = self._build_output_directory
            self.all_contrasts = self._all_available_contrasts()
            self.task_file = self._taskfile_validator()



      def _taskfile_validator(self):
            """
            Confirms existence of task_information.json. This file should have been created
            in the Subject object!

            Returns
                  Task-specific parameters from JSON file
            """

            if not os.path.exists('./task_information.json'):
                  raise OSError(f'Task information JSON file not found ... your directory: {os.getcwd()}')

            with open('./task_information.json') as incoming:
                  return json.load(incoming)[self.task]



      def _build_output_directory(self):
            """
            Builds out group-level directory hierarchy for second-level modeling

            Returns
                  Relative path to second-level output
            """

            # Base output directory
            target = os.path.join(self.bids_root, f'derivatives/second-level-output/task-{self.task}')

            # Create directory if it doesn't exist
            if not os.path.exists(target):
                  pathlib.Path(target).mkdir(exist_ok=True, parents=True)

            # Create corrected / uncorrected subdirs for models and plots
            for secondary in ['models', 'plots']:
                  for tertiary in ['corrected', 'uncorrected']:
                        temp = os.path.join(target, secondary, tertiary)

                        if not os.path.exists(temp):
                              pathlib.Path(temp).mkdir(exist_ok=True, parents=True)

            return target



      def _iso_BIDSsubjects(self):
            """
            Aggregates list of subjects that have undergone first-level modeling

            Returns
                  List of first-level modeled subjects
            """

            # List of ALL BIDS subjects
            bids_subjects = BIDSLayout(self.bids_root).get_subjects()

            # Relative path to first-level output directory
            base_path = os.path.join(self.bids_root, "derivatives", "first-level-output")

            # Empty list to append into
            keepers = []

            # Loop through all BIDS subjects
            for sub in bids_subjects:

                  # Keep subject if they have been modeled
                  temp_path = os.path.join(base_path, f"sub-{sub}", f"task-{self.task}")

                  if os.path.exists(temp_path):
                        keepers.append(sub)

            return keepers
            


      def get_brain_data(self, contrast, smoothing, discard_modulated=True, catch_duplicates=True):
            """
            Parameters
                  contrast: str | Contrast name derived from first-level modeling
                  smoothing: numeric or string | First-level smoothing maps of interest
                  discard_modulated: Boolean | if True, modulated maps not included implicitly  
                  catch_duplicates: Boolean | if True, error raised when # of maps exceeds # of subjects

            Returns
                  List of relative paths to contrast maps
            """

            # Output directory from first-level modeling
            derivatives = os.path.join(self.bids_root, 'derivatives/first-level-output')

            # User provides a numeric smoothing value
            if type(smoothing) in [int, float]:
                  smooth_string = f'{int(smoothing)}mm'

            # User is one step ahead of us
            elif "mm" in str(smoothing):
                  smooth_string = smoothing

            else:
                  smooth_string = f"{smoothing}mm"
            
            # List of NifTi files that match given contrast
            brain_maps = [x for x in glob.glob(f'{derivatives}/**/*.nii.gz', recursive=True) 
                         
                         # Identifiy maps for first-level contrast of interest
                         if contrast in x 
                         
                         # Identify maps with the appropriate first-level smoothing kernel
                         if smooth_string in x
                         
                         # Tosses regressor maps that were tossed out in first-level
                         if "_discard" not in x]

            """
            If True, modulated contrasts are not included unless explicitly specificed

            E.g.,
            If False:   high_trust_x_target, high_trust
            If True:    high_trust
            """

            if discard_modulated:
                  brain_maps = [x for x in brain_maps if f'{contrast}_x_' not in x]

            if catch_duplicates:
                  if len(brain_maps) > len(self.subjects):
                        raise ValueError(f"""
                        Please provide more specific contrast name!\n\n
                        {len(brain_maps)} first-level maps identified for {len(self.subjects)} subjects
                        """)

            if len(brain_maps) == 0:
                  raise ValueError(f"ERROR: No first-level maps identified for contrast {contrast}")

            return brain_maps



      def _all_available_contrasts(self):
            """
            Selects random participant, lists out their modeled conditions and contrasts,
            and aggregates in a dictionary

            Returns
                  Dictionary of conditions and contrasts
            """

            def iso_information(x, key):
                  
                  # Strips relative path information from filename
                  iso = os.path.basename(x)

                  # Isolate condition or contrast name directly
                  return iso.split(f"{key}-")[1].split("_smoothing")[0].strip()


            # TODO: Improve this appoach to be less subjective
            random_target = random.choice(self.subjects)

            # First-level models from target subject
            target = os.path.join(self.bids_root, 'derivatives/first-level-output',
                                 f'sub-{random_target}', f'task-{self.task}/models')

            # List and identify all condition files
            all_conditions = [x for x in glob.glob(os.path.join(target, "condition-maps/**/*.nii.gz"), recursive=True)]
            conditions = sorted([iso_information(x, "condition") for x in all_conditions])

            # List and identify all contrast files
            all_contrasts = [x for x in glob.glob(os.path.join(target, "contrast-maps/**/*.nii.gz"), recursive=True)]
            contrasts = sorted([iso_information(x, "contrast") for x in all_contrasts])

            # Returns dictionary of conditions and contrasts
            return {'conditions': conditions, 'contrasts': contrasts}



      # ---- Utility functions
      def plot_brain_mosaic(self, contrast, smoothing=8., save_local=False):
            """
            Parameters
                  contrast: str | Contrast name from first-level model
                  smoothing: float | Smoothing kernel from first-level model (this is a naming convention parameter)
                  save_local: Boolean | if True, saves to second-level-output directory
            """

            # List of relative paths to relevant NifTi files
            brain_maps = self.get_brain_data(contrast=contrast, smoothing=smoothing)

            # E.g., 8. => '8mm
            check_smooth = f'{int(smoothing)}mm'

            # Subset of NifTi files that match given smoothing kernel
            brain_maps = [x for x in brain_maps if check_smooth in x]


            # === Plot brains ===
            # Subplots rows
            rows = int(math.ceil(len(brain_maps) / 5))

            # Instantiate figure cavnas
            figure, axes = plt.subplots(nrows=rows, ncols=5, figsize=(20,20))

            if rows > 1:
                  for x in range(rows):
                        for y in range(5):
                              axes[x,y].axis('off')

            else:
                  for y in range(5):
                        axes[y].axis('off')

            figure.suptitle(f'{contrast}_{int(smoothing)}mm', fontweight='bold')

            for ix, brain in enumerate(brain_maps):

                  # Parse out subject ID
                  sub_id = brain.split('/')[-1].split('_')[0].split('-')[1]

                  # Slightly different logic depending on the shape of the output file (see axes)
                  if rows > 1:
                        k = nip.plot_glass_brain(brain, threshold=3.2, display_mode='z',
                                                plot_abs=False, colorbar=False, title=sub_id,
                                                axes=axes[int(ix/5), int(ix % 5)])

                  else:
                        k = nip.plot_glass_brain(brain, threshold=3.2, display_mode='z',
                                                plot_abs=False, colorbar=False, title=sub_id,
                                                axes=axes[int(ix % 5)])

            # Save plot locally if True
            if save_local:
                  filename = os.path.join(self.output_path, 'plots', f'brain_mosaic_{contrast}_{int(smoothing)}mm.jpg')
                  plt.savefig(filename)

            plt.show()



      # ---- Modeling functions
      def easy_loader(self, contrast, smoothing=8.):
            """
            Parameters
                  contrast: str | Contrast map derived from first-level model
                  smoothing: float | Smoothing kernel applied to first-level maps

            Returns
                  List of relative paths to all contrast-specific contrast maps
            """

            # Pattern to match in glob function
            f_path = f'{self.bids_root}/derivatives/first-level-output/**/*.nii.gz'

            # E.g., 8. => 8mm ... for naming convention purposes
            f_smooth = f'{int(smoothing)}mm'

            # Returns list of brain maps
            return [x for x in glob.glob(f_path) if contrast in x if f_smooth in x]



      def make_design_matrix(self, direction=1):
            """
            Parameters
                  direction: int | 1 or -1 (determines direction of linear contrast)

            Returns
                  Pandas DataFrame object
            """

            if direction not in [1, -1]:
                  raise ValueError(f'Direction must be 1 or -1 ... your input: {direction}')

            # List of subject id's
            subject_labels = self.subjects
            
            # [1] or [-1] * length of subject set
            design_matrix = pd.DataFrame({'subject_label': subject_labels,
                                           'intercept': [direction] * len(subject_labels)})

            return second_level.make_second_level_design_matrix(subject_labels, design_matrix)



      def uncorrected_model(self, contrast, sub_smoothing=8., group_smoothing=None, direction=1,
                           return_map=True, save_output=False):
            """
            Parameters
                  contrast: str | First-level contrast
                  sub_smoothing: float | Smoothing kernel used in first-level models
                  group_smoothing: float | Smoothing kernel to be applied to second-level model
                  direction: int | 1 or -1, determines direction of linear contrast

            Returns
                  nibabel.Image object (if return_map)
            """

            if direction not in [1, -1]:
                  raise ValueError(f'Direction must be 1 or -1 ... your input: {direction}')

            # List of brain maps matching contrast and first-level smoothing kernel
            brain_data = self.get_brain_data(contrast=contrast, smoothing=sub_smoothing)

            # Make second-level design matrix
            #design_matrix = self.make_design_matrix(direction=direction)
            design_matrix = pd.DataFrame([direction] * len(brain_data))

            print(f'\n ** {len(brain_data)} subjects in {contrast.upper()} group-level model **\n')

            # Instantiate SecondLevelModel object
            glm = second_level.SecondLevelModel(smoothing_fwhm=group_smoothing)

            # Fit brain data and 
            model = glm.fit(brain_data, design_matrix=design_matrix)

            # Run linear contrast on NifTi Image
            contrasted_model = model.compute_contrast(output_type='z_score')

            # Save local NifTi if True
            if save_output:

                  if group_smoothing is not None:
                        
                        # E.g., 8. => 8mm
                        smooth_string = f'{int(group_smoothing)}mm'

                        # Define output path for NifTi beta map
                        output_path = os.path.join(self.output_path, 
                                                  'models', 
                                                  f'second_level_contrast-{contrast}_smoothing-{smooth_string}.nii.gz')

                        # Save locally
                        contrasted_model.to_filename(output_path)

                  else:
                        # Define output path for NifTi beta map
                        output_path = os.path.join(self.output_path,
                                                  'models',
                                                  f'second_level_contrast-{contrast}_unsmoothed.nii.gz')

                        # Save locally
                        contrasted_model.to_filename(output_path)

            if return_map:
                  return contrasted_model



      def corrected_model(self, contrast, existing_model=None, height_control='fdr', alpha=.05,
                         plot_style='ortho', cluster_threshold=None, sub_smoothing=8.,
                         group_smoothing=None, direction=1, return_map=True, save_output=False):
            """
            Parameters
                  contrast: str | Valid contrast from first level model 
                  existing_model: SecondLevelModel object | This allows you to feed in uncorrected stats image
                  height_control: str | Must be in ['fpr', 'fdr', 'bonferroni']
                  alpha: float | P-value cutoff parameter for thresholding
                  plot_style: str | Must be in ['ortho', 'glass']
                  cluster_threshold: int | Only required for FPR height control models
                  sub_smoothing: float | Smoothing from first-level maps
                  group_smoothing: float | if not None, applies smoothing kernel to group-level map
                  direction: int | Must be in [1, -1]
                  reutrn_map: Boolean | if True, stats image is returned
                  save_output: Boolean | if True, plot is saved to output directory

            Returns
                  nibabel.Image object (if return_map)
            """

            # Catch any erroneous user inputs
            if height_control not in ['fdr', 'fpr', 'bonferroni']:
                  raise ValueError(f'Invalid height control parameter {height_control} supplied...')

            if plot_style not in ['ortho', 'glass']:
                  raise ValueError(f'Invalid plot_style parameter {plot_style} supplied...')

            if direction not in [1, -1]:
                  raise ValueError(f'Invalid contrast direction {direction} supplied...')


            # Fit model if none supplied
            if existing_model == None:

                  # List of NifTi files
                  brain_data = self.easy_loader(contrast=contrast, smoothing=sub_smoothing)
                  
                  # Build design matrix
                  design_matrix = self.make_design_matrix(direction=direction)
                  
                  # Instantiate SecondLevelModel
                  glm = second_level.SecondLevelModel(smoothing_fwhm=group_smoothing)
                  
                  # Fit to NifTi data
                  model = glm.fit(brain_data, design_matrix=design_matrix)

            else:
                  model = existing_model


            # === Modeling parameters ===
            if height_control != 'fpr':
                  t_map, t_thresh = threshold_stats_img(model, alpha=alpha, 
                                                       height_control=height_control)
            
            else:
                  t_map, t_thresh = threshold_stats_img(model, alpha=alpha,
                                                      height_control=height_control,
                                                      cluster_threshold=cluster_threshold)


            title = f'{contrast} @ {alpha}'

            if save_output:
                  
                  # Output name for plot
                  plot_filename = os.path.join(self.output_path, 'plots', f'{contrast}_corrected.jpg')
                  
                  # In development
                  # model_filename = os.path.join(self.output_path, 'models', f'{contrast}_corrected.nii.gz')

                  if plot_style == 'ortho':
                        nip.plot_stat_map(t_map, threshold=t_thresh, title=title,
                                          display_mode='ortho', output_file=plot_filename)

                  elif plot_style == 'glass':
                        nip.plot_glass_brain(t_map, threshold=t_thresh, display_mode='lyrz',
                                             plot_abs=False, colorbar=False, title=title,
                                             output_file=plot_filename)



      def get_clusters(self, stat_map, stat_thresh, cluster=None, two_sided=False, min_distance=8.):
            """
            This function is a wrapper for a similar function in Nilearn's reporting module

            Parameters
                  stat_map: Corrected NifTi image
                  stat_thresh: float | Should be identicial scale to stat_map param
                  cluster: int or None | Cluster size threshold for FDR
                  two_sided: Boolean | if True, employs two-sided evaluation of stat clusters
                  min_distance: float | Minimum distance between subpeaks in millimeters

            Returns
                  Cluster table (XYZ) for given NifTi stat map in Pandas DataFrame form
            """

            return get_clusters_table(stat_img=stat_map, stat_threshold=stat_thresh,
                                      cluster_threshold=cluster, two_sided=two_sided,
                                      min_distance=min_distance)
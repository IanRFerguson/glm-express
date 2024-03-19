import os

import bids
from tqdm import tqdm

from ..subject.subject import Subject
from ..utils.build import build_task_info

##########


class Aggregator:
    """
    Runs first-level GLMs on a given BIDS project.
    The user supplies the BIDS root and a functional task of interest
    and the Aggregator does the rest of the heavy lifting.
    """

    def __init__(
        self, bids_root: os.path, task: str, template_space: str = "MNI152NLin2009"
    ):
        """
        Parameters
              bids_root: str | Relative path to the top of your BIDS project (e.g., './bids')
              task: str | Functional task from your BIDS project (e.g., 'stopsignal')
              template_space: str | Template space for your preprocessed data (default='MNI152NLin2009')
        """

        self.bids_root = bids_root
        self.task = task
        self.template_space = template_space

        # List of subject ID's in your BIDS project
        self.subjects = self._derive_subjects()

    def _derive_subjects(self) -> list:
        """
        Returns
              List of subjects from your BIDS project
        """

        # Instantiate BIDSLayout object with your project root as the argument
        bids_ = bids.BIDSLayout(self.bids_root)

        # Return list of subject ID's
        return bids_.get_subjects()

    def _check_task_file(self) -> bool:
        """
        Returns
              Boolean | if True, the requisite task information file exists
        """

        return os.path.exists("./task_information.json")

    def run_all_models(
        self,
        smoothing: float = 8.0,
        verbose: bool = False,
        conditions: bool = True,
        contrasts: bool = True,
        plot_brains: bool = True,
        user_design_matrices: list = None,
        non_steady_state: bool = False,
        include_modulators: bool = False,
        auto_block_regressors: bool = False,
        motion_outliers: bool = True,
        drop_fixation: bool = True,
    ):
        """
        Loops through all subjects in BIDS project and fits first-level GLMs using
        Subject object and run_first_level_glm function

        Parameters
              smoothing: float | Smoothing kernel to apply to models (default = 8.)
              verbose: Boolean | if True, subject-wise model information will print
              conditions: Boolean | if True, unique trial type values are mapped
              contrasts: Boolean | if True, default or user-defined contrasts are mapped
              plot_brains: Boolean | if True, contrast-wise plots and summary files are saved locally
              user_design_matrices: list | You have the option to supply your own design matrices if you like (list of DataFrames)
              non_steady_state: Boolean | if True, non-steady-state regressors are automatically included as an aggregate regressor
              include_modulators: Boolean | if True, user-defined modulators are included in design matrices
              auto_block_regressors: Boolean | if True, `block_type` and `trial_type` regressors are implicitly merged
              motion_outliers: Boolean | if True, motion outlier regressors are automatically included as an aggregate regressor
              drop_fixation: Boolean | if True, explicit fixation trials are dropped from the design
        """

        # Build task information file if it doesn't already exist
        if not self._check_task_file():
            build_task_info(self.bids_root)
            print(
                "BE AWARE: task_information.json created ... your models will run with default parameters"
            )

        print(f"\n** Fitting first-level models for {len(self.subjects)} subjects **\n")

        # Loop through subject ID's
        for sub in tqdm(self.subjects):
            try:
                # Instantiate each subject as a Subject instance
                sub_instant = Subject(
                    sub,
                    task=self.task,
                    bids_root=self.bids_root,
                    suppress=True,
                    template_space=self.template_space,
                )

            except Exception as e:
                print(f"Error instantiating sub-{sub} ... {e}")
                continue

            try:
                # Run first-level GLM with your supplied arguments
                sub_instant.run_first_level_glm(
                    smoothing=smoothing,
                    verbose=verbose,
                    conditions=conditions,
                    contrasts=contrasts,
                    plot_brains=plot_brains,
                    user_design_matrices=user_design_matrices,
                    non_steady_state=non_steady_state,
                    include_modulators=include_modulators,
                    auto_block_regressors=auto_block_regressors,
                    motion_outliers=motion_outliers,
                    drop_fixation=drop_fixation,
                )

            except Exception as e:
                print(f"Error modeling sub-{sub} ... {e}")
                continue

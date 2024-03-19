"""
ABOUT -
"""

import glob
import os

from bids import BIDSLayout
from typing import Optional

#####


class Subject_Analytics:
    def __init__(
        self,
        subject_id: str,
        task: str,
        task_object: dict,
        bids_project_instance: BIDSLayout,
        full_data: bool = False,
    ):
        self.subject_id = subject_id
        self.task = task
        self.task_object = task_object
        self.bids_project_instance = bids_project_instance
        self.full_data = full_data

    def _isolate_files(
        self,
        prefix_pattern: str = "func",
        suffix_pattern: str = ".tsv",
        subject_prefix: str = "sub",
        derivatives_relative_path: str = "derivatives/fmriprep",
        filter_string: str = "event",
        full_data: bool = False,
    ) -> list:
        """
        Generic utility to isolate file types in the BIDS project

        Args:
            prefix_pattern:
            suffix_pattern:
            subject_prefix:
            derivatives_relative_path:
            filter_string:
            full_data:

        Returns:
            List of filepaths matching the input criteria
        """

        subject_id_string = f"{subject_prefix}-{self.subject_id}"

        if full_data:
            functional_path_ = os.path.join(
                self.bids_project_instance.root, subject_id_string, prefix_pattern
            )
        else:
            functional_path_ = os.path.join(
                self.bids_project_instance.root,
                derivatives_relative_path,
                subject_id_string,
                prefix_pattern,
            )

        if not suffix_pattern:
            suffix_pattern = "**"
        else:
            suffix_pattern = f"**/*{suffix_pattern}"

        glob_pattern = glob.glob(
            os.path.join(functional_path_, suffix_pattern), recursive=True
        )

        return [x for x in glob_pattern if self.task in x if filter_string in x]

    def _derive_functional_runs(self) -> int:
        if self._raw_bold_images:
            return len(self._raw_bold_images)

        # TODO - Add logging
        return 0

    def _initialize_glmx_container(self, run_values: Optional[list] = None) -> dict:
        """
        Initializes internal storage object to organize run-specific
        bold / event / confound information

        Args:
            run_values: If provided, used as keys in storage object
            (otherwise these are generated automatically). This should
            be a list of integer values -> `[1,2,3]` becomes `['run-1', 'run-2', 'run-3']`
            in this function

        Returns:
            Storage object
        """

        container = {}

        if not run_values:
            run_values = self._derive_functional_runs() + 1

        run_values = [f"run-{x}" for x in range(run_values)]

        container = {x: {} for x in run_values}

        container["all_raw_images"] = []
        container["all_events"] = []
        container["all_preprocessed_images"] = []
        container["all_confounds"] = []

        return container

    def _build_glmx_container(self, run_values: Optional[list] = None) -> dict:
        """
        Populates storage container object with subject metadata

        Returns:
            Object of subject metadata
        """

        container = self._initialize_glmx_container(run_values=run_values)

        raw_files = self._isolate_files()
        events_files = self._isolate_files()
        preprocessed_files = self._isolate_files()
        confound_files = self._isolate_files()

        for run_ in container.keys():
            current_raw_file = ""
            current_event_file = ""
            current_preprocessed_file = ""
            current_confound_file = ""

            container[run_] = {
                "raw_bold_image": current_raw_file,
                "event_file": current_event_file,
                "preprocessed_bold_image": current_preprocessed_file,
                "confound_file": current_confound_file,
            }

            container["all_raw_images"].append(current_raw_file)
            container["all_events"].append(current_event_file)
            container["all_preprocessed_images"].append(current_preprocessed_file)
            container["all_confounds"].append(current_confound_file)

    ###

    def load_subject_events_file(self, run: Optional[str] = None):
        pass

    def load_subject_confound_file(self, run: Optional[str] = None):
        pass

    ###

    def generate_design_matrix(self):
        pass

    def first_level_design(self):
        pass

    def run_single_contrast(self):
        pass

    def run_first_level_glm(self):
        pass

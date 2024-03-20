"""
ABOUT -
"""

import glob
import os

from bids import BIDSLayout
from glmx.utilities.logger import logger
import pandas as pd
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

        self.container = self._build_glmx_container()

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
        """
        Gets number of functional runs for the current task
        """

        all_scanner_files = [
            x
            for x in glob.glob(
                os.path.join(self.bids_project_instance.root, "**/*.nii.gz")
            )
            if self.subject_id in x
            if self.task in x
        ]

        # FIXME - Yikes!
        runs_ = set([x.split("run-")[1].split("_")[0] for x in all_scanner_files])

        return len(runs_)

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
            current_raw_file = self._isolate_file_per_run(
                run_value=run_, file_array=raw_files
            )
            current_event_file = self._isolate_file_per_run(
                run_value=run_, file_array=events_files
            )
            current_preprocessed_file = self._isolate_file_per_run(
                run_value=run_, file_array=preprocessed_files
            )
            current_confound_file = self._isolate_file_per_run(
                run_value=run_, file_array=confound_files
            )

            container[run_] = {
                "raw_bold_image": current_raw_file,
                "event_file": current_event_file,
                "preprocessed_bold_image": current_preprocessed_file,
                "confound_file": current_confound_file,
            }

            container["all_raw_images"].extend(current_raw_file)
            container["all_events"].extend(current_event_file)
            container["all_preprocessed_images"].extend(current_preprocessed_file)
            container["all_confounds"].extend(current_confound_file)

        return container

    def _isolate_file_per_run(self, run_value: str, file_array: list):
        """
        Attempts to pull filename out of list
        """

        matching_files = [x for x in file_array if run_value in x]

        if len(matching_files) > 1:
            logger.warning(
                f"{len(matching_files)} files found for sub-{self.subject_id} {run_value}"
            )
        elif len(matching_files) == 0:
            logger.error(f"No files matched for sub-{self.subject_id} {run_value}")

        return matching_files

    ###

    def load_subject_events_file(
        self, run: Optional[str] = None, csv_delimiter: str = "\t"
    ) -> pd.DataFrame:
        """
        Loads subject's events files (or, optionally, a run-specific file)

        Args:
            run: Should be provided in `run-{{ value }}` format. See `self.container.keys()` for reference
            csv_delimiter: Standard delimiter value, defaults to tab-delimiter

        Returns:
            Pandas Dataframe representing relevant event information
        """

        if run:
            run_values = [run]
        else:
            run_values = [x for x in self.container.keys() if "run" in x]

        files = []
        for val_ in run_values:
            temp_event_file = pd.read_csv(
                self.container[val_]["event_file"], sep=csv_delimiter
            )

            if "run" not in temp_event_file.columns:
                temp_event_file["run"] = [val_] * len(temp_event_file)

            files.append(temp_event_file)

        return pd.concat(files)

    def load_subject_confound_file(
        self, run: Optional[str] = None, csv_delimiter: str = "\t"
    ):
        """
         Loads subject's confounds files (or, optionally, a run-specific file)

        Args:
            run: Should be provided in `run-{{ value }}` format. See `self.container.keys()` for reference
            csv_delimiter: Standard delimiter value, defaults to tab-delimiter

        Returns:
            Pandas Dataframe representing relevant confound information
        """

        if run:
            run_values = [run]
        else:
            run_values = [x for x in self.container.keys() if "run" in x]

        files = []
        for val_ in run_values:
            temp_event_file = pd.read_csv(
                self.container[val_]["confound_file"], sep=csv_delimiter
            )

            if "run" not in temp_event_file.columns:
                temp_event_file["run"] = [val_] * len(temp_event_file)

            files.append(temp_event_file)

        return pd.concat(files)

    ###

    def generate_design_matrix(self):
        pass

    def first_level_design(self):
        pass

    def run_single_contrast(self):
        pass

    def run_first_level_glm(self):
        pass

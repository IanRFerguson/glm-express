"""
ABOUT - All classes in this project leverage several
shared fields that provide information about the tasks,
subjects, etc.
"""

import json
import os
import pathlib

from bids import BIDSLayout
from typing import Optional

#####


class GLMX_Base:
    def __init__(
        self,
        bids_project_root: str = "./bids",
        task: Optional[str] = None,
        suppress: bool = False,
        template_space: str = "MNI152NLin2009",
        **kwargs,
    ):
        # === Core attributes ===
        self.bids_project_root = bids_project_root
        self.task = task
        self.suppress = suppress
        self.template_space = template_space

        # === Optional attributes ===
        vars(self).update(kwargs=kwargs)

    @property
    def bids_project(self):
        return BIDSLayout(self.bids_project_root)

    def _load_project_task_file(self) -> dict:
        """
        Builds `task_information.json` if it doesn't exist
        """

        path_ = pathlib.Path(self.bids_project_root).parents[0]
        filepath_ = os.path.abspath(os.path.join(path_, "task_information.json"))

        if not os.path.exists(filepath_):
            raise OSError(
                f"Task information file has not been set up [expected_path={filepath_}]"
            )

        with open(filepath_) as file_:
            return json.load(file_)[self.task]

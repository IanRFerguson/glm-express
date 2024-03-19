"""
ABOUT
"""

import json

from .analytics import Subject_Analytics
from ..meta.project_metadata import GLMX_Base

#####


class Subject:
    """
    Analytical interface for single-subject functional
    neuroimaging analysis
    """

    def __init__(
        self,
        bids_project_root: str,
        subject_id: str,
        task: str,
        suppress: bool = False,
        template_space: str = "MNI152NLin2009",
        repetition_time: float = 1.0,
        dummy_scans: int = 0,
    ):
        GLMX_Base().__init__(
            bids_project_root=bids_project_root,
            subject_id=subject_id,
            task=task,
            template_space=template_space,
            repetition_time=repetition_time,
            dummy_scans=dummy_scans,
            suppress=suppress,
        )

        task_object = self._load_project_task_file()

        Subject_Analytics().__init__(
            subject_id=subject_id, task=task, task_object=task_object
        )

    def __str__(self) -> dict:
        return json.dumps(
            {
                "Subject ID": self.subject_id,
                "Task": self.task,
                "# of Functional Runs": self.functional_runs,
                "Output Directory": self.first_level_output,
                "Defined Contrasts": self.contrasts,
                "Confound Regressors": self.confound_regressors,
            },
            indent=4,
        )

"""
ABOUT -
"""

import glob
import os

#####


class Subject_Analytics:
    def __init__(
        self,
        subject_id: str,
        task: str,
        task_object: dict,
        bids_project_instance,
        full_data: bool = False,
    ):
        self.subject_id = subject_id
        self.task = task
        self.task_object = task_object
        self.bids_project_instance = bids_project_instance
        self.full_data = full_data

    def _isolate_files(self, pattern: str):
        pass

#!/bin/python3
import argparse, os
from .wrapper import build_task_info
from .general_utils import build_dataset_description
from bids import BIDSLayout


##########


def get_args():
    """
    Provides infrastructre to handle user-provided
    command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Provide relative path to BIDS project"
    )

    #
    parser.add_argument(
        "--path", type=str, required=True, help="Path to BIDS project, e.g., `./bids/`"
    )

    #
    parser.add_argument(
        "--validate",
        type="str",
        required=False,
        default="N",
        help="Validate BIDS project (Y/N)",
    )

    return parser.parse_args()


def glmx_setup():

    arguments = get_args()

    path_ = arguments.path
    val_ = arguments.validate.upper().strip()

    ######

    # Validate BIDS Project
    if val_ == "Y":
        try:
            BIDSLayout(path_)

        except Exception as e:
            print("\n** BIDS VALIDATION FAILED **\n")
            raise e

    ######

    # Build task information.json
    build_task_info(bids_root=path_, verbose=True)

    ######

    if not os.path.exists(os.path.join(path_, "dataset_description.json")):
        build_dataset_description(bids_root=path_)

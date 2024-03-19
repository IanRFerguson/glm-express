#!/bin/python3
"""
I'm glad you're here!

GLM Express is a lightweight, object-oriented solution
to first- and second-level modeling of functional neuroimaging data
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

__version__ = "2.0.3"

#####

from .aggregator.aggregator import Aggregator
from .group_level.group import GroupLevel
from .rest.resting_state import RestingState
from .subject.subject import Subject
from .utils.build import glmx_setup

#####

#!/bin/python3

"""
GLM Express is a lightweight, object-oriented approach
to first- and second-level modeling of functional neuroimaging data.
For a primer and demo, see github.com/IanRFerguson/glm-express
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

__version__ = "2.0.0"

from .aggregator.aggregator import Aggregator
from .subject.subject import Subject
from .group_level.group import GroupLevel
from .rest.resting_state import RestingState
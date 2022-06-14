#!/bin/python3

"""
GLM Express is a lightweight, object-oriented approach
to first- and second-level modeling of functional neuroimaging data.
For a primer and demo, see github.com/IanRFerguson/glm-express
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

__version__ = "1.0.7"

from glm_express.aggregator.aggregator import Aggregator
from glm_express.group_level.group import GroupLevel
from glm_express.subject.subject import Subject
from glm_express.rest.resting_state import RestingState
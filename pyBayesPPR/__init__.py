"""
A python package for Bayesian Projection Pursuit Regression

"""
__all__ = ["bppr", "bpprModel"]

__version__ = "1.0.2"

import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 9:
    raise ImportError("Python Version 3.9 or above is required for pyBayesPPR.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .pyBayesPPR import *

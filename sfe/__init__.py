from . import connect
from . import outputs
from . import ai
from . import connectors
from . import figures

from .core import (
    rolling_corr, rolling_drho, reff, reff_joint,
    pair_table, nonstationarity_flag, f_N,
    band_gap, reff_corrected,
    OPERATING_ENVELOPE,
)

__version__ = "0.1.0"
__author__  = "Jesus David Calderas Cervantes"
__license__ = "MIT"

__all__ = [
    # core numerical primitives
    "rolling_corr", "rolling_drho", "reff", "reff_joint",
    "pair_table", "nonstationarity_flag", "f_N",
    "band_gap", "reff_corrected",
    "OPERATING_ENVELOPE",
    # modules
    "connect",
    "outputs",
    "ai",
    "connectors",
    "figures",
]

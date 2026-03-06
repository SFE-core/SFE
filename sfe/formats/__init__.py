# -*- coding: utf-8 -*-
"""
sfe/formats/ — Automatic file format detection and connector routing.

This layer is invisible to users. Runners never import connectors directly;
they call sfe.formats.load() and get back an SFEResult regardless of file type.

    from sfe.formats import load

    result = load("trackedmodes.mat", domain="shm",  W=24)
    result = load("sampledata.csv",   domain="strain", W=60)
    result = load("S001R04.edf",      domain="eeg",    W=160)
    result = load("pems_bay.h5",      domain="traffic", W=12)
    result = load("prices.csv",       domain="finance", W=20)
"""

from .detect import load

__all__ = ["load"]

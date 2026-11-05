"""
Pipeline counts log (results/pipeline_counts.json).

Scripts 02 and 04 each write their own section(s), so the methods' PA and
transect counts can be regenerated from one file.
"""

import json
from pathlib import Path

import numpy as np


def _to_builtin(obj):
    """JSON fallback for numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def update_counts(path, section, data):
    """
    Write ``data`` under ``section`` in the counts JSON, keeping other sections.

    Parameters
    ----------
    path : str or Path
        Counts file (config.PIPELINE_COUNTS).
    section : str
        Section name, e.g. '02_pas', '02_transects', '04_ndvi'.
    data : dict
        JSON-serializable content (numpy scalars are converted).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = json.loads(path.read_text()) if path.exists() else {}
    counts[section] = data
    path.write_text(json.dumps(counts, indent=2, default=_to_builtin))

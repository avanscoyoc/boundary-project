"""
Pipeline counts log (results/pipeline_counts.json). Scripts 01, 02 and 04 each write their own
section(s), so the methods' PA and transect counts can be regenerated from one file.
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
    """Write `data` (JSON-serializable; numpy scalars are converted) as `section` (e.g. '02_pas',
    '04_ndvi') of the counts file `path`, keeping its other sections.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = json.loads(path.read_text()) if path.exists() else {}
    counts[section] = data
    path.write_text(json.dumps(counts, indent=2, default=_to_builtin))

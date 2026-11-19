"""
Modules package for Protected Area Edge Analysis.

This package contains domain-specific logic organized into:
- config: Configuration parameters and file paths
- remotesensing: Google Earth Engine functions (scripts 01, 03, 12)
- preprocessing: Geometry and transect generation functions (script 02)
- analysis: Data processing, edge metrics, and trend classification (script 04)
- validation: Displaced-transect validation (scripts 11-13)
- counts: Pipeline counts written to results/pipeline_counts.json
The R statistics (scripts 05-10, 14) are in modules/r/.
"""

__version__ = "1.0.0"

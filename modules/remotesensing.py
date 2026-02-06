"""
Google Earth Engine functions for remote sensing data collection.

This module contains functions for:
- Filtering protected area datasets
- Computing vegetation and built-up indices
- Generating spatial gradients
- Managing GEE task submissions
"""

import ee
from modules.config import (
    MIN_AREA_KM2, VALID_STATUS, EXCLUDED_PIDS,
    INDEX_CONFIGS, START_YEAR, END_YEAR
)


def get_pa_filter(type="Polygon"):
    """
    Create a combined filter for WDPA (World Database on Protected Areas) features.
    
    Filters out marine areas, marine protected areas, UNESCO biosphere reserves,
    and applies constraints on area, status, and excludes specific problem PIDs.
    
    Parameters
    ----------
    type : str, optional
        Geometry type to filter for (default is "Polygon").
    
    Returns
    -------
    ee.Filter
        Combined Earth Engine filter object for WDPA feature collection.
    
    Notes
    -----
    - Excludes marine areas (MARINE == "0")
    - Excludes Marine Protected Areas and UNESCO-MAB Biosphere Reserves
    - Only includes areas with status from config.VALID_STATUS
    - Requires minimum area from config.MIN_AREA_KM2
    - Excludes specific problematic WDPA_PIDs from config
    """
    polygon_filter = ee.Filter.eq("geometry_type", type)
    not_marine_filter = ee.Filter.eq("MARINE", "0")
    not_mpa_filter = ee.Filter.neq("DESIG_ENG", "Marine Protected Area")
    not_unesco_filter = ee.Filter.neq("DESIG_ENG", "UNESCO-MAB Biosphere Reserve")
    status_filter = ee.Filter.inList("STATUS", VALID_STATUS)
    area_filter = ee.Filter.gte("GIS_AREA", MIN_AREA_KM2)
    not_pids_filter = ee.Filter.inList("WDPA_PID", EXCLUDED_PIDS).Not()

    combined_filter = ee.Filter.And(
        polygon_filter,
        not_marine_filter,
        not_mpa_filter,
        not_unesco_filter,
        status_filter,
        area_filter,
        not_pids_filter
    )
    return combined_filter


def set_geometry_type(feature):
    """
    Add geometry type as a property to an Earth Engine feature.
    
    Parameters
    ----------
    feature : ee.Feature
        Earth Engine feature to process.
    
    Returns
    -------
    ee.Feature
        Feature with added 'geometry_type' property.
    """
    return feature.set('geometry_type', feature.geometry().type())


def get_biome(pa_feature):
    """
    Assign biome name to a protected area feature from RESOLVE Ecoregions.
    
    Uses the centroid of the protected area to determine which ecoregion it falls
    within, then extracts the biome name from that ecoregion.
    
    Parameters
    ----------
    pa_feature : ee.Feature
        Protected area feature from WDPA.
    
    Returns
    -------
    ee.Feature
        Protected area feature with added 'BIOME_NAME' property.
    
    Notes
    -----
    Uses RESOLVE/ECOREGIONS/2017 dataset. If no intersecting ecoregion is found,
    sets BIOME_NAME to 'Unknown'.
    """
    # Get ECOREGIONS dynamically
    ECOREGIONS = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    
    # Get centroid of protected area
    centroid = pa_feature.geometry().centroid()
    # Find which ecoregion contains this centroid
    intersecting_ecoregion = ECOREGIONS.filterBounds(centroid).first()
    # Get biome name from that ecoregion
    biome_name = ee.Algorithms.If(
        intersecting_ecoregion,
        intersecting_ecoregion.get('BIOME_NAME'),
        ee.String('Unknown')
    )
    return pa_feature.set('BIOME_NAME', biome_name)


def check_task_status(submitted_tasks):
    """
    Check status of submitted Earth Engine tasks and filter out completed ones.
    
    Prints status messages for completed, failed, or cancelled tasks and returns
    only tasks that are still running.
    
    Parameters
    ----------
    submitted_tasks : list of tuple
        List of tuples where each tuple contains (task_object, year).
        task_object is an ee.batch.Task instance.
    
    Returns
    -------
    tuple of (list, int)
        - active_tasks : list of tuple
            Filtered list containing only active tasks.
        - num_active : int
            Count of active tasks.
    
    Notes
    -----
    Completed states include: 'COMPLETED', 'FAILED', 'CANCELLED'.
    Active states include: 'READY', 'RUNNING'.
    """
    active_tasks = []
    for task_obj, year in submitted_tasks:
        task_status = task_obj.status()
        if task_status['state'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print(f"Task {year} {task_status['state']}")
        else:
            active_tasks.append((task_obj, year))
    return active_tasks, len(active_tasks)


def make_gradient(index_name, y):
    """
    Generic gradient function for computing spatial gradients of various indices.
    
    Computes annual median composites and their spatial gradient magnitude for
    NDVI, NDBI, LAI, or FPAR indices. Automatically applies appropriate scaling
    factors and cloud masking based on index configuration.
    
    Parameters
    ----------
    index_name : str
        Index to compute. Must be one of: 'ndvi', 'ndbi', 'lai', 'fpar'
    y : int or ee.Number
        Year to process (e.g., 2015)
    
    Returns
    -------
    ee.Image
        Single-band image containing gradient magnitude, with band name 'grad'.
        Unmasked areas are set to -9999.
    
    Examples
    --------
    >>> # Create gradient function for NDVI
    >>> ndvi_2015 = make_gradient('ndvi', 2015)
    >>> 
    >>> # Map over multiple years
    >>> years = ee.List.sequence(2001, 2021)
    >>> ndvi_grads = years.map(lambda y: make_gradient('ndvi', y))
    
    Notes
    -----
    Scale factors applied:
    - NDVI: 0.0001 (MOD13A1 stored as integers)
    - NDBI: 0.0001 (applied to reflectance bands before index calculation)
    - LAI: 0.1 (MCD15A3H stored as integers)
    - FPAR: 0.01 (MCD15A3H stored as integers)
    
    NDBI requires cloud masking using the state_1km QA band from MOD09GA.
    """
    config = INDEX_CONFIGS[index_name]
    collection = ee.ImageCollection(config['collection'])
    
    if index_name == 'ndbi':
        # Special handling for NDBI - compute from reflectance bands
        annual = (collection
                  .filter(ee.Filter.calendarRange(y, y, 'year'))
                  .select(['sur_refl_b02', 'sur_refl_b06'])
                  .median()
                  .multiply(config['scale_factor']))
        index_img = annual.normalizedDifference(['sur_refl_b06', 'sur_refl_b02'])
    else:
        # NDVI, LAI, FPAR - direct band selection
        annual = (collection
                  .filter(ee.Filter.calendarRange(y, y, 'year'))
                  .select(config['band'])
                  .median())
        if config['scale_factor'] != 1.0:
            annual = annual.multiply(config['scale_factor'])
        index_img = annual
    
    # Compute gradient magnitude
    grad = index_img.gradient()
    grad_mag = grad.select('x').hypot(grad.select('y')).unmask(-9999)
    return grad_mag.rename(['grad'])

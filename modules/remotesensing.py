"""
Google Earth Engine functions for remote sensing data collection.

This module contains functions for:
- Filtering protected area datasets
- Computing vegetation and built-up indices
- Generating spatial gradients
- Managing GEE task submissions
"""

import time
import ee
from modules.config import (
    MIN_AREA_KM2, VALID_STATUS, EXCLUDED_PIDS,
    INDEX_CONFIGS, INDEX_NAME, START_YEAR, END_YEAR
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


def get_gradient_magnitude(index_name, y):
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
    >>> ndvi_2015 = get_gradient_magnitude('ndvi', 2015)
    >>> 
    >>> # Map over multiple years
    >>> years = ee.List.sequence(2001, 2021)
    >>> ndvi_grads = years.map(lambda y: get_gradient_magnitude('ndvi', y))
    
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


def get_annual_gradient_magnitude(y):
    """
    Wrapper around get_gradient_magnitude using the index configured in config.INDEX_NAME.

    Designed for use with ee.List.map() to build a multi-year gradient image stack.

    Parameters
    ----------
    y : ee.Number
        Year to process, as passed by ee.List.map().

    Returns
    -------
    ee.Image
        Single-band gradient magnitude image for the configured index and year.
    """
    return get_gradient_magnitude(INDEX_NAME, y)


def batch_sample_assets(asset_path, image, selectors, folder_name, index_name,
                        chunk_size=50_000, batch_size=10, chunks_to_run=None):
    """
    Sample an image at transect points from a GEE asset and export results to Google Drive.

    Splits the asset FeatureCollection into sub-chunks, samples the provided image at
    each point using reduceRegions with ee.Reducer.first(), and submits batched export
    tasks to Google Drive. Waits for each batch to complete before submitting the next.

    If a task fails (typically due to GEE memory limits), it is automatically retried
    by splitting the failed chunk into two halves (chunk_size // 2). The two retry tasks
    are named with 'a' / 'b' suffixes (e.g., chunk_2a, chunk_2b) and are polled to
    completion before proceeding. No manual intervention is required.

    Parameters
    ----------
    asset_path : str
        Path to the Earth Engine FeatureCollection asset
        (e.g., 'projects/dse-staff/assets/chunk_000').
    image : ee.Image
        Multi-band image to sample (static layers + annual gradient bands).
    selectors : list of str
        Property and band names to include in the exported CSV.
    folder_name : str
        Google Drive folder name for export output.
    index_name : str
        Index name used in export task descriptions (e.g., 'ndvi', 'lai').
    chunk_size : int, optional
        Number of features per sub-chunk (default: 50,000).
    batch_size : int, optional
        Number of export tasks to submit and monitor simultaneously (default: 10).
    chunks_to_run : list of int, optional
        Specific sub-chunk indices to process. If None, all chunks are processed.
    """
    samples = ee.FeatureCollection(asset_path)
    size = samples.size().getInfo()
    nChunks = int((size + chunk_size - 1) // chunk_size)
    tasks = []

    if chunks_to_run is None:
        chunks_to_run = list(range(nChunks))

    asset_num = asset_path.split("_")[-1]

    for i in chunks_to_run:
        fcChunk = ee.FeatureCollection(samples.toList(chunk_size, i * chunk_size))
        sampled = image.reduceRegions(
            collection=fcChunk,
            reducer=ee.Reducer.first(),
            scale=500
        )
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f'{index_name}_raw_grad_{asset_num}_chunk_{i}',
            fileFormat='CSV',
            selectors=selectors,
            folder=folder_name
        )
        tasks.append((i, task))

    for j in range(0, len(tasks), batch_size):
        batch = tasks[j:j + batch_size]
        for idx, t in batch:
            t.start()

        chunk_nums = [idx for idx, _ in batch]
        print(f"  Processing chunks {chunk_nums}...")

        while True:
            statuses = [t.status()['state'] for _, t in batch]
            if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                print(f"  Completed chunks {chunk_nums}")
                break
            time.sleep(30)

        # Retry any failed chunks at half chunk_size
        failed = [(idx, t) for idx, t in batch if t.status()['state'] == 'FAILED']
        if failed:
            half = chunk_size // 2
            failed_nums = [idx for idx, _ in failed]
            print(f"  {len(failed)} chunk(s) failed: {failed_nums}. Retrying at half size ({half:,} points)...")
            retry_tasks = []
            for i, _ in failed:
                for sub, offset in enumerate([i * chunk_size, i * chunk_size + half]):
                    fcSub = ee.FeatureCollection(samples.toList(half, offset))
                    sampled = image.reduceRegions(
                        collection=fcSub,
                        reducer=ee.Reducer.first(),
                        scale=500
                    )
                    suffix = f'0{sub + 1}'
                    task = ee.batch.Export.table.toDrive(
                        collection=sampled,
                        description=f'{index_name}_raw_grad_{asset_num}_chunk_{i}_{suffix}',
                        fileFormat='CSV',
                        selectors=selectors,
                        folder=folder_name
                    )
                    retry_tasks.append(task)
                    task.start()

            print(f"  Polling {len(retry_tasks)} retry tasks...")
            while True:
                statuses = [t.status()['state'] for t in retry_tasks]
                if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                    still_failed = [t.config['description'] for t in retry_tasks
                                    if t.status()['state'] == 'FAILED']
                    if still_failed:
                        print(f"  WARNING: retry tasks still failed: {still_failed}")
                    else:
                        print(f"  Retry tasks completed successfully")
                    break
                time.sleep(30)
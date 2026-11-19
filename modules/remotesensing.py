"""
Google Earth Engine functions for remote sensing data collection.

This module contains functions for:
- Filtering protected area datasets (script 01)
- Computing NDVI gradient magnitude and the sampling image (scripts 03, 12)
- Managing GEE task submissions
"""

import time
import ee
from modules.config import (
    MIN_AREA_KM2, VALID_STATUS, EXCLUDED_PIDS, START_YEAR, END_YEAR,
    NDVI_COLLECTION, NDVI_BAND, NDVI_SCALE_FACTOR
)


def pa_filter_steps(type="Polygon"):
    """
    The WDPA filters of script 01, in order, as (name, ee.Filter) pairs.

    - terrestrial: MARINE == "0"
    - not_marine_protected_area: DESIG_ENG is not "Marine Protected Area"
    - not_unesco_mab: DESIG_ENG is not "UNESCO-MAB Biosphere Reserve"
    - status: STATUS in config.VALID_STATUS
    - min_area: GIS_AREA >= config.MIN_AREA_KM2
    - not_excluded_pid: WDPA_PID not in config.EXCLUDED_PIDS
    - single_part: geometry type `type` (default "Polygon"; drops multi-part PAs). Needs the
      geometry_type property of set_geometry_type(); last, so that counting the PAs left
      after each step (script 01) inspects only the geometries left by the attribute filters.

    The combined filter (get_pa_filter) gives the same PAs in any order.
    """
    return [
        ("terrestrial", ee.Filter.eq("MARINE", "0")),
        ("not_marine_protected_area", ee.Filter.neq("DESIG_ENG", "Marine Protected Area")),
        ("not_unesco_mab", ee.Filter.neq("DESIG_ENG", "UNESCO-MAB Biosphere Reserve")),
        ("status", ee.Filter.inList("STATUS", VALID_STATUS)),
        ("min_area", ee.Filter.gte("GIS_AREA", MIN_AREA_KM2)),
        ("not_excluded_pid", ee.Filter.inList("WDPA_PID", EXCLUDED_PIDS).Not()),
        ("single_part", ee.Filter.eq("geometry_type", type)),
    ]


def get_pa_filter(type="Polygon"):
    """All filters of pa_filter_steps() combined (ee.Filter.And)."""
    return ee.Filter.And(*[f for _, f in pa_filter_steps(type)])


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
    
    Finds all intersecting ecoregions and assigns the biome of the ecoregion
    with the largest intersection area with the protected area polygon.
    
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
    ECOREGIONS = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    pa_geom = pa_feature.geometry()

    # Get all ecoregions that intersect the PA
    intersecting = ECOREGIONS.filterBounds(pa_geom)

    # Compute actual intersection area for each candidate ecoregion
    def add_intersection_area(eco):
        area = eco.geometry().intersection(pa_geom, ee.ErrorMargin(1)).area()
        return eco.set('intersection_area', area)

    dominant = intersecting.map(add_intersection_area).sort('intersection_area', False).first()

    biome_name = ee.Algorithms.If(
        dominant,
        dominant.get('BIOME_NAME'),
        ee.String('Unknown')
    )
    return pa_feature.set('BIOME_NAME', biome_name)


def get_gradient_magnitude(y):
    """
    Compute the spatial gradient magnitude of the annual median NDVI composite.

    Builds the annual median of MOD13A1 NDVI for year ``y``, applies the scale
    factor (0.0001), and returns sqrt(gx^2 + gy^2) from ee.Image.gradient().

    Parameters
    ----------
    y : int or ee.Number
        Year to process (e.g., 2015).

    Returns
    -------
    ee.Image
        Single-band image 'grad' with the NDVI gradient magnitude.
        Masked pixels are set to -9999.

    Notes
    -----
    No quality (SummaryQA) masking is applied; the annual median uses every
    16-day composite in the year.

    The median composite is tagged with the native MOD13A1 projection (sinusoidal,
    ~463 m) so gradient() works on the native pixel grid rather than the default
    WGS84 projection of a composite. Sample it with the same projection
    (get_ndvi_projection) in reduceRegions. Earth Engine does not document the
    units of gradient(); the values are treated as relative contrast.
    """
    annual = (ee.ImageCollection(NDVI_COLLECTION)
              .filter(ee.Filter.calendarRange(y, y, 'year'))
              .select(NDVI_BAND)
              .median()
              .multiply(NDVI_SCALE_FACTOR)
              .setDefaultProjection(get_ndvi_projection()))

    grad = annual.gradient()
    grad_mag = grad.select('x').hypot(grad.select('y')).unmask(-9999)
    return grad_mag.rename(['grad'])


def get_ndvi_projection():
    """Native projection of MOD13A1 NDVI (MODIS sinusoidal, ~463 m pixels)."""
    return ee.ImageCollection(NDVI_COLLECTION).first().select(NDVI_BAND).projection()


def sampling_image(with_biome=False):
    """
    The image sampled at every transect point, by script 03 (PA transects) and script 12
    (their displaced copies), so both are measured identically.

    Bands, in order: max_extent (JRC Global Surface Water 1.4), gHM (global human
    modification, mean of the collection), elevation and terrain_slope (Copernicus GLO-30),
    BIOME_NUM (RESOLVE Ecoregions 2017; only with ``with_biome``, script 12), then one NDVI
    gradient band per year, named by the year (get_gradient_magnitude).

    Returns
    -------
    (ee.Image, list of str)
        The image, and the export selectors (point ID columns + band names).
    """
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    hm = ee.ImageCollection('CSP/HM/GlobalHumanModification').mean()
    # Copernicus DEM: tag the mosaic with the tiles' native projection. Without it,
    # ee.Terrain.slope works on the mosaic's default 1-degree grid and returns
    # near-zero or missing slopes (Earth Engine catalog: COPERNICUS/DEM/GLO30_2024_1).
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').select('DEM')
    elevation = dem.mosaic().setDefaultProjection(dem.first().projection()).rename('elevation')
    slope = ee.Terrain.slope(elevation).rename('terrain_slope')
    bands = [gsw.select('max_extent'), hm.rename('gHM'), elevation, slope]
    names = ['max_extent', 'gHM', 'elevation', 'terrain_slope']
    if with_biome:
        bands.append(ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
                     .reduceToImage(['BIOME_NUM'], ee.Reducer.first()).rename('BIOME_NUM'))
        names.append('BIOME_NUM')

    years = ee.List.sequence(START_YEAR, END_YEAR)
    grad_names = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    gradient = ee.ImageCollection.fromImages(years.map(get_gradient_magnitude)).toBands().rename(grad_names)
    image = ee.Image.cat(bands).addBands(gradient)
    return image, ['WDPA_PID', 'transectID', 'pointID'] + names + grad_names


def _retry(fn, what, attempts=10, wait=60):
    """
    Call ``fn()``; on an error (e.g. a dropped connection), wait ``wait`` seconds
    and try again, up to ``attempts`` times in total (~10 minutes by default).

    Only used for read-only requests (task status, collection size), never for
    starting tasks, so a retry can't submit an export twice.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"  {what} failed ({type(e).__name__}: {e}); "
                  f"retrying in {wait}s ({attempt}/{attempts - 1})", flush=True)
            time.sleep(wait)


def _state(task):
    """Task state ('READY', 'RUNNING', 'COMPLETED', 'FAILED', ...), with retries."""
    return _retry(lambda: task.status()['state'], 'Task status check')


def batch_sample_assets(asset_path, image, selectors, folder_name, export_prefix,
                        chunk_size=50_000, batch_size=10, chunks_to_run=None,
                        scale=500, crs=None):
    """
    Sample an image at transect points from a GEE asset and export results to Google Drive.

    Splits the asset FeatureCollection into sub-chunks, samples the provided image at
    each point using reduceRegions with ee.Reducer.first(), and submits batched export
    tasks to Google Drive. Waits for each batch to complete before submitting the next.

    Each sub-chunk is exported as '{export_prefix}{asset_num}_part_{i}' (e.g.
    ndvi_gradient_003_part_7). If a task FAILS (typically due to GEE memory
    limits), it is retried once by splitting it into two halves (chunk_size // 2),
    named with '_01' / '_02' suffixes (e.g. ndvi_gradient_003_part_7_01). CANCELLED
    tasks are not retried.

    Parameters
    ----------
    asset_path : str
        Path to the Earth Engine FeatureCollection asset
        (e.g., 'projects/my-project/assets/transect_points_000').
        The asset number is the text after the last '_'.
    image : ee.Image
        Multi-band image to sample (static layers + annual gradient bands).
    selectors : list of str
        Property and band names to include in the exported CSV.
    folder_name : str
        Google Drive folder name for export output.
    export_prefix : str
        Start of the export task / CSV names (e.g., 'ndvi_gradient_').
    chunk_size : int, optional
        Number of features per sub-chunk (default: 50,000).
    batch_size : int, optional
        Number of export tasks to submit and monitor simultaneously (default: 10).
    chunks_to_run : list of int, optional
        Specific sub-chunk indices to process. If None, all chunks are processed.
    scale : float, optional
        Sampling scale in meters for reduceRegions (default 500).
    crs : ee.Projection or str, optional
        Sampling projection for reduceRegions (default: the image's projection).
    """
    samples = ee.FeatureCollection(asset_path)
    size = _retry(lambda: samples.size().getInfo(), 'Asset size request')
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
            scale=scale,
            crs=crs
        )
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f'{export_prefix}{asset_num}_part_{i}',
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
            statuses = [_state(t) for _, t in batch]
            if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                print(f"  Completed chunks {chunk_nums}")
                break
            time.sleep(30)

        # Retry any failed chunks at half chunk_size
        failed = [(idx, t) for idx, t in batch if _state(t) == 'FAILED']
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
                        scale=scale,
                        crs=crs
                    )
                    suffix = f'0{sub + 1}'
                    task = ee.batch.Export.table.toDrive(
                        collection=sampled,
                        description=f'{export_prefix}{asset_num}_part_{i}_{suffix}',
                        fileFormat='CSV',
                        selectors=selectors,
                        folder=folder_name
                    )
                    retry_tasks.append(task)
                    task.start()

            print(f"  Polling {len(retry_tasks)} retry tasks...")
            while True:
                statuses = [_state(t) for t in retry_tasks]
                if all(s in ['COMPLETED', 'FAILED', 'CANCELLED'] for s in statuses):
                    still_failed = [t.config['description'] for t in retry_tasks
                                    if _state(t) == 'FAILED']
                    if still_failed:
                        print(f"  WARNING: retry tasks still failed: {still_failed}")
                    else:
                        print(f"  Retry tasks completed successfully")
                    break
                time.sleep(30)
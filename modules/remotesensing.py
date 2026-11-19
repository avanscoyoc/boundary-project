"""
Google Earth Engine functions: the WDPA filters and PA biome (script 01), the NDVI gradient and
sampling image (scripts 03, 12), and batched CSV exports with retries (batch_sample_assets).
"""

import time
import ee
from modules.config import (
    MIN_AREA_KM2, VALID_STATUS, EXCLUDED_PIDS, START_YEAR, END_YEAR,
    NDVI_COLLECTION, NDVI_BAND, NDVI_SCALE_FACTOR
)


def pa_filter_steps(type="Polygon"):
    """The WDPA filters of script 01, in order, as (name, ee.Filter) pairs. The single-part filter
    (`type`; needs set_geometry_type()) is last, so counting the PAs left after each step only
    inspects the geometries the attribute filters leave. get_pa_filter combines them (order does
    not change the result).
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
    """The feature with its geometry type as the property 'geometry_type'."""
    return feature.set('geometry_type', feature.geometry().type())


def get_biome(pa_feature):
    """The PA feature with BIOME_NAME: the biome of the RESOLVE Ecoregions 2017 ecoregion with the
    largest intersection with the PA polygon ('Unknown' if none intersects).
    """
    ECOREGIONS = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    pa_geom = pa_feature.geometry()
    intersecting = ECOREGIONS.filterBounds(pa_geom)

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
    """Gradient magnitude sqrt(gx² + gy²) of year `y`'s median MOD13A1 NDVI (all 16-day composites of
    the year, scaled by 0.0001; no quality masking), as band 'grad'; masked pixels are -9999.

    The median is tagged with the native MOD13A1 projection (sinusoidal, ~463 m), so gradient()
    works on the native pixel grid; sample it in that projection (get_ndvi_projection). Earth
    Engine does not document the units of gradient(): the values are relative contrast.
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
    """The image sampled at every transect point by script 03 (PA transects) and script 12 (their
    displaced copies), so both are measured identically. Bands: max_extent (JRC Global Surface
    Water 1.4), gHM (global human modification, 2016), elevation and terrain_slope (Copernicus
    GLO-30), BIOME_NUM (RESOLVE 2017; only `with_biome`, script 12), then one NDVI gradient band
    per year. Returns (image, export selectors).
    """
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    # Pinned to the 2016 image (currently the collection's only one), so additions can't change it
    hm = ee.Image('CSP/HM/GlobalHumanModification/2016')
    # Tag the DEM mosaic with the tiles' native projection. Without it, ee.Terrain.slope works on
    # the mosaic's default 1-degree grid and returns near-zero or missing slopes (Earth Engine
    # catalog: COPERNICUS/DEM/GLO30_2024_1).
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
    """Call ``fn()``, retrying on any error (e.g. a dropped connection) every ``wait`` s, up to
    ``attempts`` calls (~10 minutes by default). Only for read-only requests, never for starting
    tasks, so a retry can't submit an export twice."""
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
    """Sample `image` at the points of one Earth Engine asset (`asset_path`, ending in _NNN) and export
    CSVs to Drive folder `folder_name`.

    The asset is split into parts of `chunk_size` points, each sampled with reduceRegions
    (Reducer.first, in `crs` at `scale`) and exported as '{export_prefix}{NNN}_part_{i}'; tasks are
    submitted `batch_size` at a time, waiting for each batch. A FAILED part (typically Earth Engine
    memory) is retried once as two halves ('_01', '_02'); CANCELLED parts are not retried.
    `chunks_to_run` limits the parts (e.g. [0] for --test).
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

        # Retry failed chunks as two halves
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
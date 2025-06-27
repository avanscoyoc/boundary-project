from utils import *
from config import *
import pandas as pd
import concurrent.futures


def run_analysis(wdpaid, year, show_map=False, band_name=None):
    """Function to analyze habitat edge at protected area boundary"""
    # Initialize classes
    geo_ops = GeometryOperations()
    img_ops = ImageOperations()
    stats_ops = StatsOperations()
    viz = Visualization()
    feature_processor = FeatureProcessor(geo_ops, img_ops, stats_ops)
    exporter = ExportResults()

    # Load and process protected area geometry
    pa = load_protected_area(wdpaid)
    pa_geometry = pa.geometry()
    aoi = geo_ops.buffer_polygon(pa_geometry)
    aoi = geo_ops.mask_water(aoi)

    # Process imagery and add indices
    modis_ic = img_ops.modis.filter(img_ops.filter_for_year(aoi, year))
    band_names = modis_ic.first().bandNames()
    composite = modis_ic.reduce(ee.Reducer.median()).rename(band_names).clip(aoi)
    image = img_ops.add_indices_to_image(composite)

    # Process features and collect statistics
    feature_info = feature_processor.collect_feature_info(pa, aoi)
    features = feature_processor.process_all_bands_ee(image, pa_geometry, aoi, feature_info, year)
    stats_fc = ee.FeatureCollection(features)

    # Save results
    #exporter.export_table_to_cloud(wdpaid, year, stats_fc)
    task = ee.batch.Export.table.toCloudStorage(
        collection=stats_fc,
        description=f'{wdpaid}_{year}',
        bucket='dse-staff',
        fileNamePrefix=f'protected_areas/tables/{wdpaid}_{year}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Export task started for {wdpaid}, {year}")  

    # Visualization
    if show_map:
        band_stats = next(cs for cs in features if cs["band_name"] == band_name)
        Map = viz.create_map(pa_geometry, band_stats['buffer_pixels'], band_stats['boundary_pixels'])
        return Map
    
    return print("Analysis complete for WDPA ID:", wdpaid, "for the year:", year)


def run_all(wdpaids, start_year, n_years, max_workers=4):
    """
    Runs analysis for all wdpaids and years in parallel.
    Prints completion messages as each run finishes.
    """
    years = [start_year + i for i in range(n_years)]
    tasks = [(wdpaid, year) for wdpaid in wdpaids for year in years]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_analysis, wdpaid, year): (wdpaid, year)
            for wdpaid, year in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            wdpaid, year = future_to_task[future]
            try:
                future.result()
            except Exception as exc:
                print(f"WDPA ID {wdpaid}, Year {year} generated an exception: {exc}")
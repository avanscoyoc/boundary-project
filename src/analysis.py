from utils import *
from config import *
import pandas as pd
import concurrent.futures
import ee
import time


def run_analysis(wdpaid, year):
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
    print("Analysis complete for WDPA ID:", wdpaid, "for the year:", year)
    return task


def run_all(wdpaids, start_year, n_years, max_concurrent=12, poll_interval=30):
    """
    Submits up to max_concurrent GEE export tasks at a time, waits for completion before submitting more.
    """
    years = [start_year + i for i in range(n_years)]
    tasks = [(wdpaid, year) for wdpaid in wdpaids for year in years]
    task_queue = []

    for wdpaid, year in tasks:
        # Start a new export task
        task = run_analysis(wdpaid, year) 
        task_queue.append((task, wdpaid, year))

        # If we've reached the max_concurrent limit, wait for at least one to finish
        while len([t for t,_,_ in task_queue if t.active()]) >= max_concurrent:
            print(f"Waiting for tasks to finish... ({len(task_queue)} submitted)")
            time.sleep(poll_interval)
            # Remove finished tasks from the queue
            task_queue = [(t, w, y) for t, w, y in task_queue if t.active()]

    # Wait for all remaining tasks to finish
    while any(t.active() for t,_,_ in task_queue):
        print(f"Waiting for final tasks to finish... ({len(task_queue)} total)")
        time.sleep(poll_interval)
        task_queue = [(t, w, y) for t, w, y in task_queue if t.active()]

    print("All exports complete.")



def image_analysis(wdpaid, year, band_name):
    print(f"Processing WDPA ID {wdpaid} for year {year}")

    pa_geometry = load_protected_area(wdpaid).geometry()
    aoi = geo_ops.buffer_polygon(pa_geometry, 10000) 

    modis_ic = img_ops.modis.filter(img_ops.filter_for_year(aoi, year))
    band_names = modis_ic.first().bandNames()
    composite = modis_ic.reduce(ee.Reducer.median()).rename(band_names).clip(aoi)
    image = img_ops.add_indices_to_image(composite)
    single_band = image.select(band_name)

    buffer_img = img_ops.get_gradient_magnitude(single_band).clip(aoi)
    boundary_buffer_1km = geo_ops.buffer_polygon(pa_geometry, 1000)
    boundary_img = buffer_img.clip(boundary_buffer_1km)

    exporter.export_image_to_cloud(boundary_img, band_name, wdpaid, year)


def analysis_to_image(wdpaids, start_year, n_years, band_name, max_workers=4):
    years = [start_year + i for i in range(n_years)]
    tasks = [(wdpaid, year) for wdpaid in wdpaids for year in years]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(image_analysis, wdpaid, year, band_name): (wdpaid, year)
            for wdpaid, year in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            wdpaid, year = future_to_task[future]
            try:
                future.result()
            except Exception as exc:
                print(f"WDPA ID {wdpaid}, Year {year} generated an exception: {exc}")
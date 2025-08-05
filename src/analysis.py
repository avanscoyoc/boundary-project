from utils import *
from config import *
import pandas as pd
import concurrent.futures
import ee
import time


def run_analysis(wdpaid, year):
    """Full Analysis workflow: Function to analyze habitat edge at protected area boundary"""
    # Initialize classes
    geo_ops = GeometryOperations()
    img_ops = ImageOperations()
    stats_ops = StatsOperations()
    feature_processor = FeatureProcessor(geo_ops, img_ops, stats_ops)

    # Load and process protected area geometry
    pa = load_protected_area_by_id(wdpaid)
    pa_geometry = pa.geometry()
    aoi = geo_ops.buffer_polygon(pa_geometry)
    aoi = geo_ops.mask_water(aoi)

    # Process imagery, add indices
    modis_ic = img_ops.modis.filter(img_ops.filter_for_year(aoi, year))
    band_names = modis_ic.first().bandNames()
    composite = modis_ic.reduce(ee.Reducer.median()).rename(band_names).clip(aoi)
    image = img_ops.add_indices_to_image(composite)

    # Add feature info, process bands to calculate gradient, edge index, mean gHM
    feature_info = feature_processor.collect_feature_info(pa, aoi)
    features = feature_processor.process_all_bands_ee(image, pa_geometry, aoi, feature_info, year)
    stats_fc = ee.FeatureCollection(features)

    # Save results
    task = ee.batch.Export.table.toCloudStorage(
        collection=stats_fc,
        description=f'{wdpaid}_{year}',
        bucket='dse-staff',
        fileNamePrefix=f'protected_areas/movement/{wdpaid}_{year}_NDVI',
        fileFormat='CSV'
    )
    task.start()
    print(f"Analysis submitted for WDPA ID: {wdpaid} for the year: {year}")
    return task


def run_all(wdpaids, start_year, n_years, max_concurrent=12, poll_interval=30):
    """
    Iteration: Submits up to max_concurrent GEE export tasks at a time, counts progress, waits for completion before submitting more.
    """
    years = [start_year + i for i in range(n_years)]
    tasks = [(wdpaid, year) for wdpaid in wdpaids for year in years]
    task_queue = []
    counter = 0

    def get_concurrent_tasks():
        tasks = ee.data.getTaskList()
        return sum(1 for task in tasks if task['state'] == 'RUNNING')

    for wdpaid, year in tasks:
        # start a new export task
        task = run_analysis(wdpaid, year) 
        task_queue.append((task, wdpaid, year))
        # count submitted tasks
        counter += 1
        print(f"Submitted {counter}/{len(tasks)} tasks")

        # If we've reached the max_concurrent limit, wait for at least one to finish
        while len([t for t,_,_ in task_queue if t.active()]) >= max_concurrent:
            print(f"Waiting for tasks to finish... ({len(task_queue)} submitted)")
            time.sleep(poll_interval)
            #get concurrent_tasks()
            num_concurrent_tasks = get_concurrent_tasks()
            print('Number of concurrent tasks:', num_concurrent_tasks)
            # Remove finished tasks from the queue
            task_queue = [(t, w, y) for t, w, y in task_queue if t.active()]
    
    # Wait for all remaining tasks to finish
    while any(t.active() for t,_,_ in task_queue):
        print(f"Waiting for final tasks to finish... ({len(task_queue)} total)")
        time.sleep(poll_interval)
        task_queue = [(t, w, y) for t, w, y in task_queue if t.active()]

    print("All exports complete.")



# These functions are for exporting images with specific bands and buffer sizes, it takes too long to be relevant

def run_analysis_image(wdpaid, year, band_name, buffer_size):
    print(f"Processing WDPA ID {wdpaid} for year {year}")
    # initialize classes
    geo_ops = GeometryOperations()
    img_ops = ImageOperations()
    stats_ops = StatsOperations()
    viz = Visualization()
    feature_processor = FeatureProcessor(geo_ops, img_ops, stats_ops)
    exporter = ExportResults()

    pa_geometry = load_protected_area_by_id(wdpaid).geometry()
    aoi = geo_ops.buffer_polygon(pa_geometry, buffer_size)
    aoi = geo_ops.mask_water(aoi)

    modis_ic = img_ops.modis.filter(img_ops.filter_for_year(aoi, year))
    band_names = modis_ic.first().bandNames()
    composite = modis_ic.reduce(ee.Reducer.median()).rename(band_names).clip(aoi)
    image = img_ops.add_indices_to_image(composite)
    single_band = image.select(band_name)
    buffer_img = img_ops.get_gradient_magnitude(single_band).clip(aoi)

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f'image_{wdpaid}_{year}_{band_name}_{buffer_size}',
        bucket='dse-staff', 
        fileNamePrefix=f'protected_areas/images/{wdpaid}_{year}_{band_name}_{buffer_size}',  
        fileFormat='GeoTIFF', 
        formatOptions={
            'cloudOptimized': True,  
        },
        maxPixels=1e8,  
        scale=500  
    )
    task.start()
    print("Analysis complete for image of WDPA ID:", wdpaid, "for the year:", year, "with band:", band_name, "and buffer size:", buffer_size)
    return task


def run_all_image(wdpaids, start_year, n_years, band_names, buffer_size, max_concurrent=12, poll_interval=30):
    """
    Submits up to max_concurrent GEE export tasks at a time, waits for completion before submitting more.
    """
    years = [start_year + i for i in range(n_years)]
    tasks = [(wdpaid, year, band_name) for wdpaid in wdpaids for year in years for band_name in band_names]
    task_queue = []

    for wdpaid, year, band_name in tasks:
        # Start a new export task
        task = run_analysis_image(wdpaid, year, band_name, buffer_size) 
        task_queue.append((task, wdpaid, year, band_name))

        # If we've reached the max_concurrent limit, wait for at least one to finish
        while len([t for t,_,_,_ in task_queue if t.active()]) >= max_concurrent:
            print(f"Waiting for tasks to finish... ({len(task_queue)} submitted)")
            time.sleep(poll_interval)
            # Remove finished tasks from the queue
            task_queue = [(t, w, y, b) for t, w, y, b in task_queue if t.active()]

    # Wait for all remaining tasks to finish
    while any(t.active() for t,_,_,_ in task_queue):
        print(f"Waiting for final tasks to finish... ({len(task_queue)} total)")
        time.sleep(poll_interval)
        task_queue = [(t, w, y, b) for t, w, y, b in task_queue if t.active()]

    print("All exports complete.")
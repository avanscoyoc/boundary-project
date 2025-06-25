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
    aoi_with_biome = geo_ops.get_biome(aoi)

    # Process imagery and add indices
    modis_ic = img_ops.modis.filter(img_ops.filter_for_year(aoi, year))
    band_names = modis_ic.first().bandNames()
    composite = modis_ic.reduce(ee.Reducer.median()).rename(band_names).clip(aoi)
    image = img_ops.add_indices_to_image(composite)

    # Process features and collect statistics
    feature_info = feature_processor.collect_feature_info(pa, aoi_with_biome)
    computed_stats = feature_processor.process_all_bands(image, pa_geometry, aoi)
    all_stats = feature_processor.compile_statistics(feature_info, computed_stats, year)
    
    # Save results
    df = pd.DataFrame(all_stats)
    exporter.save_df_to_gcs(df, 'dse-staff', wdpaid, year)

    # Visualization
    if show_map:
        band_stats = next(cs for cs in computed_stats if cs["band_name"] == band_name)
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
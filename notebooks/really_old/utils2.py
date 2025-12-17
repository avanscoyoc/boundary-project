import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from glob import glob
import ee
import geemap
import os
import io
import pandas as pd
from datetime import datetime
from google.cloud import storage
import matplotlib.pyplot as plt
import tifffile
import imageio
import numpy as np
from IPython.display import Image, display


class NDVIPipeline:
    def __init__(self, max_error=1):
        self.water_mask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('max_extent')
        self.max_error = max_error

    # --- Geometry processing ---
    def buffer_polygon(self, geom, buffer_distance=10000):
        """Create donut-shaped buffered polygon (outer buffer minus inner buffer)."""
        out = geom.buffer(buffer_distance)
        inn = geom.buffer(-buffer_distance)
        aoi = out.difference(inn, self.max_error)
        return aoi

    def mask_water(self, feat):
        """Mask water bodies from feature."""
        water_no_holes = self.water_mask.focalMax(radius=30, units='meters', kernelType='square')\
                                        .focalMin(radius=30, units='meters', kernelType='square')
        water_vect = water_no_holes.reduceToVectors(
            reducer=ee.Reducer.countEvery(),
            geometry=feat.buffer(1000),
            scale=30,
            maxPixels=1e10,
            geometryType='polygon',
            eightConnected=False)
        geom = feat.difference(water_vect.geometry(), maxError=self.max_error)
        return geom

    def get_biome(self, geom):
        """Get biome with largest overlap and return biome name string."""
        ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
        intersecting = ecoregions.map(lambda eco: eco.set(
            'intersection_area', 
            eco.geometry().intersection(geom).area()
        )).filterBounds(geom)
        largest_ecoregion = intersecting.sort('intersection_area', False).first()
        biome_name = ee.Algorithms.If(
            largest_ecoregion,
            largest_ecoregion.get('BIOME_NAME'),
            ee.String('Unknown')
        )
        return biome_name

    # --- MODIS NDVI processing pipeline functions ---
    def mask_modis_clouds(self, image):
        QA = image.select('StateQA')
        # Keep only pixels where bits 0-1 == 0 (clear)
        cloud_state = QA.bitwiseAnd(3)  # bits 0-1
        mask = cloud_state.eq(0)
        return image.updateMask(mask)

    def add_ndvi(self, image):
        ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI')
        return image.addBands(ndvi).copyProperties(image, ['system:time_start'])

    def prepare_modis_collection(self, aoi, start_date, end_date):
        col = (ee.ImageCollection('MODIS/061/MOD09A1')
               .filterDate(start_date, end_date).filterBounds(aoi))
        print("MODIS images before cloud mask:", col.size().getInfo())
        first_img = col.first()
        print("First image bands:", first_img.bandNames().getInfo())
        col = (col
            .map(self.mask_modis_clouds)
            .map(self.add_ndvi)
            .filter(ee.Filter.listContains("bandNames", "NDVI"))
            .filter(ee.Filter.notNull(['NDVI'])))
        print("MODIS images after cloud mask and NDVI:", col.size().getInfo())
        return col

    def monthly_median(self, collection):
        def by_month(month):
            start = ee.Date.fromYMD(2001, month, 1)
            end = start.advance(1, 'month')
            filtered = collection.filterDate(start, end)
            median = filtered.median().set('month', month).set('system:time_start', start.millis())
            return median.select('NDVI')
        months = ee.List.sequence(1, 12)
        monthly = ee.ImageCollection(months.map(by_month))
        return monthly

    def rolling_median(self, collection, window=3):
        """
        Apply rolling median smoothing over monthly NDVI ImageCollection.
        This function assumes that each image has a 'month' property from 1 to 12.
        """
        def roll(month):
                month = ee.Number(month)
                half_window = ee.Number((window - 1) // 2)

                window_start = month.subtract(half_window).max(1)
                window_end = month.add(half_window).min(12)

                valid_months = ee.List.sequence(window_start, window_end)

                # Filter images by 'month' property
                filtered = collection.filter(ee.Filter.inList('month', valid_months))
                median = filtered.median().set('month', month).set('system:time_start', ee.Date.fromYMD(2001, month, 1).millis())

                return median

        months = ee.List.sequence(1, 12)
        return ee.ImageCollection(months.map(roll))

    def annual_median(self, monthly_collection):
        def by_year(year):
            start = ee.Date.fromYMD(year, 1, 1)
            end = start.advance(1, 'year')
            filtered = monthly_collection.filterDate(start, end)
            median = filtered.median().set('year', year).set('system:time_start', start.millis())
            return median.select('NDVI')
        years = ee.List.sequence(2001, 2023)
        annual = ee.ImageCollection(years.map(by_year))
        return annual

    def extract_time_series(self, annual_collection, geom, reducer=ee.Reducer.median()):
        """
        Extract annual NDVI time series for a single polygon geometry.
        Returns a pandas DataFrame with columns ['year', 'NDVI']
        """
        def reduce_image(img):
            stat = img.reduceRegion(
                reducer=reducer,
                geometry=geom,
                scale=500,
                maxPixels=1e13
            )
            ndvi = stat.get('NDVI')
            year = ee.Date(img.get('system:time_start')).get('year')
            return ee.Feature(None, {'year': year, 'NDVI': ndvi})

        ts = annual_collection.map(reduce_image).filter(ee.Filter.notNull(['NDVI']))
        ts_list = ts.getInfo()['features']

        data = []
        for f in ts_list:
            props = f['properties']
            if props['NDVI'] is not None:
                data.append({'year': int(props['year']), 'NDVI': float(props['NDVI'])})
        df = pd.DataFrame(data).sort_values('year')
        return df

# Optional: harmonic modeling (simplified)
   # 8. Optional: Harmonic modeling (simple example using GEE harmonic regression)
    def harmonic_modeling(annual_collection):
        # This example fits a harmonic model to each pixel's NDVI time series,
        # here simplified to return the fitted trend image.
        # In practice, harmonic modeling is usually done on monthly or more frequent data.
        def harmonicFit(imageCollection):
            # Requires time in fractional years for regression independent variable
            def addTime(image):
                date = ee.Date(image.get('system:time_start'))
                year = date.get('year')
                doy = date.getRelative('day', 'year')
                fracYear = year.add(doy.divide(365))
                return image.addBands(ee.Image(fracYear).rename('t').float())
            
            withTime = imageCollection.map(addTime)

            # Prepare regression
            harmonicTerms = withTime.map(lambda image: image
                                        .addBands(image.select('t').multiply(2 * np.pi).cos().rename('cos'))
                                        .addBands(image.select('t').multiply(2 * np.pi).sin().rename('sin')))
            
            # Stack bands for regression: NDVI, t, cos, sin
            dependent = harmonicTerms.select(['NDVI'])
            independent = harmonicTerms.select(['t', 'cos', 'sin'])

            # Run linear regression per pixel
            regression = dependent.addBands(independent).reduce(ee.Reducer.linearRegression(numX=3, numY=1))

            coefficients = regression.select('coefficients').arrayProject([0]).arrayFlatten([['Intercept', 't', 'cos', 'sin']])

            return coefficients

        coeffs = harmonicFit(annual_collection)
        return coeffs  # coefficients image you can visualize or export


    # === Full pipeline for single input polygon geometry ===
    def run_pipeline(self, input_geom, buffer_distance=10000, start='2001-01-01', end='2023-12-31',
                     cloud_bits=[0,1], apply_smoothing=True, apply_harmonic=False):
        # 1. Buffer polygon (donut shape)
        buffered_geom = self.buffer_polygon(input_geom, buffer_distance)

        # 2. Mask water within buffered geometry
        water_masked_geom = self.mask_water(buffered_geom)

        # 3. Get biome name string
        biome_name = self.get_biome(water_masked_geom).getInfo()
        print(biome_name)
        # 4. Prepare MODIS collection and NDVI calculation
        modis_col = self.prepare_modis_collection(water_masked_geom, start, end)
        size = modis_col.size().getInfo()
        if size == 0:
            raise ValueError("MODIS collection is empty after filtering — check dates, cloud mask, or geometry.")

        # 5. Monthly median aggregation
        monthly = self.monthly_median(modis_col)

        # 6. Optional rolling median smoothing
        if apply_smoothing:
            monthly = self.rolling_median(monthly, window=3)

        # 7. Annual median aggregation
        annual = self.annual_median(monthly)

        # 8. Optional harmonic modeling
        harmonic_results = None
        if apply_harmonic:
            harmonic_results = self.harmonic_modeling(monthly)

        # 9. Extract NDVI time series for water-masked buffered polygon
        ndvi_ts = self.extract_time_series(annual, water_masked_geom)

        # Return biome and NDVI DataFrame and optionally harmonic coefficients
        return {
            'biome': biome_name,
            'ndvi_timeseries': ndvi_ts,
            'harmonic': harmonic_results
        }

    # Additional helper functions for export, plotting, trend from previous step can be added here or outside the class
import matplotlib.pyplot as plt
from scipy import stats

# --- Existing functions above remain unchanged ---

def export_results_to_csv(results_dict, folder_path='./ndvi_results'):
    """
    Export NDVI time series results dictionary to CSV files.
    Each polygon's data is saved as {folder_path}/ndvi_{polygon_id}.csv
    """
    import os
    os.makedirs(folder_path, exist_ok=True)
    for polygon_id, df in results_dict.items():
        filename = f"{folder_path}/ndvi_{polygon_id}.csv"
        df.to_csv(filename, index=False)
        print(f"Exported {filename}")

def plot_ndvi_timeseries(results_dict, polygons_to_plot=None):
    """
    Plot NDVI time series for specified polygons.
    polygons_to_plot: list of polygon IDs to plot (default: all)
    """
    if polygons_to_plot is None:
        polygons_to_plot = list(results_dict.keys())
    plt.figure(figsize=(10,6))
    for pid in polygons_to_plot:
        df = results_dict.get(pid)
        if df is not None and not df.empty:
            plt.plot(df['year'], df['NDVI'], marker='o', label=f'Polygon {pid}')
    plt.xlabel('Year')
    plt.ylabel('NDVI')
    plt.title('Annual NDVI Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_trend(results_dict):
    """
    Compute linear trend (slope and p-value) for each polygon's NDVI time series.
    Returns a dict: {polygon_id: {'slope': slope, 'pvalue': pvalue}}
    """
    trends = {}
    for pid, df in results_dict.items():
        if df is not None and len(df) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['year'], df['NDVI'])
            trends[pid] = {'slope': slope, 'pvalue': p_value}
        else:
            trends[pid] = {'slope': None, 'pvalue': None}
    return trends

# === Example usage snippet ===
# results, _ = run_ndvi_pipeline(polygons, buffer_m=1000, apply_smoothing=True)
# export_results_to_csv(results, folder_path='./ndvi_output')
# plot_ndvi_timeseries(results)
# trends = compute_trend(results)
# print(trends)



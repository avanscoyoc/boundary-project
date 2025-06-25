from glob import glob
import ee
import geemap
import os
import io
import pandas as pd
from datetime import datetime
from google.cloud import storage
import matplotlib.pyplot as plt


class GeometryOperations:
    def __init__(self, max_error=1):
        self.max_error = max_error
        self.water_mask = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')

    def buffer_polygon(self, geom, buffer_distance=10000):
        """Create buffer around polygon"""
        out = geom.buffer(buffer_distance)
        inn = geom.buffer(-buffer_distance)
        aoi = out.difference(inn, self.max_error)
        return aoi

    def mask_water(self, feat):
        """Mask water bodies from feature"""
        water_no_holes = self.water_mask.select('max_extent')\
            .focalMax(radius=30, units='meters', kernelType='square')\
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
        """Get biome with largest overlap for a feature, add BIOME_NAME property"""
        ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
        intersecting = ecoregions.map(lambda eco: eco.set(
            'intersection_area', 
            eco.geometry().intersection(geom).area()
        )).filterBounds(geom)
        largest_ecoregion = intersecting.sort('intersection_area', False).first()
        result = ee.Feature(geom).set('BIOME_NAME', largest_ecoregion.get('BIOME_NAME'))
        return result
        

class ImageOperations:
    def __init__(self):
        self.modis = ee.ImageCollection('MODIS/006/MOD09A1')

    def filter_for_year(self, feat, year):
        """Filter images for specific year"""
        start = ee.Date.fromYMD(year, 1, 1)
        return ee.Filter.And(
            ee.Filter.bounds(feat),
            ee.Filter.date(start, start.advance(1, "year"))
        )

    def add_indices_to_image(self, image):
        """Add vegetation indices to image"""
        EVI = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                'NIR': image.select("sur_refl_b02"),
                'RED': image.select("sur_refl_b01"),
                'BLUE': image.select("sur_refl_b03")
            }
        ).rename("EVI")

        NDVI = image.expression(
            "(NIR - RED) / (NIR + RED)",
            {
                'NIR': image.select("sur_refl_b02"),
                'RED': image.select("sur_refl_b01")
            }
        ).rename("NDVI")

        return image.addBands([EVI, NDVI])

    def get_gradient_magnitude(self, image):
        """Calculate gradient magnitude"""
        gradient = image.gradient()
        gradient_x = gradient.select('x')
        gradient_y = gradient.select('y')
        magnitude = gradient_x.pow(2).add(gradient_y.pow(2)).sqrt()
        return magnitude


class StatsOperations:
    def __init__(self):
        self.gHM_collection = ee.ImageCollection('CSP/HM/GlobalHumanModification')

    def calculate_gradient_statistics(self, layer, name='buffer'):
        """Calculate mean and standard deviation of gradient magnitude"""
        stats = layer.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.count(),
                sharedInputs=True
            ),
            geometry=layer.geometry(),
            scale=500,
            maxPixels=1e10
        )
        return stats

    def calculate_human_modification(self, geometry, scale=500):
        """Calculate mean Global Human Modification value"""
        mean_gHM = self.gHM_collection.mean().clip(geometry)
        gHM_value = mean_gHM.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e9)
        return gHM_value.get('gHM')


class FeatureProcessor:
    def __init__(self, geo_ops, img_ops, stats_ops):
        self.geo_ops = geo_ops
        self.img_ops = img_ops
        self.stats_ops = stats_ops
        self.bands_to_process = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'EVI', 'NDVI']
        
    def collect_feature_info(self, pa, aoi_with_biome):
        """Collect basic protected area feature information"""
        return {
            'WDPA_PID': pa.get('WDPA_PID'),
            'ORIG_NAME': pa.get('ORIG_NAME'),
            'BIOME_NAME': aoi_with_biome.get('BIOME_NAME'),
            'GIS_AREA': pa.get('GIS_AREA')
        }
    
    def process_single_band(self, band_name, image, pa_geometry, aoi):
        """Process a single band and return its statistics"""
        single_band = image.select(band_name)
        buffer_img = self.img_ops.get_gradient_magnitude(single_band).clip(aoi)
        boundary = self.geo_ops.buffer_polygon(pa_geometry, 1000)
        boundary_img = buffer_img.clip(boundary)
        return {
            'band_name': band_name,
            'boundary_stats': self.stats_ops.calculate_gradient_statistics(boundary_img, name='boundary'),
            'buffer_stats': self.stats_ops.calculate_gradient_statistics(buffer_img, name='buffer'),
            'boundary_pixels': boundary_img,
            'buffer_pixels': buffer_img
        }

    def process_all_bands(self, image, pa_geometry, aoi):
        """Process all bands and collect their statistics"""
        return [self.process_single_band(band_name, image, pa_geometry, aoi)
                for band_name in self.bands_to_process]
    
    def compile_statistics(self, feature_info, computed_stats, year):
        """Compile all statistics into a list of dictionaries"""
        feature_values = {k: v.getInfo() for k, v in feature_info.items()}
        all_stats = []
        for stat in computed_stats:
            row_stats = {
                **feature_values,
                'band_name': stat['band_name'],
                'year': year,
                **{f"boundary_{k}": v for k, v in stat['boundary_stats'].getInfo().items()},
                **{f"buffer_{k}": v for k, v in stat['buffer_stats'].getInfo().items()}}
            all_stats.append(row_stats)       
        return all_stats


class ExportResults: 
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_df_to_gcs(self, df, bucket_name, wdpaid, year):
        """Save DataFrame as CSV and upload to GCS."""
        import io
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        client = storage.Client(project=bucket_name)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f'protected_areas/tables/_{wdpaid.replace(" ", "_")}_{year}_{self.timestamp}.csv')
        blob.upload_from_file(buffer, content_type='text/csv')
        print(f"Uploaded to: gs://{bucket_name}/{blob.name}")

    def export_image_to_cloud(self, image, wdpaid, year):
        """Save Image as a COG and upload to GCS."""
        filepath = f'protected_areas/images/_{wdpaid.replace(" ", "_")}_{year}_{self.timestamp}.csv'
        print(filepath)
        export_task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description='modis_export_cog',
        bucket='dse-staff', 
        fileNamePrefix=filepath,  
        fileFormat='GeoTIFF', 
        formatOptions={
            'cloudOptimized': True,  
        },
        maxPixels=1e8,  
        scale=30  
        )
        export_task.start()
        return print(f'{self.timestamp} saved')

    def combine_gcs_csvs(self, bucket_name, folder_path):
        """Combine all CSV files from a GCS folder into a single DataFrame."""
        bucket = storage.Client(bucket_name).bucket(bucket_name)
        dfs = [pd.read_csv(io.BytesIO(blob.download_as_bytes()))
               for blob in bucket.list_blobs(prefix=folder_path) if blob.name.endswith('.csv')]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class Visualization:
    def __init__(self): 
        self.default_vis_params = {
            'min': -0.5,
            'max': 1,
            'palette': ['black', 'gray', 'white']
        }

    def create_map(self, geometry, gradient, boundary_pixels, vis_params=None):
        """Create and return an interactive map"""
        if vis_params is None:
            vis_params = self.default_vis_params
        Map = geemap.Map()
        Map.add_basemap('HYBRID')
        Map.centerObject(geometry, 8)
        Map.addLayer(geometry, {'color': 'red'}, 'Protected Area Geometry')
        Map.addLayer(gradient, vis_params, 'Gradient Layer')
        Map.addLayer(boundary_pixels, vis_params, 'Gradient Boundary Pixels')
        return Map

    def plot_edge_ratio(self, df):
        """
        Plot edge_ratio by year for each WDPA and band.
        Line color = WDPA_PID, line style = band_name.
        """
        line_styles = ['-', '--', '-.', ':']
        bands = df['band_name'].unique()
        style_map = {band: line_styles[i % len(line_styles)] for i, band in enumerate(bands)}

        colors = plt.cm.tab10.colors  # Up to 10 distinct colors
        wdpaids = df['WDPA_PID'].unique()
        color_map = {wdpa: colors[i % len(colors)] for i, wdpa in enumerate(wdpaids)}

        plt.figure(figsize=(10, 6))
        for band in bands:
            for wdpa in wdpaids:
                sub = df[(df['band_name'] == band) & (df['WDPA_PID'] == wdpa)]
                if not sub.empty:
                    plt.plot(
                        sub['year'],
                        sub['edge_ratio'],
                        marker='o',
                        linestyle=style_map[band],
                        color=color_map[wdpa],
                        label=f'WDPA {wdpa}, Band {band}'
                    )
        plt.xlabel('Year')
        plt.ylabel('Edge Ratio')
        plt.legend()
        plt.tight_layout()
        plt.show()
import ee
import geemap
import os
import pandas as pd
from datetime import datetime

class GeometryOperations:
    def __init__(self, buffer_distance=10000, max_error=1):
        self.buffer_distance = buffer_distance
        self.max_error = max_error
        self.water_mask = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')

    def buffer_polygon(self, feat):
        """Create buffer around polygon"""
        feat = ee.Feature(feat)
        out = feat.buffer(self.buffer_distance).geometry()
        inn = feat.buffer(-self.buffer_distance).geometry()
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
    
    def get_pixels_boundary(self, image, polygon, scale=10):
        """Get pixels that straddle the 10km surrounding the polygon boundary"""
        outer_buffer = ee.Geometry(polygon).buffer(scale/2)
        inner_buffer = ee.Geometry(polygon).buffer(-scale/2)
        boundary_region = outer_buffer.difference(inner_buffer)
        boundary_pixels = image.updateMask(
            image.clip(boundary_region)
            .mask()
            .reduce(ee.Reducer.anyNonZero())
        )
        return boundary_pixels
        

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

    def calculate_gradient_statistics(self, layer scale=500, name='buffer'):
        """Calculate mean and standard deviation of gradient magnitude"""
        magnitude_mean = layer.reduceRegion(
            reducer=ee.Reducer.mean(),
            scale=scale,
            maxPixels=1e10)
        magnitude_stddev = layer.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            scale=scale,
            maxPixels=1e10)
        object_area = layer.mask().multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        scale=scale,
        maxPixels=1e10
    )
        return {
            f"{name}_mean": magnitude_mean.get('x'),
            f"{name}_stddev": magnitude_stddev.get('x'),
            f"{name}_area": object_area.get('x')
        }
    
    def calculate_human_modification(self, geometry, scale=500):
        """Calculate mean Global Human Modification value"""
        mean_gHM = self.gHM_collection.mean().clip(geometry)
        gHM_value = mean_gHM.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e9)
        return gHM_value.get('gHM')
    
    def create_statistics_row(self, feature, gradient_stats):
        """Create a dictionary of statistics for a single feature"""
        return {
            'WDPA_PID': feature.get('WDPA_PID').getInfo(),
            'ORIG_NAME': feature.get('ORIG_NAME').getInfo(),
            'GIS_AREA': feature.get('GIS_AREA').getInfo(),
            'buffer_mean': gradient_stats['buffer_mean'].getInfo(),
            'buffer_SD': gradient_stats['buffer_stddev'].getInfo(),
            'buffer_area': gradient_stats['buffer_area'].getInfo(),
            'boundary_mean': gradient_stats['boundary_mean'].getInfo(),
            'boundary_SD': gradient_stats['boundary_stddev'].getInfo(),
            'boundary_area': gradient_stats['boundary_area'].getInfo(),
            
        }


class ExportResults: 
    def __init__(self, results_path='/workspaces/jupyter-gee-project/results'):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)
    
    def generate_filename(self):
        """Generate standardized filename for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"output_{timestamp}.csv"
    
    def save_result_csv(self, filename):
        """Save results to a file"""
        filename = self.generate_filename()
        filepath = os.path.join(self.results_path, filename)
        results_data = {
            "PA Name": pa_name,
            "Band Name": band_name,
            "Year": year,
            "Buffer Distance": buffer_distance,
            "Max Error": max_error,
            "Image Count": image_count,
            "Gradient Buffer Area": gradient_buffer_area,
            "Gradient Buffer Mean": gradient_buffer_mean,
            "Gradient Buffer StdDev": gradient_buffer_stddev,
            "Gradient Boundary Area": gradient_boundary_area,
            "Gradient Boundary Mean": gradient_boundary_mean,
            "Gradient Boundary StdDev": gradient_boundary_stddev
        }
        df = pd.DataFrame([results_data])
        df.to_csv(filepath, index=False)
        print(f"Results saved locally to: {filepath}")
        return filepath


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
        Map.centerObject(geometry, 8)
        Map.addLayer(geometry, {'color': 'red'}, 'Protected Area Geometry')
        Map.addLayer(gradient, vis_params, 'Gradient Layer')
        Map.addLayer(boundary_pixels, vis_params, 'Gradient Boundary Pixels')
        return Map
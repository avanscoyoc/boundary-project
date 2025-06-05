import ee
import geemap


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
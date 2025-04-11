import ee
import geemap

ee.Initialize()

# Example: Print available MODIS image IDs
modis = ee.ImageCollection("MODIS/006/MOD09A1").limit(5)
ids = modis.aggregate_array("system:id").getInfo()
print("Sample MODIS Image IDs:", ids)

import ee
import pandas as pd
import geopandas as gpd


def filter_protected_areas():
    """Filter protected area dataset by size, status, designation, and marine status."""
    protected_areas = ee.FeatureCollection("WCMC/WDPA/202106/polygons")
    marine_filter = ee.Filter.eq("MARINE", "0")
    not_mpa_filter = ee.Filter.neq("DESIG_ENG", "Marine Protected Area")
    status_filter = ee.Filter.inList("STATUS", ["Designated", "Established", "Inscribed"])
    designation_filter = ee.Filter.neq("DESIG_ENG", "UNESCO-MAB Biosphere Reserve")
    area_filter = ee.Filter.gte("GIS_AREA", 200)
    excluded_pids = ["555655917", "555656005", "555656013", "555665477", "555656021",
                    "555665485", "555556142", "187", "555703455", "555563456", "15894"]
    pids_filter = ee.Filter.inList("WDPA_PID", excluded_pids).Not()
    combined_filter = ee.Filter.And(
        marine_filter,
        not_mpa_filter,
        status_filter,
        designation_filter,
        pids_filter,
        area_filter
       
    )
    data = protected_areas.filter(combined_filter)
    
    return data

def load_protected_area_by_id(wdpa_id):
    protected_areas = ee.FeatureCollection("WCMC/WDPA/current/polygons")
    pa = protected_areas.filter(ee.Filter.eq('WDPA_PID', wdpa_id)).first()

    return pa

def pas_at_movement_sites(sites): 
    """Filter protected areas to those that intersect with \
          the 50km buffer of centroid animal movement locations."""
    pas = filter_protected_areas()
    filtered = pas.filterBounds(sites)
    wdpa_pids_ee = filtered.aggregate_array('WDPA_PID')
    wdpaids = wdpa_pids_ee.getInfo()
    
    return wdpaids


def load_local_data(wdpa_id):
    """Load protected area from local shapefile and convert to EE Feature"""
    shp_path = '../data/global_wdpa_June2021/Global_wdpa_footprint_June2021.shp'
    try:
        # Read shapefile
        gdf = gpd.read_file(shp_path)
        pa_gdf = gdf[gdf['WDPA_PID'] == str(wdpa_id)]
        
        if len(pa_gdf) == 0:
            raise ValueError(f"Protected area {wdpa_id} not found")
        
        # Ensure WGS84 projection
        pa_gdf = pa_gdf.to_crs('EPSG:4326')
        pa_row = pa_gdf.iloc[0]
        
        # Get GeoJSON representation directly from GeoPandas
        geojson = pa_row.geometry.__geo_interface__
        
        # Create EE geometry from GeoJSON
        ee_geometry = ee.Geometry(geojson)
        
        # Clean properties for EE Feature
        properties = {}
        for col in pa_gdf.columns:
            if col != 'geometry' and pd.notnull(pa_row[col]):
                val = pa_row[col]
                # Convert numpy/pandas types to Python native types
                if hasattr(val, 'item'):
                    val = val.item()
                properties[col] = str(val)
        
        return ee.Feature(ee_geometry, properties)
        
    except Exception as e:
        print(f"Debug - Geometry type: {pa_row.geometry.geom_type}")
        print(f"Debug - First coordinate: {list(pa_row.geometry.coords)[0]}")
        raise Exception(f"Error loading shapefile: {str(e)}")
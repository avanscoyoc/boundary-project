import ee
import pandas as pd
import pyproj
import geopandas as gpd
from shapely.geometry import Polygon


class GeometryOperations:
    def __init__(self):
        self.ECOREGIONS = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    def get_pa_filter(self, type = "Polygon"):
        polygon_filter = ee.Filter.eq("geometry_type", type)
        not_marine_filter = ee.Filter.eq("MARINE", "0")
        not_mpa_filter = ee.Filter.neq("DESIG_ENG", "Marine Protected Area")
        not_unesco_filter = ee.Filter.neq("DESIG_ENG", "UNESCO-MAB Biosphere Reserve")
        status_filter = ee.Filter.inList("STATUS", ["Designated", "Established", "Inscribed"])
        area_filter = ee.Filter.gte("GIS_AREA", 200)
        excluded_pids = ["555655917", "555656005", "555656013", "555665477", "555656021",
                        "555665485", "555556142", "187", "555703455", "555563456", "15894"]
        not_pids_filter = ee.Filter.inList("WDPA_PID", excluded_pids).Not()

        combined_filter = ee.Filter.And(
            polygon_filter,
            not_marine_filter,
            not_mpa_filter,
            not_unesco_filter,
            status_filter,
            area_filter,
            not_pids_filter
        )
        return combined_filter

    def set_geometry_type(self, feature):
        return feature.set('geometry_type', feature.geometry().type())

    def get_biome(self, pa_feature):
        # Get centroid of protected area
        centroid = pa_feature.geometry().centroid()
        # Find which ecoregion contains this centroid
        intersecting_ecoregion = self.ECOREGIONS.filterBounds(centroid).first()
        # Get biome name from that ecoregion
        biome_name = ee.Algorithms.If(
            intersecting_ecoregion,
            intersecting_ecoregion.get('BIOME_NAME'),
            ee.String('Unknown')
        )
        return pa_feature.set('BIOME_NAME', biome_name)

    def fill_holes(self, gdf, max_hole_area=2250000):  #1500m * 1500m = 2250000 sq meters
        """Fill small holes in polygons using vector operations - handles MultiPolygons"""
        from shapely.geometry import MultiPolygon, Polygon
        
        filled_geoms = []
        for geom in gdf.geometry:
            if geom.geom_type == 'Polygon':
                # Handle single Polygon
                if hasattr(geom, 'interiors') and geom.interiors:
                    exterior = geom.exterior
                    large_holes = [interior for interior in geom.interiors 
                                  if Polygon(interior).area > max_hole_area]
                    filled_geoms.append(Polygon(exterior, large_holes))
                else:
                    filled_geoms.append(geom)
                    
            elif geom.geom_type == 'MultiPolygon':
                # Handle MultiPolygon - process each component polygon
                filled_polygons = []
                for poly in geom.geoms:
                    if hasattr(poly, 'interiors') and poly.interiors:
                        exterior = poly.exterior
                        large_holes = [interior for interior in poly.interiors 
                                      if Polygon(interior).area > max_hole_area]
                        filled_polygons.append(Polygon(exterior, large_holes))
                    else:
                        filled_polygons.append(poly)
                filled_geoms.append(MultiPolygon(filled_polygons))
            else:
                # Keep other geometry types unchanged
                filled_geoms.append(geom)
        
        result = gdf.copy()
        result['geometry'] = filled_geoms
        return result
    
    def find_overlap_groups(self, gdf, overlap_threshold=90):
        """Find groups of geometries that overlap above threshold"""
        print(f"Finding overlap groups with >{overlap_threshold}% overlap...")
        
        overlap_groups = []
        processed_indices = set()
        
        for idx, park in gdf.iterrows():
            if idx in processed_indices:
                continue
                
            current_group = [idx]
            current_geom = park['geometry']
            
            # Find all parks that overlap with current park
            for other_idx, other_park in gdf.iterrows():
                if other_idx == idx or other_idx in processed_indices:
                    continue
                    
                other_geom = other_park['geometry']
                
                try:
                    if current_geom.intersects(other_geom):
                        intersection = current_geom.intersection(other_geom)
                        if not intersection.is_empty:
                            intersection_area = intersection.area
                            overlap_pct_current = (intersection_area / current_geom.area) * 100
                            overlap_pct_other = (intersection_area / other_geom.area) * 100
                            max_overlap = max(overlap_pct_current, overlap_pct_other)
                            
                            if max_overlap > overlap_threshold:
                                current_group.append(other_idx)
                except:
                    continue
            
            # Mark all indices in this group as processed
            for group_idx in current_group:
                processed_indices.add(group_idx)
                
            overlap_groups.append(current_group)
        
        return overlap_groups

    def get_min_year_from_group(self, group_df):
        """Get the row with minimum non-zero STATUS_YR, or first row if all are 0"""
        non_zero = group_df[group_df['STATUS_YR'] != 0]
        if len(non_zero) > 0:
            return non_zero.loc[non_zero['STATUS_YR'].idxmin()]
        else:
            return group_df.iloc[0]
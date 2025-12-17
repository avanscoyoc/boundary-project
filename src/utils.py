import ee
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


# Earth Engine Functions
# =====================================================================

def get_pa_filter(type="Polygon"):
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


def set_geometry_type(feature):
    return feature.set('geometry_type', feature.geometry().type())


def get_biome(pa_feature):
    """Get biome name for a protected area feature from ecoregions."""
    # Get ECOREGIONS dynamically
    ECOREGIONS = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    
    # Get centroid of protected area
    centroid = pa_feature.geometry().centroid()
    # Find which ecoregion contains this centroid
    intersecting_ecoregion = ECOREGIONS.filterBounds(centroid).first()
    # Get biome name from that ecoregion
    biome_name = ee.Algorithms.If(
        intersecting_ecoregion,
        intersecting_ecoregion.get('BIOME_NAME'),
        ee.String('Unknown')
    )
    return pa_feature.set('BIOME_NAME', biome_name)


# Geometry Processing Functions  
# =====================================================================

def fill_holes(gdf, max_hole_area=2250000):  # 1500m * 1500m = 2250000 sq meters
    """Fill small holes in polygons using vector operations - handles MultiPolygons"""
    
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


def find_overlap_groups(gdf, overlap_threshold=90):
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


def get_min_year_from_group(group_df):
    """Get the row with minimum non-zero STATUS_YR, or first row if all are 0"""
    non_zero = group_df[group_df['STATUS_YR'] != 0]
    if len(non_zero) > 0:
        return non_zero.loc[non_zero['STATUS_YR'].idxmin()]
    else:
        return group_df.iloc[0]


# Transect Generation Functions
# =====================================================================

def evenspace(xy, sep, start=0):
    """
    Creates points along lines with a set distance.
    
    Parameters:
    -----------
    xy : array-like
        Nx2 array of coordinates (x, y)
    sep : float
        Separation distance between points
    start : float, optional
        Starting distance along the line (default=0)
    
    Returns:
    --------
    DataFrame with columns: x, y, x0, y0, x1, y1, theta
    """
    xy = np.array(xy)
    
    # Calculate differences and segment distances
    dx = np.concatenate([[0], np.diff(xy[:, 0])])
    dy = np.concatenate([[0], np.diff(xy[:, 1])])
    dseg = np.sqrt(dx**2 + dy**2)
    dtotal = np.cumsum(dseg)
    
    linelength = np.sum(dseg)
    
    # Generate positions along the line
    pos = np.arange(start, linelength, sep)
    pos = pos[:-1]  # Remove last point to avoid enclosed point
    
    if len(pos) == 0:
        return pd.DataFrame(columns=['x', 'y', 'x0', 'y0', 'x1', 'y1', 'theta'])
    
    # Find which segment each position falls in
    whichseg = np.array([np.sum(dtotal <= x) for x in pos])
    
    # Ensure whichseg doesn't exceed array bounds
    max_seg = len(xy) - 2  # Maximum valid segment index
    whichseg = np.clip(whichseg, 0, max_seg)
    
    # Create dataframe with position information
    pos_df = pd.DataFrame({
        'pos': pos,
        'whichseg': whichseg,
        'x0': xy[whichseg, 0],
        'y0': xy[whichseg, 1],
        'dseg': dseg[whichseg + 1],
        'dtotal': dtotal[whichseg],
        'x1': xy[whichseg + 1, 0],
        'y1': xy[whichseg + 1, 1]
    })
    
    # Calculate exact positions
    pos_df['further'] = pos_df['pos'] - pos_df['dtotal']
    pos_df['f'] = pos_df['further'] / pos_df['dseg']
    pos_df['x'] = pos_df['x0'] + pos_df['f'] * (pos_df['x1'] - pos_df['x0'])
    pos_df['y'] = pos_df['y0'] + pos_df['f'] * (pos_df['y1'] - pos_df['y0'])
    
    # Calculate angle
    pos_df['theta'] = np.arctan2(pos_df['y0'] - pos_df['y1'], pos_df['x0'] - pos_df['x1'])
    
    return pos_df[['x', 'y', 'x0', 'y0', 'x1', 'y1', 'theta']]


def transect(tpts, tlen, npts=1):
    """
    Creates points perpendicular to a line with set distance.
    
    Parameters:
    -----------
    tpts : DataFrame
        DataFrame from evenspace with columns: x, y, theta
    tlen : float
        Length of transect steps
    npts : int, optional
        Number of points on one side in addition to center (default=1)
    
    Returns:
    --------
    DataFrame with columns: transectID, point_position, x, y
    """
    if len(tpts) == 0:
        return pd.DataFrame(columns=['transectID', 'point_position', 'x', 'y'])
    
    tpts = tpts.copy()
    tpts['thetaT'] = tpts['theta'] + np.pi / 2
    
    dx = tlen * np.cos(tpts['thetaT'])
    dy = tlen * np.sin(tpts['thetaT'])
    
    x = tpts['x'].values
    y = tpts['y'].values
    
    # Create inner points (negative positions)
    x_inner = np.column_stack([x + i * dx for i in range(npts, 0, -1)])
    y_inner = np.column_stack([y + i * dy for i in range(npts, 0, -1)])
    inner_names = [f"-{i}" for i in range(npts, 0, -1)]
    
    # Create outer points (positive positions, including center at 0)
    x_outer = np.column_stack([x - i * dx for i in range(0, npts + 1)])
    y_outer = np.column_stack([y - i * dy for i in range(0, npts + 1)])
    outer_names = [f"+{i}" for i in range(0, npts + 1)]
    
    # Combine inner and outer
    xx = np.column_stack([x_inner, x_outer])
    yy = np.column_stack([y_inner, y_outer])
    all_names = inner_names + outer_names
    
    # Create long format dataframe
    n_transects = len(tpts)
    n_points_per_transect = 2 * npts + 1
    
    result = []
    for i in range(n_transects):
        for j, name in enumerate(all_names):
            result.append({
                'transectID': i + 1,
                'point_position': float(name),
                'x': xx[i, j],
                'y': yy[i, j]
            })
    
    xy = pd.DataFrame(result)
    xy = xy.sort_values(['transectID', 'point_position']).reset_index(drop=True)
    
    return xy


def extract_coords(geom):
    """Extract coordinates from geometry (Polygon or MultiPolygon)."""
    if geom.geom_type == 'Polygon':
        return np.array(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        # Use the largest polygon for MultiPolygon
        largest = max(geom.geoms, key=lambda p: p.area)
        return np.array(largest.exterior.coords)
    else:
        return None


def create_transects(park_row, sample_dist, transect_unit, transect_pts):
    """Process a single park to generate transect points."""
    park_data = park_row[1]  # Get the Series from (index, Series) tuple
    geom = park_data.geometry
    
    # Extract coordinates
    coords = extract_coords(geom)
    if coords is None or len(coords) < 3:
        return None
    
    # Create evenly spaced boundary points
    div_pts = evenspace(coords, sample_dist)
    if len(div_pts) == 0:
        return None
    
    # Create transect points
    transect_pts_df = transect(div_pts, transect_unit, npts=transect_pts)
    if len(transect_pts_df) == 0:
        return None
    
    # Add park attributes
    base_props = {col: park_data[col] for col in park_data.index if col not in ['geometry', 'geometry_t']}
    transect_pts_df = transect_pts_df.assign(**base_props)
    
    return transect_pts_df
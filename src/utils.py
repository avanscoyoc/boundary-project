import ee
import os
import gc
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


# =====================================================================
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


def check_task_status(submitted_tasks):
    """
    Check status of submitted Earth Engine tasks and remove completed ones.
    
    Parameters:
    -----------
    submitted_tasks : list
        List of tuples (task_object, year)
    
    Returns:
    --------
    tuple: (active_tasks, num_active)
        Updated list of active tasks and count of active tasks
    """
    active_tasks = []
    for task_obj, year in submitted_tasks:
        task_status = task_obj.status()
        if task_status['state'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print(f"Task {year} {task_status['state']}")
        else:
            active_tasks.append((task_obj, year))
    return active_tasks, len(active_tasks)


# =====================================================================
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


# =====================================================================
# Transect Creation / Filtering Functions
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


def remove_bad_transects(transect_df, park_geom, buffer_geom, crs):
    """
    Filter out bad transects where inner points are either:
    1. Inside the inner buffer (indicates bad angle)
    2. Outside the PA polygon (crossed to opposite side)
    
    Parameters:
    -----------
    transect_df : DataFrame
        Transect points with columns: transectID, point_position, x, y, etc.
    park_geom : shapely geometry
        Protected area polygon geometry
    buffer_geom : shapely geometry
        Inner buffer geometry
    crs : CRS
        Coordinate reference system
    
    Returns:
    --------
    tuple: (filtered_df, n_bad_inside_buffer, n_bad_outside_pa)
    """
    # Get inner points only (negative positions)
    inner_pts = transect_df[transect_df['point_position'] < 0].copy()
    
    if len(inner_pts) == 0:
        return transect_df, 0, 0
    
    # Create minimal geodataframe for spatial check
    inner_gdf = gpd.GeoDataFrame(
        inner_pts[['WDPA_PID', 'transectID']],
        geometry=gpd.points_from_xy(inner_pts['x'], inner_pts['y']),
        crs=crs
    )
    
    # Mark inner points inside the inner buffer as bad (bad angle)
    bad_inside_buffer = inner_gdf[inner_gdf.geometry.within(buffer_geom)]['transectID'].unique()
    
    # Mark inner points outside the PA polygon as bad (crossed to opposite side)
    bad_outside_pa = inner_gdf[~inner_gdf.geometry.within(park_geom)]['transectID'].unique()
    
    # Combine both sets of bad transects
    bad_transects = np.unique(np.concatenate([bad_inside_buffer, bad_outside_pa]))
    
    # Filter out bad transects
    filtered_df = transect_df[~transect_df['transectID'].isin(bad_transects)]
    
    return filtered_df, len(bad_inside_buffer), len(bad_outside_pa)


def remove_pa_transects_in_chunks(wdpa_gdf, wdpa_buffer_dict, sample_dist, transect_unit, 
                          transect_pts, output_dir, chunk_size=500):
    """
    Process protected areas to generate transect points, filter bad transects,
    and write results to chunk files.
    
    Parameters:
    -----------
    wdpa_gdf : GeoDataFrame
        Protected areas geodataframe
    wdpa_buffer_dict : dict
        Dictionary mapping WDPA_PID to inner buffer geometries
    sample_dist : float
        Transect spacing in meters
    transect_unit : float
        Distance between samples along a transect in meters
    transect_pts : int
        Number of points on each side of boundary point
    output_dir : str
        Output directory for chunk files
    chunk_size : int, optional
        Number of PAs to process before writing a chunk (default=500)
    
    Returns:
    --------
    dict with processing statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    crs = wdpa_gdf.crs
    
    print(f"Processing {len(wdpa_gdf)} protected areas with streaming filter...")
    chunk_files = []
    chunk_num = 0
    chunk_data = []
    total_points = 0
    total_transects = 0
    pas_processed = 0
    
    # Diagnostic counters
    empty_buffer = 0
    all_filtered = 0
    bad_inside_buffer = 0
    bad_outside_pa = 0
    
    for idx, (_, park_row) in enumerate(wdpa_gdf.iterrows()):
        # Generate transects for single PA
        transect_df = create_transects((idx, park_row), sample_dist, transect_unit, transect_pts)
        
        if transect_df is None:
            continue
        
        # Check buffer exists and is not empty
        pid = park_row['WDPA_PID']
        if pid not in wdpa_buffer_dict or wdpa_buffer_dict[pid].is_empty:
            empty_buffer += 1
            continue
        
        # Filter bad transects
        buffer_geom = wdpa_buffer_dict[pid]
        transect_df, n_bad_inside, n_bad_outside = remove_bad_transects(
            transect_df, park_row.geometry, buffer_geom, crs
        )
        bad_inside_buffer += n_bad_inside
        bad_outside_pa += n_bad_outside
        
        # Add to chunk data if any transects remain
        if len(transect_df) > 0:
            total_points += len(transect_df)
            total_transects += transect_df['transectID'].nunique()
            pas_processed += 1
            chunk_data.append(transect_df)
        else:
            all_filtered += 1
        
        # Write chunk every N PAs
        if len(chunk_data) >= chunk_size:
            chunk_file = f"{output_dir}/chunk_{chunk_num:03d}.csv"
            pd.concat(chunk_data, ignore_index=True).to_csv(chunk_file, index=False)
            chunk_files.append(chunk_file)
            chunk_data = []
            chunk_num += 1
            gc.collect()
            print(f"  Processed {idx + 1}/{len(wdpa_gdf)} PAs, wrote chunk {chunk_num} | Total points: {total_points:,}")
    
    # Write final chunk
    if chunk_data:
        chunk_file = f"{output_dir}/chunk_{chunk_num:03d}.csv"
        pd.concat(chunk_data, ignore_index=True).to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
        print(f"  Wrote final chunk {chunk_num + 1}")
    
    # Return statistics
    return {
        'total_pas': idx + 1,
        'empty_buffer': empty_buffer,
        'all_filtered': all_filtered,
        'bad_inside_buffer': bad_inside_buffer,
        'bad_outside_pa': bad_outside_pa,
        'pas_processed': pas_processed,
        'total_points': total_points,
        'total_transects': total_transects,
        'chunk_files': chunk_files
    }


def transform_chunks_crs(chunk_pattern, source_crs, target_crs):
    """
    Transform CRS of all chunk files matching the pattern.
    
    Parameters:
    -----------
    chunk_pattern : str
        Glob pattern for chunk files (e.g., '../data/transect_chunks/chunk_*.csv')
    source_crs : str
        Source CRS (e.g., 'ESRI:54009')
    target_crs : str
        Target CRS (e.g., 'EPSG:4326')
    """

    chunk_files = sorted(glob.glob(chunk_pattern))
    print(f"Transforming {len(chunk_files)} chunks from {source_crs} to {target_crs}...")
    
    for i, chunk_file in enumerate(chunk_files, start=1):
        chunk = pd.read_csv(chunk_file, low_memory=False)
        chunk_gdf = gpd.GeoDataFrame(
            chunk, 
            geometry=gpd.points_from_xy(chunk['x'], chunk['y']), 
            crs=source_crs
        )
        chunk_gdf = chunk_gdf.to_crs(target_crs)
        chunk['x'] = chunk_gdf.geometry.x
        chunk['y'] = chunk_gdf.geometry.y
        chunk.to_csv(chunk_file, index=False)
        if i % 3 == 0 or i == len(chunk_files):
            print(f"  Transformed {i}/{len(chunk_files)} chunks")
    
    print("CRS transformation complete!")


def combine_chunks_to_files(chunk_pattern, transect_output, attributes_output, 
                            transect_cols=['WDPA_PID', 'transectID', 'point_position', 'x', 'y']):
    """
    Combine chunk files into two outputs: transects (essential columns) and attributes (metadata).
    
    Parameters:
    -----------
    chunk_pattern : str
        Glob pattern for chunk files (e.g., '../data/transect_chunks/chunk_*.csv')
    transect_output : str
        Output path for transects file with essential columns
    attributes_output : str
        Output path for attributes file with metadata keyed by WDPA_PID
    transect_cols : list, optional
        List of essential columns for transects (default: ['WDPA_PID', 'transectID', 'point_position', 'x', 'y'])
    """
    
    chunk_files = sorted(glob.glob(chunk_pattern))
    print(f"Found {len(chunk_files)} chunk files")
    print(f"Combining into:\n  - {transect_output}\n  - {attributes_output}")
    
    # Get all columns from first chunk to determine attribute columns
    first_chunk = pd.read_csv(chunk_files[0], low_memory=False)
    all_cols = first_chunk.columns.tolist()
    attr_cols = [col for col in all_cols if col not in ['transectID', 'point_position', 'x', 'y']]
    
    print(f"\nTransect columns: {transect_cols}")
    print(f"Attribute columns: {attr_cols}")
    
    # Write transects file
    print("\nWriting transects file...")
    first_chunk[transect_cols].to_csv(transect_output, index=False, mode='w')
    print(f"  Wrote chunk 1/{len(chunk_files)}")
    
    for i, chunk_file in enumerate(chunk_files[1:], start=2):
        chunk = pd.read_csv(chunk_file, low_memory=False)
        chunk[transect_cols].to_csv(transect_output, index=False, mode='a', header=False)
        print(f"  Wrote chunk {i}/{len(chunk_files)}")
        del chunk
    
    print(f"Transects saved: {os.path.getsize(transect_output) / 1024**2:.1f} MB")
    
    # Extract unique attributes by WDPA_PID
    print("\nExtracting unique attributes by WDPA_PID...")
    all_attributes = []
    for i, chunk_file in enumerate(chunk_files, start=1):
        chunk = pd.read_csv(chunk_file, low_memory=False)
        # Get unique rows per WDPA_PID with all attribute columns
        unique_attrs = chunk[attr_cols].drop_duplicates(subset=['WDPA_PID'])
        all_attributes.append(unique_attrs)
        if i % 3 == 0 or i == len(chunk_files):
            print(f"  Processed {i}/{len(chunk_files)} chunks")
    
    # Combine and deduplicate
    attributes_df = pd.concat(all_attributes, ignore_index=True).drop_duplicates(subset=['WDPA_PID'])
    attributes_df.to_csv(attributes_output, index=False)
    
    print(f"\nAttributes saved: {os.path.getsize(attributes_output) / 1024**2:.1f} MB")
    print(f"Unique WDPA_PIDs: {len(attributes_df)}")
    print("\nCombining complete!")
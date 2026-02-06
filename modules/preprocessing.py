"""
Geometry processing and transect generation functions.

This module contains functions for:
- Cleaning and filtering polygon geometries
- Identifying and removing overlapping protected areas  
- Generating perpendicular transects at protected area boundaries
- Filtering problematic transects
"""

import os
import gc
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def fill_holes(gdf, max_hole_area=2250000):  # 1500m * 1500m = 2250000 sq meters
    """
    Fill small holes in polygon geometries while preserving large holes.
    
    Processes both Polygon and MultiPolygon geometries, removing interior holes
    (donuts) that are smaller than the specified threshold area. This is useful
    for cleaning protected area boundaries that may have small gaps.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing polygon geometries to process.
    max_hole_area : float, optional
        Maximum area (in square meters) for holes to be filled. Holes larger than
        this will be preserved. Default is 2,250,000 m² (1500m × 1500m).
    
    Returns
    -------
    GeoDataFrame
        Copy of input GeoDataFrame with filled geometries.
    
    Notes
    -----
    Handles both Polygon and MultiPolygon geometry types. For MultiPolygons,
    each component polygon is processed independently.
    """
    
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
    """
    Identify groups of geometries with significant spatial overlap.
    
    Finds sets of protected areas where the intersection area exceeds the specified
    percentage threshold relative to either geometry. Uses a greedy algorithm where
    each group is built starting from an unprocessed geometry.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing geometries to analyze for overlaps.
    overlap_threshold : float, optional
        Minimum overlap percentage (0-100) to consider geometries as overlapping.
        Default is 90%. Overlap is calculated as the maximum of:
        (intersection_area / geom1_area) * 100 or (intersection_area / geom2_area) * 100.
    
    Returns
    -------
    list of list
        List of overlap groups, where each group is a list of integer indices
        from the input GeoDataFrame. Geometries that don't overlap significantly
        with any others are returned as single-element groups.
    
    Notes
    -----
    Uses spatial indexing implicitly through iterrows. For very large datasets,
    consider using spatial index (sindex) for better performance.
    """

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
    """
    Select the oldest protected area from a group based on establishment year.
    
    Returns the row with the minimum non-zero STATUS_YR value. If all STATUS_YR
    values are 0 (unknown), returns the first row.
    
    Parameters
    ----------
    group_df : DataFrame
        DataFrame subset containing a group of overlapping protected areas.
        Must have a 'STATUS_YR' column.
    
    Returns
    -------
    Series
        Row from the input DataFrame representing the oldest protected area.
    
    Notes
    -----
    Used for deduplication when multiple protected areas overlap significantly.
    Prioritizes older designations as they typically represent the original boundary.
    """

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
    Generate evenly spaced points along a polyline at specified intervals.
    
    Creates sample points at regular distance intervals along a line defined by
    vertices. Returns point coordinates along with segment information and angle.
    The last point is excluded to avoid duplication at the start/end of closed loops.
    
    Parameters
    ----------
    xy : array-like of shape (N, 2)
        Array of coordinates defining the polyline vertices, where each row is [x, y].
    sep : float
        Separation distance between consecutive points along the line (in map units).
    start : float, optional
        Starting distance along the line for the first point. Default is 0.
    
    Returns
    -------
    DataFrame
        DataFrame with columns:
        - x, y : Coordinates of the evenly spaced points
        - x0, y0 : Start coordinates of the line segment containing each point
        - x1, y1 : End coordinates of the line segment containing each point
        - theta : Angle (in radians) of the line segment at each point
    
    Notes
    -----
    Returns empty DataFrame if the line length is too short to place any points.
    The theta angle is calculated from segment direction and used for perpendicular
    transect generation.
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
    Generate perpendicular transect points from boundary sample points.
    
    Creates transects perpendicular to a polyline, with multiple points at regular
    intervals extending both inward and outward from the boundary. Each transect
    includes a center point (position 0) and symmetric points on both sides.
    
    Parameters
    ----------
    tpts : DataFrame
        DataFrame from evenspace() containing boundary points with columns:
        x, y (point coordinates) and theta (line angle in radians).
    tlen : float
        Distance between consecutive points along each transect (in map units).
    npts : int, optional
        Number of points on each side of the boundary point (not including center).
        Default is 1. Total points per transect = 2*npts + 1.
    
    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:
        - transectID : Integer identifier for each transect (1-indexed)
        - pointID : Position along transect (negative=inward, 0=boundary, positive=outward)
        - x, y : Coordinates of each transect point
    
    Notes
    -----
    Point positions are labeled as floats: negative values (e.g., -2.0, -1.0) for
    inward points, 0.0 for boundary, and positive values (e.g., +1.0, +2.0) for
    outward points.
    """
    if len(tpts) == 0:
        return pd.DataFrame(columns=['transectID', 'pointID', 'x', 'y'])
    
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
                'pointID': float(name),
                'x': xx[i, j],
                'y': yy[i, j]
            })
    
    xy = pd.DataFrame(result)
    xy = xy.sort_values(['transectID', 'pointID']).reset_index(drop=True)
    
    return xy


def extract_coords(geom):
    """
    Extract exterior boundary coordinates from a polygon geometry.
    
    For MultiPolygon geometries, extracts coordinates from the largest component
    polygon by area.
    
    Parameters
    ----------
    geom : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Input geometry to extract coordinates from.
    
    Returns
    -------
    numpy.ndarray or None
        Array of shape (N, 2) containing [x, y] coordinates of the exterior boundary.
        Returns None if geometry type is not Polygon or MultiPolygon.
    """
    if geom.geom_type == 'Polygon':
        return np.array(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        # Use the largest polygon for MultiPolygon
        largest = max(geom.geoms, key=lambda p: p.area)
        return np.array(largest.exterior.coords)
    else:
        return None


def create_transects(park_row, sample_dist, transect_unit, transect_pts):
    """
    Generate complete transect point dataset for a single protected area.
    
    Combines boundary sampling, perpendicular transect generation, and attribute
    assignment into a single processing function. Returns a DataFrame with all
    transect points and their associated protected area metadata.
    
    Parameters
    ----------
    park_row : tuple of (int, Series)
        Tuple containing (index, row_data) from iterrows(), where row_data
        contains geometry and all protected area attributes.
    sample_dist : float
        Distance between boundary sample points (in meters).
    transect_unit : float
        Distance between points along each transect (in meters).
    transect_pts : int
        Number of points on each side of the boundary (not including center).
    
    Returns
    -------
    DataFrame or None
        DataFrame containing all transect points with columns for coordinates
        (x, y), transect identifiers (transectID, pointID), and all
        protected area attributes. Returns None if geometry is invalid or
        insufficient boundary points are generated.
    """
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
    Filter out problematic transects based on inner point spatial relationships.
    
    Removes transects where the inward-facing points (negative positions) exhibit
    geometric issues:
    1. Points fall inside the inner buffer zone (indicates bad transect angle)
    2. Points fall outside the protected area polygon (transect crossed to opposite side)
    
    Parameters
    ----------
    transect_df : DataFrame
        Transect points containing transectID, pointID, x, y columns,
        plus additional protected area attributes.
    park_geom : shapely.geometry.Polygon or MultiPolygon
        Protected area boundary geometry.
    buffer_geom : shapely.geometry.Polygon or MultiPolygon
        Inner buffer geometry (typically 5500m inward from boundary).
    crs : pyproj.CRS or str
        Coordinate reference system for spatial operations.
    
    Returns
    -------
    tuple of (DataFrame, int, int)
        - filtered_df : DataFrame with bad transects removed
        - n_bad_inside_buffer : Number of transects removed due to points in buffer
        - n_bad_outside_pa : Number of transects removed due to points outside PA
    
    Notes
    -----
    Only examines inner points (pointID < 0). If a transect is flagged by
    either criterion, all points belonging to that transect are removed.
    """
    # Get inner points only (negative positions)
    inner_pts = transect_df[transect_df['pointID'] < 0].copy()
    
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
                          transect_pts, output_dir, chunk_size=500, output_crs='EPSG:4326'):
    """
    Generate and filter transects for protected areas with streaming chunk output.
    
    Processes protected areas in batches, generating transects, filtering problematic
    ones, and writing results to sequential shapefile chunks to avoid memory issues.
    Writes directly in EPSG:4326 for GEE upload efficiency.
    
    Parameters
    ----------
    wdpa_gdf : GeoDataFrame
        Protected areas geodataframe with geometry and WDPA attributes.
    wdpa_buffer_dict : dict
        Dictionary mapping WDPA_PID (str) to inner buffer geometries (shapely.geometry).
    sample_dist : float
        Distance between boundary sample points in meters (e.g., 500).
    transect_unit : float
        Distance between consecutive points along each transect in meters (e.g., 2500).
    transect_pts : int
        Number of points on each side of the boundary point (e.g., 2 for 5 total points).
    output_dir : str
        Directory path where chunk shapefiles will be written.
    chunk_size : int, optional
        Number of protected areas to process before writing a chunk file.
        Default is 500 (~1 million points per chunk).
    output_crs : str, optional
        Output coordinate reference system. Default 'EPSG:4326' for GEE.
    
    Returns
    -------
    dict
        Statistics dictionary with keys:
        - total_pas : Total number of PAs processed
        - empty_buffer : Number of PAs with missing or empty buffers
        - all_filtered : Number of PAs where all transects were filtered out
        - bad_inside_buffer : Total transects filtered (points in buffer)
        - bad_outside_pa : Total transects filtered (points outside PA)
        - pas_processed : Number of PAs that yielded valid transects
        - total_points : Total number of transect points generated
        - total_transects : Total number of valid transects
        - chunk_files : List of created shapefile paths
    
    Notes
    -----
    Writes shapefiles directly in target CRS to eliminate separate transformation step.
    Uses streaming approach with periodic garbage collection. Only keeps WDPA_PID,
    transectID, pointID columns for GEE upload efficiency.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    source_crs = wdpa_gdf.crs
    
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
            transect_df, park_row.geometry, buffer_geom, source_crs
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
            chunk_file = f"{output_dir}/chunk_{chunk_num:03d}.shp"
            chunk_combined = pd.concat(chunk_data, ignore_index=True)
            
            # Create GeoDataFrame in source CRS, transform to output CRS
            chunk_gdf = gpd.GeoDataFrame(
                chunk_combined,
                geometry=gpd.points_from_xy(chunk_combined['x'], chunk_combined['y']),
                crs=source_crs
            ).to_crs(output_crs)
            
            # Keep only essential columns for GEE (drop x, y - stored in geometry)
            chunk_gdf = chunk_gdf[['WDPA_PID', 'transectID', 'pointID', 'geometry']]

            chunk_gdf.to_file(chunk_file)
            
            chunk_files.append(chunk_file)
            chunk_data = []
            chunk_num += 1
            gc.collect()
            print(f"  Processed {idx + 1}/{len(wdpa_gdf)} PAs, wrote chunk {chunk_num} | Total points: {total_points:,}")
    
    # Write final chunk
    if chunk_data:
        chunk_file = f"{output_dir}/chunk_{chunk_num:03d}.shp"
        chunk_combined = pd.concat(chunk_data, ignore_index=True)
        
        chunk_gdf = gpd.GeoDataFrame(
            chunk_combined,
            geometry=gpd.points_from_xy(chunk_combined['x'], chunk_combined['y']),
            crs=source_crs
        ).to_crs(output_crs)
        
        chunk_gdf = chunk_gdf[['WDPA_PID', 'transectID', 'pointID', 'geometry']]

        chunk_gdf.to_file(chunk_file)
        
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


def combine_chunks_to_files(chunk_pattern, transect_output, attributes_output, 
                            transect_cols=['WDPA_PID', 'transectID', 'pointID', 'x', 'y']):
    """
    Combine chunk shapefiles into separate transect CSV and attribute CSV files.
    
    Reads shapefile chunks, extracts x,y from geometry, and creates two output
    files: a minimal transect CSV for analysis and an attributes CSV with
    protected area metadata that can be rejoined via WDPA_PID.
    
    Parameters
    ----------
    chunk_pattern : str
        Glob pattern to match chunk shapefiles (e.g., 'data/intermediate/transect_chunks/chunk_*.shp').
    transect_output : str
        Output file path for the transects dataset (typically transects_final.csv).
    attributes_output : str
        Output file path for the attributes dataset (typically attributes_final.csv).
    transect_cols : list of str, optional
        Columns to include in the transects output file. Default is
        ['WDPA_PID', 'transectID', 'pointID', 'x', 'y'].
    
    Returns
    -------
    None
        Writes two CSV files to disk.
    
    Notes
    -----
    - Reads from shapefiles and extracts x,y coordinates from geometry
    - Transect file: Contains millions of points with minimal columns for efficiency
    - Attributes file: Contains one row per protected area with all metadata
    - Uses streaming writes for transect file to minimize memory usage
    - Prints file sizes and unique PA counts for verification
    
    Examples
    --------
    >>> combine_chunks_to_files(
    ...     'data/intermediate/transect_chunks/chunk_*.shp',
    ...     'data/processed/transects_final.csv',
    ...     'data/processed/attributes_final.csv'
    ... )
    """
    
    chunk_files = sorted(glob.glob(chunk_pattern))
    print(f"Found {len(chunk_files)} chunk files")
    print(f"Combining into:\n  - {transect_output}\n  - {attributes_output}")
    
    # Get all columns from first chunk to determine attribute columns
    first_gdf = gpd.read_file(chunk_files[0])
    first_chunk = pd.DataFrame(first_gdf.drop(columns='geometry'))
    first_chunk['x'] = first_gdf.geometry.x
    first_chunk['y'] = first_gdf.geometry.y
    
    all_cols = first_chunk.columns.tolist()
    attr_cols = [col for col in all_cols if col not in ['transectID', 'pointID', 'x', 'y']]
    
    print(f"\nTransect columns: {transect_cols}")
    print(f"Attribute columns: {attr_cols}")
    
    # Write transects file
    print("\nWriting transects file...")
    first_chunk[transect_cols].to_csv(transect_output, index=False, mode='w')
    print(f"  Wrote chunk 1/{len(chunk_files)}")
    
    for i, chunk_file in enumerate(chunk_files[1:], start=2):
        chunk_gdf = gpd.read_file(chunk_file)
        chunk = pd.DataFrame(chunk_gdf.drop(columns='geometry'))
        chunk['x'] = chunk_gdf.geometry.x
        chunk['y'] = chunk_gdf.geometry.y

        chunk[transect_cols].to_csv(transect_output, index=False, mode='a', header=False)
        print(f"  Wrote chunk {i}/{len(chunk_files)}")
        del chunk, chunk_gdf
    
    print(f"Transects saved: {os.path.getsize(transect_output) / 1024**2:.1f} MB")
    
    # Extract unique attributes by WDPA_PID
    print("\nExtracting unique attributes by WDPA_PID...")
    all_attributes = []
    for i, chunk_file in enumerate(chunk_files, start=1):
        chunk_gdf = gpd.read_file(chunk_file)
        chunk = pd.DataFrame(chunk_gdf.drop(columns='geometry'))
        chunk['x'] = chunk_gdf.geometry.x
        chunk['y'] = chunk_gdf.geometry.y

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

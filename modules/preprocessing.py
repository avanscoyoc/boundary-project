"""
Geometry processing and transect generation functions (script 02).

Part 1, choosing PAs:
- fill_holes: fill holes smaller than a threshold
- resolve_overlaps: drop duplicate and nested PAs (spatial index)
- merge_same_name: dissolve touching/overlapping PAs with the same name and country
- add_shape_metrics: geodesic area, perimeter, and perimeter/sqrt(area)

Part 2, transects:
- evenspace / transect / extract_coords: boundary points and perpendicular transects
- make_pa_transects: one PA, in its own azimuthal equidistant projection, with the
  whole-line check
- flag_other_pa: flag transects whose outer points fall inside another PA
- generate_transects: loop over PAs, write upload zips and transect_points.parquet
"""

import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Geod, Proj, Transformer
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


# =====================================================================
# Part 1: choosing PAs
# =====================================================================

def _fill_small_holes(geom, max_hole_area):
    """Return ``geom`` with interior rings smaller than ``max_hole_area`` removed."""
    if geom.geom_type == 'Polygon':
        if not geom.interiors:
            return geom
        large_holes = [ring for ring in geom.interiors if Polygon(ring).area > max_hole_area]
        return Polygon(geom.exterior, large_holes)
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([_fill_small_holes(p, max_hole_area) for p in geom.geoms])
    return geom


def fill_holes(gdf, max_hole_area=2250000):  # 1500m * 1500m = 2250000 sq meters
    """
    Fill small holes in polygon geometries while preserving large holes.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons in a projected CRS with units of meters.
    max_hole_area : float, optional
        Holes with area (m²) at or below this are filled. Default 2,250,000 m²
        (1500 m × 1500 m, a 3 × 3 MODIS pixel window).

    Returns
    -------
    tuple of (GeoDataFrame, int)
        Copy of ``gdf`` with filled geometries, and the number of PAs changed.
    """
    filled = [_fill_small_holes(g, max_hole_area) for g in gdf.geometry]
    n_changed = int(sum(not a.equals(b) for a, b in zip(gdf.geometry, filled)))
    result = gdf.copy()
    result['geometry'] = filled
    return result, n_changed


def get_min_year_from_group(group_df):
    """
    Select the oldest protected area from a group based on establishment year.

    Returns the row with the minimum non-zero STATUS_YR value. If all STATUS_YR
    values are 0 (unknown), returns the first row.

    Parameters
    ----------
    group_df : DataFrame
        Group of protected areas with a 'STATUS_YR' column.

    Returns
    -------
    Series
        Row of the oldest protected area (its ``.name`` is the row label).
    """
    non_zero = group_df[group_df['STATUS_YR'] != 0]
    if len(non_zero) > 0:
        return non_zero.loc[non_zero['STATUS_YR'].idxmin()]
    return group_df.iloc[0]


def _components(n, pairs):
    """Connected components (union-find) of ``n`` items linked by ``pairs``."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def resolve_overlaps(gdf, overlap_threshold=90):
    """
    Remove duplicate and nested protected areas.

    Candidate pairs come from a spatial index. For each intersecting pair, the
    intersection area is compared with each polygon's area:

    - **Duplicates**: overlap > threshold of BOTH polygons. Duplicates are grouped
      transitively (A~B and B~C puts A, B, C in one group) and the oldest
      non-zero STATUS_YR is kept (get_min_year_from_group).
    - **Nested**: overlap > threshold of the smaller polygon only. The smaller
      polygon is removed and the larger one kept. If a polygon sits inside
      several larger ones, it is assigned to the largest.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons in an equal-area CRS, with 'WDPA_PID' (str) and 'STATUS_YR'.
    overlap_threshold : float, optional
        Percent overlap threshold. Default 90.

    Returns
    -------
    tuple of (GeoDataFrame, DataFrame, dict)
        - kept PAs (index reset)
        - removed rows: original columns plus 'reason' ('duplicate' or 'nested')
          and 'replaced_by' (PID of the PA kept in its place)
        - counts: n_in, n_invalid_fixed, removed_duplicate, removed_nested, n_out
    """
    gdf = gdf.reset_index(drop=True)
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    invalid = ~shapely.is_valid(geoms)
    if invalid.any():
        geoms[invalid] = shapely.make_valid(geoms[invalid])
        gdf = gdf.set_geometry(gpd.GeoSeries(geoms, crs=gdf.crs, index=gdf.index))
    areas = shapely.area(geoms)
    pids = gdf['WDPA_PID'].to_numpy()

    tree = shapely.STRtree(geoms)
    i, j = tree.query(geoms, predicate='intersects')
    keep_pair = i < j
    i, j = i[keep_pair], j[keep_pair]
    inter = shapely.area(shapely.intersection(geoms[i], geoms[j]))
    pct_i = inter / areas[i] * 100
    pct_j = inter / areas[j] * 100

    dup = (pct_i > overlap_threshold) & (pct_j > overlap_threshold)
    nested = ~dup & ((pct_i > overlap_threshold) | (pct_j > overlap_threshold))

    reason = np.full(len(gdf), None, dtype=object)
    replaced_by = np.full(len(gdf), None, dtype=object)

    # Duplicates: transitive groups, keep the oldest
    roots = _components(len(gdf), zip(i[dup], j[dup]))
    for r in np.unique(roots[np.isin(np.arange(len(gdf)), np.r_[i[dup], j[dup]])]):
        members = np.flatnonzero(roots == r)
        keep = get_min_year_from_group(gdf.iloc[members]).name
        for m in members:
            if m != keep:
                reason[m], replaced_by[m] = 'duplicate', pids[keep]

    # Nested: remove the smaller; assign to the largest container
    small = np.where(pct_i > overlap_threshold, i, j)[nested]
    large = np.where(pct_i > overlap_threshold, j, i)[nested]
    order = np.argsort(-areas[large])
    for s, l in zip(small[order], large[order]):
        if reason[s] is None:
            reason[s], replaced_by[s] = 'nested', pids[l]

    is_removed = pd.notna(reason)
    removed = gdf[is_removed].copy()
    removed['reason'] = reason[is_removed]
    removed['replaced_by'] = replaced_by[is_removed]
    kept = gdf[~is_removed].reset_index(drop=True)

    counts = {
        'n_in': len(gdf),
        'n_invalid_fixed': int(invalid.sum()),
        'removed_duplicate': int((reason == 'duplicate').sum()),
        'removed_nested': int((reason == 'nested').sum()),
        'n_out': len(kept),
    }
    return kept, removed, counts


def merge_same_name(gdf, max_hole_area=2250000):
    """
    Dissolve protected areas that share a name and country and touch or overlap.

    PAs are grouped by ORIG_NAME + ISO3. Within a group, members that touch or
    overlap (directly or through each other) are dissolved into one polygon
    that keeps the attributes of the oldest non-zero STATUS_YR member. The
    'MERGED_PID' column lists the PIDs merged into it (';'-separated). Members
    that don't touch stay separate PAs. If a union isn't a single Polygon, its
    members are left separate. Small holes in each union are filled again.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons in a projected CRS (meters), with 'WDPA_PID', 'ORIG_NAME',
        'ISO3', 'STATUS_YR'.
    max_hole_area : float, optional
        Hole-filling threshold (m²) applied to each union.

    Returns
    -------
    tuple of (GeoDataFrame, DataFrame, dict)
        - PAs after merging (index reset), with 'MERGED_PID'
        - removed (merged-away) rows with 'reason' = 'same_name_merged' and
          'replaced_by'
        - counts: n_in, groups_merged, pas_merged_away, groups_not_single_polygon, n_out
    """
    gdf = gdf.reset_index(drop=True).copy()
    gdf['MERGED_PID'] = ''
    key = gdf['ORIG_NAME'].fillna('<NA>').astype(str) + '||' + gdf['ISO3'].fillna('<NA>').astype(str)
    pids = gdf['WDPA_PID'].to_numpy()

    drop, reasons, replaced = [], [], []
    groups_merged = not_single = 0
    for _, pos in gdf.groupby(key, sort=False).indices.items():
        if len(pos) < 2:
            continue
        geoms = np.asarray(gdf.geometry.values[pos], dtype=object)
        a, b = shapely.STRtree(geoms).query(geoms, predicate='intersects')
        roots = _components(len(pos), zip(a[a < b], b[a < b]))
        for r in np.unique(roots):
            members = pos[roots == r]
            if len(members) < 2:
                continue
            union = _fill_small_holes(unary_union(list(gdf.geometry.values[members])), max_hole_area)
            if union.geom_type != 'Polygon':
                not_single += 1
                continue
            keep = get_min_year_from_group(gdf.iloc[members]).name
            others = [m for m in members if m != keep]
            gdf.at[keep, 'geometry'] = union
            gdf.at[keep, 'MERGED_PID'] = ';'.join(pids[others])
            drop += others
            replaced += [pids[keep]] * len(others)
            groups_merged += 1

    removed = gdf.loc[drop].copy()
    removed['reason'] = 'same_name_merged'
    removed['replaced_by'] = replaced
    kept = gdf.drop(index=drop).reset_index(drop=True)

    counts = {
        'n_in': len(gdf),
        'groups_merged': groups_merged,
        'pas_merged_away': len(drop),
        'groups_not_single_polygon': not_single,
        'n_out': len(kept),
    }
    return kept, removed, counts


def add_shape_metrics(gdf):
    """
    Add geodesic area, perimeter, and shape ratio columns.

    Computed on the WGS84 ellipsoid (pyproj.Geod), so no projection distortion.
    Perimeter includes interior rings (holes).

    - AREA_KM2: area in km²
    - PERIM_KM: perimeter in km
    - SHAPE_RATIO: perimeter / sqrt(area), dimensionless and size-neutral
      (a circle is 3.54, a square 4.0; higher = more convoluted). Reported
      for information; no PAs are removed on shape.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons in any CRS.

    Returns
    -------
    GeoDataFrame
        Copy with the three columns added.
    """
    geod = Geod(ellps='WGS84')
    lonlat = gdf.geometry.to_crs('EPSG:4326')
    area_perim = [geod.geometry_area_perimeter(g) for g in lonlat]
    area = np.abs(np.array([a for a, _ in area_perim]))
    perim = np.array([p for _, p in area_perim])
    out = gdf.copy()
    out['AREA_KM2'] = area / 1e6
    out['PERIM_KM'] = perim / 1e3
    out['SHAPE_RATIO'] = perim / np.sqrt(area)
    return out


# =====================================================================
# Part 2: transects
# =====================================================================

def evenspace(xy, sep, start=0):
    """
    Generate evenly spaced points along a polyline at specified intervals.

    Parameters
    ----------
    xy : array-like of shape (N, 2)
        Polyline vertices [x, y] (a closed ring has first == last).
    sep : float
        Separation between consecutive points along the line (map units).
    start : float, optional
        Distance along the line of the first point. Default 0.

    Returns
    -------
    DataFrame
        Columns x, y (points on the line), x0, y0, x1, y1 (the segment each
        point lies on), theta (segment angle, radians). The last point is
        dropped so a closed ring doesn't get a point at both start and end.
    """
    xy = np.asarray(xy, dtype=float)

    dx = np.concatenate([[0], np.diff(xy[:, 0])])
    dy = np.concatenate([[0], np.diff(xy[:, 1])])
    dseg = np.sqrt(dx**2 + dy**2)        # dseg[k] = length of segment k-1 -> k
    dtotal = np.cumsum(dseg)             # dtotal[k] = distance along line at vertex k
    linelength = dtotal[-1]

    pos = np.arange(start, linelength, sep)
    pos = pos[:-1]  # drop last point to avoid duplicating the start of a closed ring
    if len(pos) == 0:
        return pd.DataFrame(columns=['x', 'y', 'x0', 'y0', 'x1', 'y1', 'theta'])

    # Segment s (vertex s -> s+1) with dtotal[s] <= pos < dtotal[s+1]
    whichseg = np.searchsorted(dtotal, pos, side='right') - 1
    whichseg = np.clip(whichseg, 0, len(xy) - 2)

    x0, y0 = xy[whichseg, 0], xy[whichseg, 1]
    x1, y1 = xy[whichseg + 1, 0], xy[whichseg + 1, 1]
    f = (pos - dtotal[whichseg]) / dseg[whichseg + 1]

    return pd.DataFrame({
        'x': x0 + f * (x1 - x0),
        'y': y0 + f * (y1 - y0),
        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
        'theta': np.arctan2(y0 - y1, x0 - x1),
    })


def transect(tpts, tlen, npts=1):
    """
    Generate perpendicular transect points from boundary sample points.

    Parameters
    ----------
    tpts : DataFrame
        Output of evenspace() (x, y, theta).
    tlen : float
        Distance between consecutive points along each transect (map units).
    npts : int, optional
        Points on each side of the boundary point. Total per transect = 2*npts + 1.

    Returns
    -------
    DataFrame
        Columns transectID (int, 1-indexed), pointID (int: negative = inward,
        0 = boundary, positive = outward), x, y; sorted by transectID, pointID.
        Inner points are on the left of the segment direction, which is inside
        the PA for a clockwise ring (see extract_coords).
    """
    if len(tpts) == 0:
        return pd.DataFrame(columns=['transectID', 'pointID', 'x', 'y'])

    theta_t = tpts['theta'].to_numpy() + np.pi / 2
    dx = tlen * np.cos(theta_t)
    dy = tlen * np.sin(theta_t)
    x = tpts['x'].to_numpy()
    y = tpts['y'].to_numpy()

    offsets = np.arange(-npts, npts + 1)              # -npts ... 0 ... +npts
    xx = x[:, None] - offsets[None, :] * dx[:, None]  # inner (negative) = x + k*dx
    yy = y[:, None] - offsets[None, :] * dy[:, None]

    n = len(tpts)
    return pd.DataFrame({
        'transectID': np.repeat(np.arange(1, n + 1), len(offsets)),
        'pointID': np.tile(offsets, n),
        'x': xx.ravel(),
        'y': yy.ravel(),
    })


def extract_coords(geom):
    """
    Extract exterior boundary coordinates from a polygon geometry.

    For MultiPolygon geometries, uses the largest component polygon by area.
    The exterior ring is forced clockwise (orient sign=-1.0) so that transect()
    places inner points inside the PA, regardless of the input file format.

    Returns
    -------
    numpy.ndarray or None
        (N, 2) clockwise exterior coordinates, or None for other geometry types.
    """
    if geom.geom_type == 'Polygon':
        return np.array(orient(geom, sign=-1.0).exterior.coords)
    if geom.geom_type == 'MultiPolygon':
        largest = max(geom.geoms, key=lambda p: p.area)
        return np.array(orient(largest, sign=-1.0).exterior.coords)
    return None


@lru_cache(maxsize=8)
def _to_lonlat(src_crs):
    """Cached Transformer from ``src_crs`` to EPSG:4326 (lon, lat)."""
    return Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)


def _transform(geom, transformer):
    """Apply a pyproj Transformer to a shapely geometry."""
    return shapely.transform(
        geom, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1]))
    )


def make_pa_transects(geom, src_crs, sample_dist, transect_unit, npts, buffer_dist, eps=1.0):
    """
    Build and check the transects of one protected area.

    The PA is projected into an azimuthal equidistant (AEQD) projection centered
    on its centroid, so the boundary spacing, transect spacing, and inner buffer
    are true distances in meters and the transects are truly perpendicular.

    Whole-line check: a transect is kept only if its inner half (pt_-npts to
    1 m before pt_0) lies within the PA, and its outer half (1 m past pt_0 to
    pt_+npts) does not touch the PA. Together these mean the inner points are
    inside, the outer points are outside, and the line crosses the boundary
    once.

    Parameters
    ----------
    geom : shapely Polygon
        PA geometry in ``src_crs``.
    src_crs : str or pyproj.CRS
        CRS of ``geom``.
    sample_dist, transect_unit : float
        Boundary point spacing and transect point spacing (meters).
    npts : int
        Points on each side of the boundary.
    buffer_dist : float
        Inner buffer distance (meters). The PA is skipped if no interior point is
        farther than this from the boundary (radius of the largest inscribed
        circle <= buffer_dist), i.e. the inward buffer would be empty.
    eps : float, optional
        Gap (meters) left around pt_0 in the whole-line check. Default 1.

    Returns
    -------
    dict
        status ('ok', 'empty_inner_buffer', 'no_boundary_points'), df (kept
        points: transectID, pointID, lon, lat), n_generated, n_bad_inner,
        n_bad_outer, and max_gap_pct: the longest run of consecutive dropped
        transects as a percentage of all transect positions on the boundary
        (0 = full coverage; NaN when none are kept).
    """
    empty = pd.DataFrame(columns=['transectID', 'pointID', 'lon', 'lat'])
    result = {'status': 'ok', 'df': empty, 'n_generated': 0, 'n_bad_inner': 0, 'n_bad_outer': 0,
              'max_gap_pct': np.nan}

    to_ll = _to_lonlat(src_crs)
    c = geom.centroid
    lon0, lat0 = to_ll.transform(c.x, c.y)
    aeqd = Proj(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84 +units=m')
    poly = shapely.transform(
        _transform(geom, to_ll), lambda xy: np.column_stack(aeqd(xy[:, 0], xy[:, 1]))
    )

    # Equivalent to poly.buffer(-buffer_dist).is_empty, but much faster and without
    # the tiny slivers a negative buffer can leave
    if shapely.maximum_inscribed_circle(poly, tolerance=1.0).length <= buffer_dist:
        result['status'] = 'empty_inner_buffer'
        return result

    coords = extract_coords(poly)
    pts = evenspace(coords, sample_dist) if coords is not None and len(coords) >= 3 else []
    if len(pts) == 0:
        result['status'] = 'no_boundary_points'
        return result

    tr = transect(pts, transect_unit, npts=npts)
    k = 2 * npts + 1
    n = len(pts)
    X = tr['x'].to_numpy().reshape(n, k)
    Y = tr['y'].to_numpy().reshape(n, k)

    # Unit vector from innermost to outermost point
    ux, uy = X[:, -1] - X[:, 0], Y[:, -1] - Y[:, 0]
    norm = np.hypot(ux, uy)
    ux, uy = ux / norm, uy / norm

    inner = shapely.linestrings(np.stack([
        np.column_stack([X[:, 0], Y[:, 0]]),
        np.column_stack([X[:, npts] - eps * ux, Y[:, npts] - eps * uy]),
    ], axis=1))
    outer = shapely.linestrings(np.stack([
        np.column_stack([X[:, npts] + eps * ux, Y[:, npts] + eps * uy]),
        np.column_stack([X[:, -1], Y[:, -1]]),
    ], axis=1))

    shapely.prepare(poly)
    inner_ok = shapely.covers(poly, inner)
    outer_ok = ~shapely.intersects(poly, outer)
    keep = inner_ok & outer_ok

    result['n_generated'] = n
    result['n_bad_inner'] = int((~inner_ok).sum())
    result['n_bad_outer'] = int((inner_ok & ~outer_ok).sum())

    # Evenness: longest run of dropped transects between kept ones (circular), as a
    # share of the boundary
    kept_idx = np.flatnonzero(keep)
    if len(kept_idx):
        dropped_runs = np.diff(np.r_[kept_idx, kept_idx[0] + n]) - 1
        result['max_gap_pct'] = float(100 * dropped_runs.max() / n)

    if keep.any():
        lon, lat = aeqd(X[keep].ravel(), Y[keep].ravel(), inverse=True)
        result['df'] = pd.DataFrame({
            'transectID': np.repeat(np.flatnonzero(keep) + 1, k).astype('int32'),
            'pointID': np.tile(np.arange(-npts, npts + 1), int(keep.sum())).astype('int8'),
            'lon': lon,
            'lat': lat,
        })
    return result


def flag_other_pa(points, tree, tree_pids, own_pids):
    """
    Flag transects whose outer points fall inside another protected area.

    Parameters
    ----------
    points : DataFrame
        Transect points with WDPA_PID, transectID, pointID, lon, lat.
    tree : shapely.STRtree
        Index of reference PA polygons (EPSG:4326).
    tree_pids : numpy.ndarray of str
        WDPA_PID of each polygon in ``tree``.
    own_pids : dict
        PID -> set of PIDs that count as the same PA (itself, and PAs merged
        into it or removed as its duplicate or nested PA).

    Returns
    -------
    numpy.ndarray of int8
        1 if any outer point (pointID > 0) of the row's transect lies within a
        PA that is not one of its own PIDs, else 0; one value per row of ``points``.
    """
    outer = points[points['pointID'] > 0]
    pt_idx, poly_idx = tree.query(
        shapely.points(outer['lon'].to_numpy(), outer['lat'].to_numpy()), predicate='within'
    )
    own = outer['WDPA_PID'].to_numpy()[pt_idx]
    hit = tree_pids[poly_idx]
    other = np.array([h not in own_pids.get(o, {o}) for o, h in zip(own, hit)], dtype=bool)

    flagged = set(zip(own[other], outer['transectID'].to_numpy()[pt_idx][other]))
    keys = zip(points['WDPA_PID'].to_numpy(), points['transectID'].to_numpy())
    return np.fromiter((k in flagged for k in keys), dtype=np.int8, count=len(points))


def write_upload_zip(points, zip_path):
    """
    Write transect points as a zipped shapefile for Earth Engine table upload.

    Fields: WDPA_PID (str), transectID (int), pointID (int); point geometry in
    EPSG:4326. The shapefile is written to a local temporary folder (fast) and
    only the finished zip is written to ``zip_path``; writing a shapefile
    directly to a slow mounted folder takes about a minute per file.
    """
    zip_path = Path(zip_path)
    gdf = gpd.GeoDataFrame(
        {
            'WDPA_PID': points['WDPA_PID'].astype(str).to_numpy(),
            'transectID': points['transectID'].astype('int32').to_numpy(),
            'pointID': points['pointID'].astype('int32').to_numpy(),
        },
        geometry=gpd.points_from_xy(points['lon'], points['lat']),
        crs='EPSG:4326',
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # RESIZE=YES shrinks the text field widths in the .dbf to the longest value
        gdf.to_file(tmp / f'{zip_path.stem}.shp', engine='pyogrio', layer_options={'RESIZE': 'YES'})
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.iterdir()):
                z.write(f, arcname=f.name)


def generate_transects(pas, src_crs, sample_dist, transect_unit, npts, buffer_dist,
                       chunk_size, upload_dir, upload_prefix, points_path,
                       tree, tree_pids, own_pids, max_gap_pct_limit=None):
    """
    Generate transects for every PA and write the upload zips and points table.

    PAs are processed one at a time (make_pa_transects). Every ``chunk_size`` PAs
    that have at least one kept transect, the chunk is flagged for other-PA
    overlap, written as one upload zip ``{upload_prefix}{NNN}.zip``, and
    appended to ``points_path`` (Parquet). Each upload file holds whole PAs only.
    Old upload zips with the same prefix are deleted first.

    If ``max_gap_pct_limit`` is set, a PA whose longest run of dropped
    transects exceeds that percentage of its boundary is removed (status
    'coverage_gap') before anything is written, so its boundary isn't sampled
    unevenly.

    Returns
    -------
    dict
        status (PID -> 'ok', 'empty_inner_buffer', 'no_boundary_points',
        'no_valid_transects' or 'coverage_gap'), n_transects (PID -> transects
        that passed the whole-line check; written only for status 'ok'),
        n_generated (PID -> transects generated before the whole-line check),
        max_gap_pct (PID -> longest boundary gap between kept transects, % of boundary),
        zip_files, and transect counts: generated, removed_inner_half_leaves_pa,
        removed_outer_half_enters_pa, in_pas_removed_for_coverage_gap, kept,
        flagged_in_other_pa, points.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    for old in upload_dir.glob(f'{upload_prefix}*.zip'):
        old.unlink()
    Path(points_path).parent.mkdir(parents=True, exist_ok=True)

    status, n_transects, n_generated, max_gap_pct, zip_files = {}, {}, {}, {}, []
    totals = dict(generated=0, removed_inner_half_leaves_pa=0, removed_outer_half_enters_pa=0,
                  in_pas_removed_for_coverage_gap=0, kept=0, flagged_in_other_pa=0, points=0)
    writer = None
    chunk = []

    def flush(chunk_frames):
        nonlocal writer
        pts = pd.concat(chunk_frames, ignore_index=True)
        pts['in_other_pa'] = flag_other_pa(pts, tree, tree_pids, own_pids)
        zip_path = upload_dir / f'{upload_prefix}{len(zip_files):03d}.zip'
        write_upload_zip(pts, zip_path)
        zip_files.append(str(zip_path))
        table = pa.Table.from_pandas(
            pts[['WDPA_PID', 'transectID', 'pointID', 'lon', 'lat', 'in_other_pa']], preserve_index=False
        )
        if writer is None:
            writer = pq.ParquetWriter(points_path, table.schema)
        writer.write_table(table)
        totals['flagged_in_other_pa'] += int(pts.loc[pts['pointID'] == 0, 'in_other_pa'].sum())
        totals['points'] += len(pts)
        print(f"  wrote {zip_path.name}: {pts['WDPA_PID'].nunique()} PAs, {len(pts):,} points")

    pids = pas['WDPA_PID'].to_numpy()
    for n, (pid, geom) in enumerate(zip(pids, pas.geometry.values), start=1):
        res = make_pa_transects(geom, src_crs, sample_dist, transect_unit, npts, buffer_dist)
        totals['generated'] += res['n_generated']
        totals['removed_inner_half_leaves_pa'] += res['n_bad_inner']
        totals['removed_outer_half_enters_pa'] += res['n_bad_outer']
        df = res['df']
        n_kept = len(df) // (2 * npts + 1)          # passed the whole-line check
        if res['status'] == 'ok' and n_kept == 0:
            res['status'] = 'no_valid_transects'
        elif (res['status'] == 'ok' and max_gap_pct_limit is not None
              and res['max_gap_pct'] > max_gap_pct_limit):
            res['status'] = 'coverage_gap'
            totals['in_pas_removed_for_coverage_gap'] += n_kept
        status[pid] = res['status']
        n_transects[pid] = n_kept
        n_generated[pid] = res['n_generated']
        max_gap_pct[pid] = res['max_gap_pct']
        if res['status'] == 'ok':
            totals['kept'] += n_transects[pid]
            df.insert(0, 'WDPA_PID', pid)
            chunk.append(df)
            if len(chunk) == chunk_size:
                flush(chunk)
                chunk = []
        if n % 500 == 0:
            print(f"  {n:,}/{len(pids):,} PAs processed, {totals['kept']:,} transects kept")

    if chunk:
        flush(chunk)
    if writer is not None:
        writer.close()

    return {'status': status, 'n_transects': n_transects, 'n_generated': n_generated,
            'max_gap_pct': max_gap_pct, 'zip_files': zip_files, 'totals': totals}

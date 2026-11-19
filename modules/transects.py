"""
Geometry processing and transect generation functions (script 02).

Part 1, choosing PAs:
- drop_excluded_pids: drop PAs listed in config.EXCLUDED_PIDS
- fill_holes: fill holes smaller than a threshold
- resolve_overlaps: drop duplicate and nested PAs (spatial index)
- merge_same_name: dissolve touching/overlapping PAs with the same name and country
- add_shape_metrics: geodesic area, perimeter, and perimeter/sqrt(area)

Part 2, transects:
- evenspace / transect / extract_coords: boundary points and perpendicular transects
- make_pa_transects: one PA, in its own azimuthal equidistant projection, with the
  whole-line check
- own_pid_sets: the PIDs that count as the same PA (merged, duplicate or nested)
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

def drop_excluded_pids(gdf, excluded):
    """Drop rows whose WDPA_PID (as text) is in `excluded`; returns (rows left, number dropped).

    Script 01 applies the same list in Earth Engine; applying it here as well makes an export made
    before a PID was added to the list give the same PA set as a fresh one.
    """
    drop = gdf['WDPA_PID'].astype(str).isin({str(p) for p in excluded})
    return gdf.loc[~drop].copy(), int(drop.sum())


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


def fill_holes(gdf, max_hole_area=2250000):
    """Fill holes of at most `max_hole_area` m² (default 1,500 m × 1,500 m, a 3 × 3 MODIS pixel
    window); larger holes stay. `gdf` in a metric CRS. Returns (copy, number of PAs changed).
    """
    filled = [_fill_small_holes(g, max_hole_area) for g in gdf.geometry]
    n_changed = int(sum(not a.equals(b) for a, b in zip(gdf.geometry, filled)))
    result = gdf.copy()
    result['geometry'] = filled
    return result, n_changed


def get_min_year_from_group(group_df):
    """The row of `group_df` with the oldest non-zero STATUS_YR (the first row if all are 0)."""
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
    """Remove duplicate and nested PAs (candidate pairs from a spatial index; `gdf` equal-area).

    - Duplicates (overlap > `overlap_threshold` % of both PAs): grouped transitively (A~B, B~C);
      the oldest non-zero STATUS_YR is kept.
    - Nested (> threshold of the smaller only): the smaller is removed; a PA inside several larger
      ones is assigned to the largest.

    Returns (kept PAs, removed rows with 'reason' and 'replaced_by', counts).
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
    """Dissolve PAs with the same ORIG_NAME + ISO3 that touch or overlap (directly or through each
    other). The merged PA takes the oldest member's WDPA_PID and the oldest non-zero STATUS_YR; its
    other attributes come from the largest member ('ATTR_PID'); 'MERGED_PID' lists the other
    members. Members that don't touch, or whose union is not a single Polygon, stay separate. Small
    holes are filled again (`max_hole_area`).

    Returns (PAs after merging, merged-away rows with 'reason' and 'replaced_by', counts).
    """
    gdf = gdf.reset_index(drop=True).copy()
    gdf['MERGED_PID'] = ''
    gdf['ATTR_PID'] = gdf['WDPA_PID']
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
            group = gdf.iloc[members]
            keep = get_min_year_from_group(group).name            # PID: oldest member
            largest = group.geometry.area.idxmax()                # attributes: largest member
            oldest_year = group['STATUS_YR'][group['STATUS_YR'] != 0].min()
            others = [m for m in members if m != keep]
            attr_cols = [c for c in gdf.columns
                         if c not in ('WDPA_PID', 'geometry', 'MERGED_PID', 'ATTR_PID', 'STATUS_YR')]
            gdf.loc[keep, attr_cols] = gdf.loc[largest, attr_cols].to_numpy()
            gdf.at[keep, 'STATUS_YR'] = 0 if pd.isna(oldest_year) else oldest_year
            gdf.at[keep, 'ATTR_PID'] = pids[largest]
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
    """Add geodesic AREA_KM2, PERIM_KM (holes included) and SHAPE_RATIO = perimeter / sqrt(area)
    (circle 3.54, square 4.0), on the WGS84 ellipsoid. For information: no PA is removed on shape.
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
    """Points every `sep` map units along polyline `xy` (N x 2; a closed ring has first == last),
    starting at distance `start`. Returns x, y, the segment each point lies on (x0, y0, x1, y1) and
    its angle theta. The last point before the end is dropped, so on a closed ring the gap between
    the last and the first point is more than `sep` and at most 2 * `sep` (never shorter: points
    closer than `sep` at the seam would give near-duplicate transects).
    """
    xy = np.asarray(xy, dtype=float)

    dx = np.concatenate([[0], np.diff(xy[:, 0])])
    dy = np.concatenate([[0], np.diff(xy[:, 1])])
    dseg = np.sqrt(dx**2 + dy**2)        # dseg[k] = length of segment k-1 -> k
    dtotal = np.cumsum(dseg)             # dtotal[k] = distance along line at vertex k
    linelength = dtotal[-1]

    pos = np.arange(start, linelength, sep)
    pos = pos[:-1]  # drop the last point: the seam gap of a closed ring is then in (sep, 2 * sep]
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
    """Transect points perpendicular to the boundary at each evenspace() point: `npts` points on each
    side, `tlen` apart. Returns transectID (1-based), pointID (negative = inward, 0 = boundary,
    positive = outward), x, y. Inner points are on the right of the direction of travel along the
    ring (x0, y0 -> x1, y1), i.e. inside the PA for a clockwise ring (extract_coords); theta points
    the opposite way (x1 -> x0).
    """
    if len(tpts) == 0:
        return pd.DataFrame(columns=['transectID', 'pointID', 'x', 'y'])

    theta_t = tpts['theta'].to_numpy() + np.pi / 2
    dx = tlen * np.cos(theta_t)
    dy = tlen * np.sin(theta_t)
    x = tpts['x'].to_numpy()
    y = tpts['y'].to_numpy()

    offsets = np.arange(-npts, npts + 1)
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
    """Clockwise exterior-ring coordinates (N x 2) of a Polygon, or of a MultiPolygon's largest part,
    so that transect() puts inner points inside the PA; None for other geometry types.
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
    """Build and check the transects of one PA, in an azimuthal equidistant projection centred on it
    (true distances in meters, truly perpendicular transects).

    The PA is skipped if no interior point is farther than `buffer_dist` from the boundary (largest
    inscribed circle). Whole-line check: a transect is kept only if its inner half (up to `eps` m
    before pt_0) lies within the PA and its outer half (from `eps` m past pt_0) does not touch it,
    so the line crosses the boundary once.

    Returns a dict: status ('ok', 'empty_inner_buffer', 'no_boundary_points'), df (kept points:
    transectID, pointID, lon, lat), n_generated, n_bad_inner, n_bad_outer, and max_gap_pct (the
    longest run of dropped transects, % of all positions; NaN when none are kept).
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


def own_pid_sets(kept_pids, removed_frames):
    """For each kept PA, the PIDs that count as the same PA for the other-PA flag: itself and every PA
    merged into it or removed as its duplicate or nested PA, following the 'replaced_by' chains of
    `removed_frames` (nested -> duplicate -> kept). A chain that does not end at a kept PA is ignored.
    """
    replaced = {}
    for frame in removed_frames:
        for pid, rep in zip(frame['WDPA_PID'], frame['replaced_by']):
            if rep:
                replaced[pid] = rep

    def resolve(pid):
        seen = set()
        while pid in replaced and pid not in seen:   # `seen` guards against a cycle
            seen.add(pid)
            pid = replaced[pid]
        return pid

    own = {pid: {pid} for pid in kept_pids}
    for pid in replaced:
        root = resolve(pid)
        if root in own:
            own[root].add(pid)
    return own


def flag_other_pa(points, tree, tree_pids, own_pids):
    """1 for each row of `points` whose transect has an outer point (pointID > 0) inside a PA of `tree`
    (EPSG:4326, PIDs `tree_pids`) that is not one of its own PIDs (`own_pids`), else 0.
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
    """Write transect points (WDPA_PID, transectID, pointID; EPSG:4326) as a zipped shapefile for
    Earth Engine upload. The shapefile is built in a local temporary folder: writing it directly
    to a slow mounted folder takes about a minute per file."""
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
    """Transects for every PA (make_pa_transects), written as upload zips of `chunk_size` whole PAs
    (`{upload_prefix}NNN.zip`, flagged for other-PA overlap) and appended to `points_path`
    (Parquet, with each PA's upload file number in column 'asset', read by script 04). Old zips
    with the prefix are deleted first. With `max_gap_pct_limit`, a PA whose longest unsampled
    stretch exceeds that % of its boundary is removed ('coverage_gap') before anything is written.

    Returns a dict: status, n_transects (passed the whole-line check), n_generated and max_gap_pct
    per PID; zip_files; and the transect totals (generated, removed_inner_half_leaves_pa,
    removed_outer_half_enters_pa, in_pas_removed_for_coverage_gap, kept, flagged_in_other_pa, points).
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
        asset = len(zip_files)                  # the upload file / Earth Engine asset NNN
        zip_path = upload_dir / f'{upload_prefix}{asset:03d}.zip'
        write_upload_zip(pts, zip_path)
        zip_files.append(str(zip_path))
        pts['asset'] = np.int16(asset)
        table = pa.Table.from_pandas(
            pts[['WDPA_PID', 'transectID', 'pointID', 'lon', 'lat', 'in_other_pa', 'asset']],
            preserve_index=False
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

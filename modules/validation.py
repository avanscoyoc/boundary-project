"""
Validation: the displaced-transect null model (Methods: Validation).

Each sampled PA's full transect set is translated by one random vector, without rotation,
in a local azimuthal equidistant projection centred on the PA, so the number, spacing,
orientation and clustering of the transects are kept.

- sample_pas: stratified random draw order of PAs within each biome (with reserves)
- displace / draw_vector / footprint_m: the translation, distance in [d, mult × d]
- transect_lines / boundary_share: the share of displaced transects with a PA boundary near
  their centre point (between points -1 and 1)
- candidate_displacements: vectors drawn until enough candidates pass the local checks
- displacement_centre / displace_polygon: the PA polygon moved with its transects (its biome is
  taken as for the PA, script 01: the largest-area ecoregion, script 12)
- candidate_checks: per PA, the first candidate whose copy passes the analysis-set rules of
  R script 05 (biome, sea, lake, years; on the Earth Engine samples, script 13)
- biome_from_num: RESOLVE BIOME_NUM -> the 8 biome groups of script 04 (point diagnostics)

The edge metrics of the accepted displaced transects are computed with the functions of
script 04 (modules.analysis), so PA and displaced copy are measured identically.
"""

import numpy as np
import pandas as pd
import pyproj
import shapely

from modules.analysis import recategorize_biome

POINT_IDS = (-2, -1, 0, 1, 2)
CANDIDATE_SEP = '#'   # upload key: "<WDPA_PID>#<candidate>"

# RESOLVE Ecoregions 2017 BIOME_NUM -> BIOME_NAME (codes outside 1-14, e.g. Rock & Ice, -> None)
RESOLVE_BIOMES = {
    1: "Tropical & Subtropical Moist Broadleaf Forests",
    2: "Tropical & Subtropical Dry Broadleaf Forests",
    3: "Tropical & Subtropical Coniferous Forests",
    4: "Temperate Broadleaf & Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests/Taiga",
    7: "Tropical & Subtropical Grasslands, Savannas & Shrublands",
    8: "Temperate Grasslands, Savannas & Shrublands",
    9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves",
}


def biome_from_num(num):
    """RESOLVE BIOME_NUM (array-like, may contain NaN) -> biome group name or None."""
    out = []
    for x in np.asarray(num, dtype='float64'):
        name = RESOLVE_BIOMES.get(int(x)) if np.isfinite(x) else None
        out.append(recategorize_biome(name) if name is not None else None)
    return np.array(out, dtype=object)


def sample_pas(pas, n_per_biome, reserve, seed):
    """
    Stratified random draw order of PAs within each biome.

    ``pas``: DataFrame with WDPA_PID and BIOME_NAME. PAs are sorted by WDPA_PID and biomes
    by name before drawing, so the result depends only on the PAs and the seed. Returns the
    first ``n_per_biome + reserve`` PAs of each biome (all of a smaller biome) with ``draw``
    = 1, 2, ... in draw order; PAs after the first ``n_per_biome`` are reserves.
    """
    rng = np.random.default_rng(seed)
    pas = pas[['WDPA_PID', 'BIOME_NAME']].copy()
    pas['WDPA_PID'] = pas['WDPA_PID'].astype(str)
    pas = pas.sort_values('WDPA_PID', kind='stable')
    out = []
    for biome in sorted(pas['BIOME_NAME'].dropna().unique()):
        g = pas[pas['BIOME_NAME'] == biome]
        g = g.iloc[rng.permutation(len(g))].head(n_per_biome + reserve).copy()
        g['draw'] = np.arange(1, len(g) + 1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def aeqd_transformers(lon0, lat0):
    """(forward, inverse) transformers between EPSG:4326 and the AEQD projection at lon0, lat0."""
    aeqd = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    inv = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    return fwd, inv


def footprint_m(x, y):
    """Diagonal of the bounding box of projected points (m): at least the PA's largest extent."""
    return float(np.hypot(np.ptp(x), np.ptp(y)))


def displacement_centre(points):
    """(lon0, lat0) of the local projection of a PA's displacement: the mean of its transect points."""
    return float(points['lon'].mean()), float(points['lat'].mean())


def displace_polygon(geom, lon0, lat0, dx, dy):
    """
    The PA polygon translated by (dx, dy) m in the AEQD projection at (lon0, lat0), the same
    translation as its transect points (candidate_displacements, displacement_centre).
    ``geom``: shapely geometry in EPSG:4326. Returns the translated geometry in EPSG:4326.
    """
    fwd, inv = aeqd_transformers(lon0, lat0)

    def move(xy):
        x, y = fwd.transform(xy[:, 0], xy[:, 1])
        return np.column_stack(inv.transform(np.asarray(x) + dx, np.asarray(y) + dy))
    return shapely.transform(geom, move)


def write_polygon_zip(keys, geoms, zip_path):
    """
    Displaced PA polygons as a zipped shapefile for Earth Engine table upload: field WDPA_PID
    (the key "<WDPA_PID>#<candidate>"), geometry in EPSG:4326. Written in a temporary folder
    first, as modules.preprocessing.write_upload_zip.
    """
    import tempfile
    import zipfile
    from pathlib import Path
    import geopandas as gpd
    zip_path = Path(zip_path)
    gdf = gpd.GeoDataFrame({'WDPA_PID': np.asarray(keys, dtype=str)}, geometry=list(geoms), crs='EPSG:4326')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gdf.to_file(tmp / f'{zip_path.stem}.shp', engine='pyogrio', layer_options={'RESIZE': 'YES'})
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.iterdir()):
                z.write(f, arcname=f.name)


def draw_vector(rng, d_min, d_max):
    """Random displacement: bearing uniform in [0, 360) degrees, distance uniform in
    [d_min, d_max] (m). Returns bearing, distance, dx (east), dy (north)."""
    bearing = rng.uniform(0.0, 360.0)
    dist = rng.uniform(d_min, d_max)
    rad = np.deg2rad(bearing)
    return bearing, dist, dist * np.sin(rad), dist * np.cos(rad)


def displace(x, y, dx, dy, inv):
    """Translate projected points by (dx, dy) and return lon, lat (EPSG:4326)."""
    return inv.transform(np.asarray(x) + dx, np.asarray(y) + dy)


def transect_lines(points, ends=(POINT_IDS[0], POINT_IDS[-1])):
    """
    One line per transect, from its point ``ends[0]`` to its point ``ends[1]``: by default the
    whole transect (-2 to 2); (-1, 1) gives its centre zone, within one point spacing of point 0.

    ``points``: DataFrame with transectID, pointID, lon, lat; every transect has both end
    points. Returns (transectIDs, shapely LineString array) in transectID order.
    """
    p = points.sort_values(['transectID', 'pointID'], kind='stable')
    first = p[p['pointID'] == ends[0]]
    last = p[p['pointID'] == ends[1]]
    if not np.array_equal(first['transectID'].to_numpy(), last['transectID'].to_numpy()):
        raise ValueError(f"every transect needs its points {ends[0]} and {ends[1]}")
    coords = np.stack([first[['lon', 'lat']].to_numpy(), last[['lon', 'lat']].to_numpy()], axis=1)
    return first['transectID'].to_numpy(), shapely.linestrings(coords)


def boundary_share(lines, tree):
    """Share of ``lines`` that intersect any PA boundary in ``tree`` (STRtree of boundaries)."""
    if len(lines) == 0 or len(tree.geometries) == 0:
        return 0.0
    hits = tree.query(lines, predicate='intersects')
    return len(np.unique(hits[0])) / len(lines)


def candidate_displacements(points, tree, rng, n_candidates, max_attempts, d_min, mult,
                            max_boundary_share):
    """
    Displacements of one PA's transect set that pass the checks needing no imagery.

    ``points``: the PA's transect points (transectID, pointID, lon, lat). The projection is
    centred on the mean point. The PA size d is the larger of ``d_min`` and the PA's footprint
    (so the copy cannot overlap the PA); distances are drawn uniformly in [d, ``mult`` × d].
    A vector is accepted when at most ``max_boundary_share`` of the displaced transects have a
    PA boundary in ``tree`` within their centre zone (points -1 to 1), and no transect spans
    more than 180 degrees of longitude (antimeridian). A boundary nearer the transect ends is
    allowed. Vectors are drawn until ``n_candidates`` are accepted or ``max_attempts`` are used.

    Returns (candidates, attempts, d_min_used): candidates is a list of dicts with candidate
    (1, 2, ...), attempt, bearing, dist_m, dx, dy, boundary_share and points (DataFrame:
    transectID, pointID, lon, lat of the displaced points).
    """
    lon0, lat0 = displacement_centre(points)
    fwd, inv = aeqd_transformers(lon0, lat0)
    x, y = fwd.transform(points['lon'].to_numpy(), points['lat'].to_numpy())
    d_min_used = max(d_min, footprint_m(x, y))
    candidates, attempts = [], 0
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        bearing, dist, dx, dy = draw_vector(rng, d_min_used, mult * d_min_used)
        lon, lat = displace(x, y, dx, dy, inv)
        moved = points[['transectID', 'pointID']].copy()
        moved['lon'], moved['lat'] = lon, lat
        _, lines = transect_lines(moved)
        span = np.abs(np.diff(shapely.get_coordinates(lines)[:, 0].reshape(-1, 2), axis=1))
        if (span > 180).any():
            continue
        share = boundary_share(transect_lines(moved, ends=(-1, 1))[1], tree)
        if share > max_boundary_share:
            continue
        candidates.append({'candidate': len(candidates) + 1, 'attempt': attempts, 'bearing': bearing,
                           'dist_m': dist, 'dx': dx, 'dy': dy, 'boundary_share': share, 'points': moved})
    return candidates, attempts, d_min_used


CHECK_RULES = ('biome_ok', 'sea_ok', 'lake_ok', 'years_ok')


def candidate_checks(candidates, pa_years, n_transects, copy_biome, pa_biome,
                     min_complete, max_water, min_years):
    """
    The analysis-set rules of R script 05 applied to each candidate copy; per PA, the first
    candidate (by number, i.e. draw order) passing all of them is accepted.

    ``candidates``: WDPA_PID, candidate (every candidate sent to Earth Engine).
    ``pa_years``: displaced_metrics() rows of the candidates (WDPA_PID, candidate, year, n_trnst,
    edge_extent, edge_intensity, water_extent_pct), one per candidate-year with at least one
    complete transect (as script 04). ``n_transects``: dict WDPA_PID -> transects of the PA (= of
    each copy). ``copy_biome``: WDPA_PID, candidate, biome (group of the displaced polygon's
    largest-area ecoregion, or None). ``pa_biome``: dict WDPA_PID -> biome group.

    Rules, as 05 (modules/r/prep.R analysis_set) on 04's PA-year table:
      biome_ok  the copy's biome is the PA's;
      sea_ok    pct_complete (median over the copy's years of 100 × n_trnst / transects) >= min_complete;
      lake_ok   water_extent_pct (mean over years) <= max_water, or missing;
      years_ok  >= min_years years of edge_extent and of edge_intensity.
    Returns one row per candidate with those values, passes, accepted.
    """
    base = candidates[['WDPA_PID', 'candidate']].drop_duplicates().copy()
    py = pa_years.copy()
    py['pct_complete'] = 100 * py['n_trnst'] / py['WDPA_PID'].map(n_transects)
    agg = (py.groupby(['WDPA_PID', 'candidate'])
             .agg(n_years=('year', 'size'), pct_complete=('pct_complete', 'median'),
                  water_extent_pct=('water_extent_pct', 'mean'),
                  n_years_edge_extent=('edge_extent', 'count'),
                  n_years_edge_intensity=('edge_intensity', 'count'))
             .reset_index())
    out = base.merge(agg, on=['WDPA_PID', 'candidate'], how='left')
    out = out.merge(copy_biome[['WDPA_PID', 'candidate', 'biome']], on=['WDPA_PID', 'candidate'], how='left')
    for c in ('n_years', 'n_years_edge_extent', 'n_years_edge_intensity'):
        out[c] = out[c].fillna(0).astype(int)
    out['biome_ok'] = [b is not None and isinstance(b, str) and b == pa_biome.get(pid)
                       for pid, b in zip(out['WDPA_PID'], out['biome'])]
    out['sea_ok'] = out['pct_complete'].notna() & (out['pct_complete'] >= min_complete)
    out['lake_ok'] = out['water_extent_pct'].isna() | (out['water_extent_pct'] <= max_water)
    out['years_ok'] = (out['n_years_edge_extent'] >= min_years) & (out['n_years_edge_intensity'] >= min_years)
    out['passes'] = out[list(CHECK_RULES)].all(axis=1)
    first = out[out['passes']].groupby('WDPA_PID')['candidate'].min()
    out['accepted'] = out['candidate'].to_numpy() == out['WDPA_PID'].map(first).to_numpy()
    return out.sort_values(['WDPA_PID', 'candidate'], kind='stable').reset_index(drop=True)


def split_key(key):
    """Upload key "<WDPA_PID>#<candidate>" -> (WDPA_PID, candidate as int)."""
    pid, cand = np.char.rpartition(np.asarray(key, dtype=str), CANDIDATE_SEP)[:, [0, 2]].T
    return pid, cand.astype(int)


def displaced_metrics(df, years):
    """
    PA-year edge metrics of transect points, with script 04's functions (all transects only).

    ``df``: points as read by modules.analysis.read_asset_csvs (WDPA_PID, transectID, pointID,
    max_extent, gHM, elevation, terrain_slope, one column per year). Returns one row per
    PA-year with at least one complete transect: WDPA_PID, year, n_trnst, edge_extent, D02, D01,
    D0m1, D0m2, edge_intensity, water_extent_pct.
    """
    from modules.analysis import build_transect_years, pa_year_metrics
    ty, _ = build_transect_years(df, years)
    m = pa_year_metrics(ty)
    cols = ['WDPA_PID', 'year', 'n_trnst', 'edge_extent', 'D02', 'D01', 'D0m1', 'D0m2', 'edge_intensity',
            'water_extent_pct']
    return m[cols].sort_values(['WDPA_PID', 'year'], kind='stable').reset_index(drop=True)


def select_pas(sample, accepted, n_per_biome):
    """
    The validation PAs: per biome, in draw order, the first ``n_per_biome`` sampled PAs with an
    accepted displacement (reserves replace PAs without one). ``sample``: WDPA_PID, BIOME_NAME,
    draw; ``accepted``: set of WDPA_PIDs with an accepted candidate. Returns the WDPA_PIDs.
    """
    s = sample[sample['WDPA_PID'].isin(accepted)].sort_values(['BIOME_NAME', 'draw'], kind='stable')
    return s.groupby('BIOME_NAME', sort=True).head(n_per_biome)['WDPA_PID'].tolist()

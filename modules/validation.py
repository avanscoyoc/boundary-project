"""
Validation: the displaced-transect null model (methods: Validation). Draft for review.

Each sampled PA's full transect set is translated by one random vector, without rotation,
in a local azimuthal equidistant projection centred on the PA, so the number, spacing,
orientation and clustering of the transects are kept.

- sample_pas: stratified random draw order of PAs within each biome (with reserves)
- displace / draw_vector / footprint_m: the translation
- transect_lines / reaches_pa_boundary: no displaced transect may cross a PA boundary
- candidate_displacements: vectors drawn until enough candidates pass the local checks
- accept_candidates: per PA, the first candidate whose points all lie in the PA's biome and
  have gradient values in every year (checked on the Earth Engine samples, script 13)
- biome_from_num: RESOLVE BIOME_NUM -> the 8 biome groups of script 04

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


def draw_vector(rng, d_min, d_span):
    """Random displacement: bearing uniform in [0, 360) degrees, distance uniform in
    [d_min, d_min + d_span] (m). Returns bearing, distance, dx (east), dy (north)."""
    bearing = rng.uniform(0.0, 360.0)
    dist = rng.uniform(d_min, d_min + d_span)
    rad = np.deg2rad(bearing)
    return bearing, dist, dist * np.sin(rad), dist * np.cos(rad)


def displace(x, y, dx, dy, inv):
    """Translate projected points by (dx, dy) and return lon, lat (EPSG:4326)."""
    return inv.transform(np.asarray(x) + dx, np.asarray(y) + dy)


def transect_lines(points):
    """
    One line per transect, from its first (pointID -2) to its last point (pointID 2).

    ``points``: DataFrame with transectID, pointID, lon, lat; every transect has all 5
    points. Returns (transectIDs, shapely LineString array) in transectID order.
    """
    p = points.sort_values(['transectID', 'pointID'], kind='stable')
    first = p[p['pointID'] == POINT_IDS[0]]
    last = p[p['pointID'] == POINT_IDS[-1]]
    if not np.array_equal(first['transectID'].to_numpy(), last['transectID'].to_numpy()):
        raise ValueError("every transect needs its points -2 and 2")
    coords = np.stack([first[['lon', 'lat']].to_numpy(), last[['lon', 'lat']].to_numpy()], axis=1)
    return first['transectID'].to_numpy(), shapely.linestrings(coords)


def reaches_pa_boundary(lines, tree):
    """True if any transect line intersects any PA boundary in ``tree`` (STRtree of boundaries)."""
    if len(tree.geometries) == 0:
        return False
    hits = tree.query(lines, predicate='intersects')
    return hits.shape[1] > 0


def candidate_displacements(points, tree, rng, n_candidates, max_attempts, d_min, d_span):
    """
    Displacements of one PA's transect set that pass the checks needing no imagery.

    ``points``: the PA's transect points (transectID, pointID, lon, lat). The projection is
    centred on the mean point. A vector is accepted when its distance is at least the larger
    of ``d_min`` and the PA's footprint (so the copy cannot overlap the PA), no displaced
    transect crosses a PA boundary in ``tree``, and no line spans more than 180 degrees of
    longitude (antimeridian). Vectors are drawn until ``n_candidates`` are accepted or
    ``max_attempts`` are used.

    Returns (candidates, attempts, d_min_used): candidates is a list of dicts with candidate
    (1, 2, ...), attempt, bearing, dist_m, dx, dy and points (DataFrame: transectID,
    pointID, lon, lat of the displaced points).
    """
    lon0, lat0 = float(points['lon'].mean()), float(points['lat'].mean())
    fwd, inv = aeqd_transformers(lon0, lat0)
    x, y = fwd.transform(points['lon'].to_numpy(), points['lat'].to_numpy())
    d_min_used = max(d_min, footprint_m(x, y))
    candidates, attempts = [], 0
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        bearing, dist, dx, dy = draw_vector(rng, d_min_used, d_span)
        lon, lat = displace(x, y, dx, dy, inv)
        moved = points[['transectID', 'pointID']].copy()
        moved['lon'], moved['lat'] = lon, lat
        _, lines = transect_lines(moved)
        span = np.abs(np.diff(shapely.get_coordinates(lines)[:, 0].reshape(-1, 2), axis=1))
        if (span > 180).any() or reaches_pa_boundary(lines, tree):
            continue
        candidates.append({'candidate': len(candidates) + 1, 'attempt': attempts, 'bearing': bearing,
                           'dist_m': dist, 'dx': dx, 'dy': dy, 'points': moved})
    return candidates, attempts, d_min_used


def accept_candidates(points, pa_biome, years):
    """
    Per PA, the first candidate whose points all lie in the PA's biome and all have a
    gradient value in every year.

    ``points``: sampled candidate points with WDPA_PID, candidate, transectID, pointID,
    biome (group name or None) and one column per year (NaN = no data). ``pa_biome``: dict
    WDPA_PID -> biome group. Returns one row per PA and candidate: WDPA_PID, candidate,
    n_points, share_in_biome, share_complete, passes, accepted (True for the first passing
    candidate of a PA).
    """
    year_cols = [str(y) for y in years]
    p = points.copy()
    p['in_biome'] = [b is not None and b == pa_biome.get(pid)
                     for pid, b in zip(p['WDPA_PID'], p['biome'])]
    p['complete'] = p[year_cols].notna().all(axis=1)
    out = (p.groupby(['WDPA_PID', 'candidate'], sort=True)
             .agg(n_points=('pointID', 'size'), share_in_biome=('in_biome', 'mean'),
                  share_complete=('complete', 'mean'))
             .reset_index())
    out['passes'] = (out['share_in_biome'] == 1) & (out['share_complete'] == 1)
    first = out[out['passes']].groupby('WDPA_PID')['candidate'].min()
    out['accepted'] = out['candidate'].to_numpy() == out['WDPA_PID'].map(first).to_numpy()
    return out


def split_key(key):
    """Upload key "<WDPA_PID>#<candidate>" -> (WDPA_PID, candidate as int)."""
    pid, cand = np.char.rpartition(np.asarray(key, dtype=str), CANDIDATE_SEP)[:, [0, 2]].T
    return pid, cand.astype(int)


def displaced_metrics(df, years):
    """
    PA-year edge metrics of transect points, with script 04's functions (all transects only).

    ``df``: points as read by modules.analysis.read_asset_csvs (WDPA_PID, transectID, pointID,
    max_extent, gHM, elevation, terrain_slope, one column per year). Returns one row per
    PA-year: WDPA_PID, year, n_trnst, edge_extent, D02, D01, D0m1, D0m2, edge_intensity.
    """
    from modules.analysis import build_transect_years, pa_year_metrics
    ty, _ = build_transect_years(df, years)
    m = pa_year_metrics(ty)
    cols = ['WDPA_PID', 'year', 'n_trnst', 'edge_extent', 'D02', 'D01', 'D0m1', 'D0m2', 'edge_intensity']
    return m[cols].sort_values(['WDPA_PID', 'year'], kind='stable').reset_index(drop=True)


def select_pas(sample, accepted, n_per_biome):
    """
    The validation PAs: per biome, in draw order, the first ``n_per_biome`` sampled PAs with an
    accepted displacement (reserves replace PAs without one). ``sample``: WDPA_PID, BIOME_NAME,
    draw; ``accepted``: set of WDPA_PIDs with an accepted candidate. Returns the WDPA_PIDs.
    """
    s = sample[sample['WDPA_PID'].isin(accepted)].sort_values(['BIOME_NAME', 'draw'], kind='stable')
    return s.groupby('BIOME_NAME', sort=True).head(n_per_biome)['WDPA_PID'].tolist()

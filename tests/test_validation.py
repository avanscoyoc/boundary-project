"""Unit tests for modules/validation.py (displaced-transect null model, scripts 11-13)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Geod

from modules import config
from modules.validation import (sample_pas, aeqd_transformers, displace, draw_vector, footprint_m,
                                transect_lines, boundary_share, candidate_displacements,
                                displacement_centre, displace_polygon, write_polygon_zip,
                                candidate_checks, select_pas, biome_from_num, split_key,
                                displaced_metrics)

GEOD = Geod(ellps='WGS84')


def toy_transects(lon0=20.0, lat0=60.0, n=8):
    """n transects of 5 points 2.5 km apart, running north-south, spaced 5 km east-west."""
    fwd, inv = aeqd_transformers(lon0, lat0)
    rows = []
    for t in range(n):
        for p in (-2, -1, 0, 1, 2):
            rows.append((t, p, t * 5000.0, p * 2500.0))
    df = pd.DataFrame(rows, columns=['transectID', 'pointID', 'x', 'y'])
    df['lon'], df['lat'] = inv.transform(df['x'].to_numpy(), df['y'].to_numpy())
    return df[['transectID', 'pointID', 'lon', 'lat']]


def test_sample_pas_stratified_reproducible_and_order_free():
    pas = pd.DataFrame({'WDPA_PID': [str(i) for i in range(30)],
                        'BIOME_NAME': ['A'] * 20 + ['B'] * 10})
    s = sample_pas(pas, n_per_biome=5, reserve=3, seed=1)
    assert s.groupby('BIOME_NAME').size().to_dict() == {'A': 8, 'B': 8}
    assert s.groupby('BIOME_NAME')['draw'].apply(list).to_dict() == {'A': list(range(1, 9)), 'B': list(range(1, 9))}
    pd.testing.assert_frame_equal(s, sample_pas(pas.iloc[::-1], 5, 3, seed=1))   # input order does not matter
    assert not s.equals(sample_pas(pas, 5, 3, seed=2))
    assert len(sample_pas(pas, 50, 0, seed=1)) == 30                            # small biomes: all PAs


def test_displacement_keeps_shape_and_distance():
    pts = toy_transects()
    fwd, inv = aeqd_transformers(pts['lon'].mean(), pts['lat'].mean())
    x, y = fwd.transform(pts['lon'].to_numpy(), pts['lat'].to_numpy())
    rng = np.random.default_rng(5)
    bearing, dist, dx, dy = draw_vector(rng, 100_000, 200_000)
    assert 100_000 <= dist <= 200_000 and 0 <= bearing < 360
    assert np.isclose(np.hypot(dx, dy), dist)
    lon, lat = displace(x, y, dx, dy, inv)
    # spacing along each transect (2.5 km) and the shift of each point are kept (< 1%)
    _, _, step = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    same_tr = pts['transectID'].to_numpy()[:-1] == pts['transectID'].to_numpy()[1:]
    assert np.allclose(step[same_tr], 2500, rtol=0.01)
    _, _, shift = GEOD.inv(pts['lon'].to_numpy(), pts['lat'].to_numpy(), lon, lat)
    assert np.allclose(shift, dist, rtol=0.01)
    assert footprint_m(x, y) == pytest.approx(np.hypot(35_000, 10_000), rel=0.01)


def test_boundary_share_and_transect_lines():
    box = shapely.box(0, 0, 1, 1)
    tree = shapely.STRtree(np.array([box.boundary]))
    crossing = shapely.linestrings([[0.5, 0.5], [1.5, 0.5]])
    inside = shapely.linestrings([[0.2, 0.2], [0.4, 0.4]])
    outside = shapely.linestrings([[2, 2], [3, 3]])
    assert boundary_share(np.array([inside, crossing]), tree) == 0.5
    assert boundary_share(np.array([inside, outside]), tree) == 0.0
    assert boundary_share(np.array([crossing]), shapely.STRtree(np.array([], dtype=object))) == 0.0
    # two boundaries crossing one line count once
    tree2 = shapely.STRtree(np.array([box.boundary, shapely.box(1.2, 0, 2, 1).boundary]))
    assert boundary_share(np.array([crossing, inside]), tree2) == 0.5
    pts = toy_transects(n=3)
    ids, lines = transect_lines(pts)
    assert list(ids) == [0, 1, 2] and len(lines) == 3
    _, centre = transect_lines(pts, ends=(-1, 1))
    _, _, full_len = GEOD.inv(*shapely.get_coordinates(lines).reshape(-1, 4).T)
    _, _, centre_len = GEOD.inv(*shapely.get_coordinates(centre).reshape(-1, 4).T)
    assert np.allclose(full_len, 10_000, rtol=0.01) and np.allclose(centre_len, 5_000, rtol=0.01)


class FixedRNG:
    """Stand-in for np.random.Generator: uniform() returns preset values in order (checked in range)."""
    def __init__(self, values):
        self.values = list(values)

    def uniform(self, lo, hi):
        v = self.values.pop(0)
        assert lo <= v <= hi
        return v


def test_candidate_displacements_tolerate_boundaries_away_from_the_centre():
    pts = toy_transects()                       # 8 north-south transects, 5 km apart
    fwd, inv = aeqd_transformers(pts['lon'].mean(), pts['lat'].mean())
    x, y = fwd.transform(pts['lon'].to_numpy(), pts['lat'].to_numpy())
    # where a shift of 100 km due north puts the points
    lon, lat = displace(x, y, 0.0, 100_000.0, inv)
    moved = pts.assign(lon=lon, lat=lat)
    t0 = moved[(moved['transectID'] == 0) & (moved['pointID'] == 0)].iloc[0]
    # a short east-west boundary through transect 0's centre point: 1 of 8 centre zones (12.5%)
    at_centre = shapely.LineString([(t0['lon'] - 0.01, t0['lat']), (t0['lon'] + 0.01, t0['lat'])])
    # a long east-west boundary between points 1 and 2 of every transect: outer parts only
    lat_outer = moved.loc[moved['pointID'].isin([1, 2]), 'lat'].mean()
    outer = shapely.LineString([(moved['lon'].min() - 0.5, lat_outer), (moved['lon'].max() + 0.5, lat_outer)])
    _, full = transect_lines(moved)
    assert shapely.intersects(full, outer).all()          # it crosses every whole transect

    def run(geoms, max_share):
        tree = shapely.STRtree(np.array(geoms))
        return candidate_displacements(pts, tree, FixedRNG([0.0, 100_000.0]), 1, 1, 50_000, 2, max_share)

    cands, attempts, d_used = run([outer], 0.0)
    assert attempts == 1 and len(cands) == 1 and cands[0]['boundary_share'] == 0.0
    assert d_used == 50_000 and np.isclose(cands[0]['dy'], 100_000)
    cands, _, _ = run([outer, at_centre], 0.10)
    assert cands == []                                     # 12.5% > 10%
    cands, _, _ = run([outer, at_centre], 0.20)
    assert len(cands) == 1 and cands[0]['boundary_share'] == pytest.approx(1 / 8)


def test_candidate_displacements_avoid_boundaries_and_overlap():
    pts = toy_transects()
    # A PA 60-400 km to the south: with a 0% limit no centre zone may touch its boundary
    fwd, inv = aeqd_transformers(pts['lon'].mean(), pts['lat'].mean())
    ring_x, ring_y = np.meshgrid(np.linspace(-400e3, 400e3, 81), np.linspace(-400e3, -60e3, 35))
    lon, lat = inv.transform(ring_x.ravel(), ring_y.ravel())
    south = shapely.box(lon.min(), lat.min(), lon.max(), lat.max())
    tree = shapely.STRtree(np.array([south.boundary]))
    rng = np.random.default_rng(3)
    cands, attempts, d_used = candidate_displacements(pts, tree, rng, n_candidates=4, max_attempts=500,
                                                      d_min=50_000, mult=2, max_boundary_share=0.0)
    assert len(cands) == 4 and attempts >= 4
    assert d_used == 50_000                                     # footprint (~36 km) < minimum
    for c in cands:
        assert 50_000 <= c['dist_m'] <= 100_000
        _, centre = transect_lines(c['points'], ends=(-1, 1))
        assert boundary_share(centre, tree) == 0.0 and c['boundary_share'] == 0.0
        assert not shapely.intersects(centre, south.boundary).any()
    # no candidate when every direction is blocked
    everywhere = shapely.box(pts['lon'].min() - 10, pts['lat'].min() - 5, pts['lon'].max() + 10, pts['lat'].max() + 5)
    grid = shapely.STRtree(np.array([shapely.LineString([(everywhere.bounds[0], y), (everywhere.bounds[2], y)])
                                     for y in np.arange(everywhere.bounds[1], everywhere.bounds[3], 0.02)]))
    none, att, _ = candidate_displacements(pts, grid, rng, 2, 30, 50_000, 2, 0.10)
    assert none == [] and att == 30


def test_candidate_checks_apply_the_analysis_set_rules_and_select():
    years = range(2003, 2009)                                   # 6 years
    def rows(pid, cand, n_trnst=10, water=0.0, ext=0.5, inten=0.8):
        n = len(years)
        def col(v):
            return list(v) if isinstance(v, (list, tuple)) else [v] * n
        return pd.DataFrame({'WDPA_PID': pid, 'candidate': cand, 'year': list(years), 'n_trnst': col(n_trnst),
                             'edge_extent': col(ext), 'edge_intensity': col(inten), 'water_extent_pct': col(water)})
    py = pd.concat([
        rows('a', 1),                                           # biome of the moved polygon differs
        rows('a', 2, n_trnst=7),                                # 70% complete: sea
        rows('a', 3, water=80.0),                               # 80% water: lake
        rows('a', 4, inten=[0.8, 0.8, 0.8, 0.8, np.nan, np.nan]),   # 4 years of intensity
        rows('a', 5, n_trnst=[10, 10, 10, 10, 4, 4]),           # median 100% complete: passes
        rows('a', 6),                                           # passes, but 5 comes first
        rows('b', 2),                                           # no biome (e.g. 'Unknown')
    ])                                                          # b1: no data at all (at sea)
    cands = pd.DataFrame({'WDPA_PID': ['a'] * 6 + ['b'] * 2, 'candidate': [1, 2, 3, 4, 5, 6, 1, 2]})
    cb = cands.assign(biome=['Tundra'] + ['Desert'] * 5 + ['Desert', None])
    chk = candidate_checks(cands, py, {'a': 10, 'b': 10}, cb, {'a': 'Desert', 'b': 'Desert'},
                           min_complete=75, max_water=75, min_years=5)
    rule = chk.set_index(['WDPA_PID', 'candidate'])
    assert chk.loc[chk['accepted'], ['WDPA_PID', 'candidate']].values.tolist() == [['a', 5]]
    assert rule.loc[('a', 6), 'passes'] and not rule.loc[('a', 6), 'accepted']
    assert not rule.loc[('a', 1), 'biome_ok'] and rule.loc[('a', 1), ['sea_ok', 'lake_ok', 'years_ok']].all()
    assert rule.loc[('a', 2), 'pct_complete'] == 70 and not rule.loc[('a', 2), 'sea_ok']
    assert rule.loc[('a', 3), 'water_extent_pct'] == 80 and not rule.loc[('a', 3), 'lake_ok']
    assert rule.loc[('a', 4), 'n_years_edge_intensity'] == 4 and not rule.loc[('a', 4), 'years_ok']
    assert rule.loc[('a', 5), 'pct_complete'] == 100            # median, not mean (80)
    b1 = rule.loc[('b', 1)]
    assert b1['n_years'] == 0 and not b1['sea_ok'] and not b1['years_ok'] and b1['lake_ok']
    assert not rule.loc[('b', 2), 'biome_ok'] and not chk.loc[chk['WDPA_PID'] == 'b', 'passes'].any()
    sample = pd.DataFrame({'WDPA_PID': ['a', 'b', 'c', 'd'], 'BIOME_NAME': 'Desert', 'draw': [1, 2, 3, 4]})
    assert select_pas(sample, {'a', 'c', 'd'}, 2) == ['a', 'c']   # c (reserve) replaces b


def test_displaced_polygon_moves_with_its_transects(tmp_path):
    pts = toy_transects()                                       # N-S transects, points at y = -5 ... 5 km
    fwd0, inv0 = aeqd_transformers(20.0, 60.0)
    # a PA south of the transects' middle: points -2 and -1 inside, 0, 1 and 2 outside
    box = shapely.segmentize(shapely.box(-2500, -20000, 37500, -1250), 500)
    xy = shapely.get_coordinates(box)
    pa = shapely.Polygon(np.column_stack(inv0.transform(xy[:, 0], xy[:, 1])))
    inside = shapely.contains_xy(pa, pts['lon'].to_numpy(), pts['lat'].to_numpy())
    assert (inside == pts['pointID'].isin([-2, -1]).to_numpy()).all()
    lon0, lat0 = displacement_centre(pts)
    fwd, inv = aeqd_transformers(lon0, lat0)
    x, y = fwd.transform(pts['lon'].to_numpy(), pts['lat'].to_numpy())
    dx, dy = 70_000.0, 40_000.0
    lon, lat = displace(x, y, dx, dy, inv)
    moved = displace_polygon(pa, lon0, lat0, dx, dy)
    assert (shapely.contains_xy(moved, lon, lat) == inside).all()  # same points inside after the move
    a, b = shapely.get_coordinates(pa), shapely.get_coordinates(moved)
    _, _, shift = GEOD.inv(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    assert np.allclose(shift, np.hypot(dx, dy), rtol=0.01)
    assert abs(GEOD.geometry_area_perimeter(moved)[0]) == pytest.approx(abs(GEOD.geometry_area_perimeter(pa)[0]), rel=0.01)
    import geopandas as gpd
    write_polygon_zip(['p#1'], [moved], tmp_path / 'polys_000.zip')
    back = gpd.read_file(f"zip://{tmp_path / 'polys_000.zip'}")
    assert back['WDPA_PID'].tolist() == ['p#1'] and back.crs.to_epsg() == 4326
    assert back.geometry.iloc[0].normalize().equals_exact(moved.normalize(), 1e-9)   # shapefile ring order


def test_copy_rules_match_the_r_analysis_set():
    import re
    r = (Path(__file__).parent.parent / 'modules' / 'r' / 'config.R').read_text()
    get = lambda name: float(re.search(rf'\b{name}\s*=\s*([0-9.]+)', r).group(1))
    assert config.VALIDATION_MIN_COMPLETE_PCT == get('min_complete_pct')
    assert config.VALIDATION_MAX_WATER_PCT == get('max_water_pct')
    assert config.VALIDATION_MIN_YEARS == get('min_years')


def test_biome_from_num_and_split_key():
    assert list(biome_from_num([1, 6, 13, 14, np.nan, 98])) == [
        'Tropical Forest', 'Boreal Forest', 'Desert', 'Mangrove', None, None]
    pid, cand = split_key(['314990_A#2', '100017#5'])
    assert list(pid) == ['314990_A', '100017'] and list(cand) == [2, 5]


def test_displaced_metrics_reproduce_script_04():
    """With no displacement, the validation metric path gives 04's PA-year metrics exactly."""
    from modules.analysis import read_asset_csvs
    # an asset's export is ordered by point position, so a PA's points span all its parts
    files = sorted(config.GEE_CSV_DIR.glob(f'{config.GEE_EXPORT_PREFIX}000_part_*.csv'))
    if not files or not config.PA_METRICS.exists() or not config.TRANSECT_POINTS.exists():
        pytest.skip("pipeline data not present")
    years = list(range(config.START_YEAR, config.END_YEAR + 1))
    df = read_asset_csvs(files, years)
    # PAs whose points are all in these parts
    import pyarrow.parquet as pq
    tp = pq.read_table(config.TRANSECT_POINTS, columns=['WDPA_PID'],
                       filters=[('WDPA_PID', 'in', df['WDPA_PID'].unique().tolist())]).to_pandas()
    n_all = tp['WDPA_PID'].astype(str).value_counts()
    n_here = df['WDPA_PID'].value_counts()
    whole = sorted(p for p in n_here.index if n_all.get(p) == n_here[p])[:3]
    assert whole, "no PA complete in asset 000"
    got = displaced_metrics(df[df['WDPA_PID'].isin(whole)], years)
    ref = pd.read_csv(config.PA_METRICS, dtype={'WDPA_PID': str},
                      usecols=['WDPA_PID', 'year', 'n_trnst', 'edge_extent', 'edge_intensity', 'water_extent_pct'])
    ref = ref[ref['WDPA_PID'].isin(whole)].sort_values(['WDPA_PID', 'year']).reset_index(drop=True)
    assert len(got) == len(ref)
    np.testing.assert_array_equal(got['n_trnst'].to_numpy(), ref['n_trnst'].to_numpy())
    np.testing.assert_allclose(got['edge_extent'].to_numpy(), ref['edge_extent'].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got['edge_intensity'].to_numpy(), ref['edge_intensity'].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(got['water_extent_pct'].to_numpy(), ref['water_extent_pct'].to_numpy(), rtol=1e-12)

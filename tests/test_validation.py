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
                                transect_lines, reaches_pa_boundary, candidate_displacements,
                                accept_candidates, select_pas, biome_from_num, split_key,
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
    bearing, dist, dx, dy = draw_vector(rng, 100_000, 50_000)
    assert 100_000 <= dist <= 150_000 and 0 <= bearing < 360
    assert np.isclose(np.hypot(dx, dy), dist)
    lon, lat = displace(x, y, dx, dy, inv)
    # spacing along each transect (2.5 km) and the shift of each point are kept (< 1%)
    _, _, step = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    same_tr = pts['transectID'].to_numpy()[:-1] == pts['transectID'].to_numpy()[1:]
    assert np.allclose(step[same_tr], 2500, rtol=0.01)
    _, _, shift = GEOD.inv(pts['lon'].to_numpy(), pts['lat'].to_numpy(), lon, lat)
    assert np.allclose(shift, dist, rtol=0.01)
    assert footprint_m(x, y) == pytest.approx(np.hypot(35_000, 10_000), rel=0.01)


def test_reaches_pa_boundary():
    box = shapely.box(0, 0, 1, 1)
    tree = shapely.STRtree(np.array([box.boundary]))
    crossing = shapely.linestrings([[0.5, 0.5], [1.5, 0.5]])
    inside = shapely.linestrings([[0.2, 0.2], [0.4, 0.4]])
    outside = shapely.linestrings([[2, 2], [3, 3]])
    assert reaches_pa_boundary(np.array([inside, crossing]), tree)
    assert not reaches_pa_boundary(np.array([inside, outside]), tree)
    assert not reaches_pa_boundary(np.array([crossing]), shapely.STRtree(np.array([], dtype=object)))
    ids, lines = transect_lines(toy_transects(n=3))
    assert list(ids) == [0, 1, 2] and len(lines) == 3


def test_candidate_displacements_avoid_boundaries_and_overlap():
    pts = toy_transects()
    # A PA whose boundary is a ring 60-400 km south: only northward shifts can pass
    fwd, inv = aeqd_transformers(pts['lon'].mean(), pts['lat'].mean())
    ring_x, ring_y = np.meshgrid(np.linspace(-400e3, 400e3, 81), np.linspace(-400e3, -60e3, 35))
    lon, lat = inv.transform(ring_x.ravel(), ring_y.ravel())
    south = shapely.box(lon.min(), lat.min(), lon.max(), lat.max())
    tree = shapely.STRtree(np.array([south.boundary]))
    rng = np.random.default_rng(3)
    cands, attempts, d_used = candidate_displacements(pts, tree, rng, n_candidates=4, max_attempts=500,
                                                      d_min=50_000, d_span=50_000)
    assert len(cands) == 4 and attempts >= 4
    assert d_used == 50_000                                     # footprint (~36 km) < minimum
    for c in cands:
        assert 50_000 <= c['dist_m'] <= 100_000
        _, lines = transect_lines(c['points'])
        assert not reaches_pa_boundary(lines, tree)
        assert c['dy'] > 0 or not shapely.intersects(lines, south).any()
    # no candidate when every direction is blocked
    everywhere = shapely.box(pts['lon'].min() - 10, pts['lat'].min() - 5, pts['lon'].max() + 10, pts['lat'].max() + 5)
    grid = shapely.STRtree(np.array([shapely.LineString([(everywhere.bounds[0], y), (everywhere.bounds[2], y)])
                                     for y in np.arange(everywhere.bounds[1], everywhere.bounds[3], 0.02)]))
    none, att, _ = candidate_displacements(pts, grid, rng, 2, 30, 50_000, 50_000)
    assert none == [] and att == 30


def test_accept_and_select():
    years = [2003, 2004]
    def pts(pid, cand, biome, gap=False):
        d = pd.DataFrame({'WDPA_PID': pid, 'candidate': cand, 'transectID': 0, 'pointID': [-2, -1, 0, 1, 2],
                          'biome': biome, '2003': 1.0, '2004': 1.0})
        if gap:
            d.loc[2, '2004'] = np.nan
        return d
    p = pd.concat([pts('a', 1, ['Desert'] * 4 + ['Tundra']),    # one point outside the biome
                   pts('a', 2, 'Desert', gap=True),                # a missing year
                   pts('a', 3, 'Desert'), pts('a', 4, 'Desert'),   # 3 is the first passing
                   pts('b', 1, [None] * 5)])                       # no biome (e.g. sea)
    acc = accept_candidates(p, {'a': 'Desert', 'b': 'Desert'}, years)
    assert acc.loc[acc['accepted'], ['WDPA_PID', 'candidate']].values.tolist() == [['a', 3]]
    assert acc.loc[(acc['WDPA_PID'] == 'a') & (acc['candidate'] == 1), 'share_in_biome'].item() == 0.8
    assert not acc.loc[acc['WDPA_PID'] == 'b', 'passes'].any()
    sample = pd.DataFrame({'WDPA_PID': ['a', 'b', 'c', 'd'], 'BIOME_NAME': 'Desert', 'draw': [1, 2, 3, 4]})
    assert select_pas(sample, {'a', 'c', 'd'}, 2) == ['a', 'c']   # c (reserve) replaces b


def test_biome_from_num_and_split_key():
    assert list(biome_from_num([1, 6, 13, 14, np.nan, 98])) == [
        'Tropical Forest', 'Boreal Forest', 'Desert', 'Mangrove', None, None]
    pid, cand = split_key(['314990_A#2', '100017#5'])
    assert list(pid) == ['314990_A', '100017'] and list(cand) == [2, 5]


def test_displaced_metrics_reproduce_script_04():
    """With no displacement, the validation metric path gives 04's PA-year metrics exactly."""
    from modules.analysis import read_asset_csvs
    # an asset's export is ordered by point position, so a PA's points span all its parts
    files = sorted(config.GEE_CSV_DIR.glob(f'{config.INDEX_NAME}_gradient_{config.RUN_TAG}_000_part_*.csv'))
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
                      usecols=['WDPA_PID', 'year', 'n_trnst', 'edge_extent', 'edge_intensity'])
    ref = ref[ref['WDPA_PID'].isin(whole)].sort_values(['WDPA_PID', 'year']).reset_index(drop=True)
    assert len(got) == len(ref)
    np.testing.assert_array_equal(got['n_trnst'].to_numpy(), ref['n_trnst'].to_numpy())
    np.testing.assert_allclose(got['edge_extent'].to_numpy(), ref['edge_extent'].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got['edge_intensity'].to_numpy(), ref['edge_intensity'].to_numpy(), rtol=1e-12)

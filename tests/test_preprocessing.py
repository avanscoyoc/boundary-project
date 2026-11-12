"""
Unit tests for modules/preprocessing.py (script 02 geometry and transect logic).

Run from the repo root:  python -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import shapely
from pyproj import Geod
from shapely.geometry import Polygon, box

from modules.preprocessing import (
    evenspace, extract_coords, make_pa_transects, resolve_overlaps,
    merge_same_name, add_shape_metrics, flag_other_pa,
)

MOLL = 'ESRI:54009'
GEOD = Geod(ellps='WGS84')


def lonlat_to_moll(geom):
    return gpd.GeoSeries([geom], crs='EPSG:4326').to_crs(MOLL).iloc[0]


def pas_frame(rows):
    """rows: list of (pid, geometry, name, iso3, year) in Mollweide meters."""
    return gpd.GeoDataFrame(
        {'WDPA_PID': [r[0] for r in rows], 'ORIG_NAME': [r[2] for r in rows],
         'ISO3': [r[3] for r in rows], 'STATUS_YR': [r[4] for r in rows]},
        geometry=[r[1] for r in rows], crs=MOLL,
    )


# ---------------------------------------------------------------- evenspace

def test_evenspace_points_lie_on_square_boundary():
    # 4 vertices only: an off-by-one error in the segment lookup would put points up to
    # 10 km off the boundary here
    sq = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    pts = evenspace(extract_coords(sq), 500)
    xy = shapely.points(pts['x'], pts['y'])
    assert shapely.distance(sq.exterior, xy).max() < 1e-6
    ring = shapely.geometry.LineString(extract_coords(sq))
    np.testing.assert_allclose(np.diff(shapely.line_locate_point(ring, xy)), 500, atol=1e-6)


# ---------------------------------------------------------------- AEQD distances

def test_transects_use_true_distances_at_high_latitude():
    # ~80 km box at 70°N, where Mollweide distances are badly distorted
    geom = lonlat_to_moll(box(19.0, 69.6, 21.0, 70.4))
    res = make_pa_transects(geom, MOLL, 500, 2500, 2, 5500)
    assert res['status'] == 'ok' and len(res['df']) > 0
    df = res['df']

    # spacing along each transect: 2,500 m (geodesic) within 1%
    for _, t in df.groupby('transectID'):
        t = t.sort_values('pointID')
        _, _, d = GEOD.inv(t['lon'].values[:-1], t['lat'].values[:-1], t['lon'].values[1:], t['lat'].values[1:])
        np.testing.assert_allclose(d, 2500, rtol=0.01)

    # boundary spacing: consecutive boundary points 500 m apart (away from corners)
    b = df[df['pointID'] == 0].sort_values('transectID')
    consecutive = np.diff(b['transectID'].values) == 1
    _, _, d = GEOD.inv(b['lon'].values[:-1], b['lat'].values[:-1], b['lon'].values[1:], b['lat'].values[1:])
    assert np.median(d[consecutive]) == pytest.approx(500, rel=0.01)


# ---------------------------------------------------------------- whole-line check

def u_shape():
    # 60 x 40 km block with a 4 km wide, 30 km deep notch (Mollweide meters near 0,0)
    return Polygon([(0, 0), (60000, 0), (60000, 40000), (32000, 40000), (32000, 10000),
                    (28000, 10000), (28000, 40000), (0, 40000)])


def test_whole_line_check_on_concave_polygon():
    poly = u_shape()
    res = make_pa_transects(poly, MOLL, 500, 2500, 2, 5500)
    assert res['n_bad_outer'] > 0          # notch walls: outer half re-enters the PA
    df = res['df']
    poly_ll = gpd.GeoSeries([poly], crs=MOLL).to_crs('EPSG:4326').iloc[0]
    pts = shapely.points(df['lon'], df['lat'])
    inside = shapely.contains(poly_ll, pts)
    assert inside[df['pointID'].values < 0].all()      # inner points inside
    assert not inside[df['pointID'].values > 0].any()  # outer points outside


def test_ring_orientation_does_not_change_transects():
    ccw = u_shape()
    cw = Polygon(list(ccw.exterior.coords)[::-1])
    assert ccw.exterior.is_ccw and not cw.exterior.is_ccw
    a = make_pa_transects(ccw, MOLL, 500, 2500, 2, 5500)['df']
    b = make_pa_transects(cw, MOLL, 500, 2500, 2, 5500)['df']
    assert len(a) > 0
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_evenness_columns():
    full = make_pa_transects(box(0, 0, 60000, 60000), MOLL, 500, 2500, 2, 5500)
    assert len(full['df']) // 5 == full['n_generated']        # convex: every transect kept
    assert full['max_gap_pct'] == 0
    notch = make_pa_transects(u_shape(), MOLL, 500, 2500, 2, 5500)
    assert notch['n_generated'] > len(notch['df']) // 5
    assert notch['max_gap_pct'] > 0                             # notch walls leave a gap


def test_narrow_pa_is_skipped_by_inner_buffer():
    strip = box(0, 0, 100000, 9000)   # 9 km wide: nothing is 5.5 km from the edge
    assert make_pa_transects(strip, MOLL, 500, 2500, 2, 5500)['status'] == 'empty_inner_buffer'


# ---------------------------------------------------------------- overlaps

def test_resolve_overlaps_duplicates_and_nested():
    gdf = pas_frame([
        ('A', box(0, 0, 10000, 10000), 'a', 'X', 2000),
        ('B', box(100, 100, 10100, 10100), 'b', 'X', 1990),     # duplicate of A, older
        ('C', box(2000, 2000, 3000, 3000), 'c', 'X', 1950),     # nested in A/B (older!)
        ('D', box(50000, 0, 60000, 10000), 'd', 'X', 2010),     # separate
        ('E', box(5000, 0, 15000, 10000), 'e', 'X', 2010),      # ~50% overlap: kept
    ])
    kept, removed, counts = resolve_overlaps(gdf, 90)
    assert sorted(kept['WDPA_PID']) == ['B', 'D', 'E']
    r = removed.set_index('WDPA_PID')
    assert r.loc['A', 'reason'] == 'duplicate' and r.loc['A', 'replaced_by'] == 'B'
    assert r.loc['C', 'reason'] == 'nested'          # larger PA kept, even though C is older
    assert counts == {'n_in': 5, 'n_invalid_fixed': 0, 'removed_duplicate': 1,
                      'removed_nested': 1, 'n_out': 3}


def test_resolve_overlaps_duplicates_are_transitive():
    gdf = pas_frame([
        ('A', box(0, 0, 10000, 10000), 'a', 'X', 2000),
        ('B', box(400, 0, 10400, 10000), 'b', 'X', 2001),       # 96% with A
        ('C', box(800, 0, 10800, 10000), 'c', 'X', 1980),       # 96% with B, 92% with A
    ])
    kept, removed, _ = resolve_overlaps(gdf, 90)
    assert list(kept['WDPA_PID']) == ['C']


# ---------------------------------------------------------------- same name

def test_merge_same_name_only_touching_same_country():
    gdf = pas_frame([
        ('P1', box(0, 0, 1000, 1000), 'X', 'AAA', 2000),
        ('P2', box(1000, 0, 2000, 1000), 'X', 'AAA', 1990),        # touches P1, older
        ('P3', box(50000, 0, 51000, 1000), 'X', 'AAA', 1980),      # same name, far away
        ('P4', box(2000, 0, 3000, 1000), 'X', 'BBB', 1970),        # touches P2, other country
    ])
    kept, removed, counts = merge_same_name(gdf)
    k = kept.set_index('WDPA_PID')
    assert sorted(k.index) == ['P2', 'P3', 'P4']
    assert k.loc['P2', 'MERGED_PID'] == 'P1'
    assert k.loc['P2', 'geometry'].geom_type == 'Polygon'
    assert abs(k.loc['P2', 'geometry'].area - 2_000_000) < 1e-6
    assert list(removed['WDPA_PID']) == ['P1'] and list(removed['replaced_by']) == ['P2']
    assert counts['groups_merged'] == 1 and counts['pas_merged_away'] == 1


def test_merge_takes_attributes_from_largest_and_oldest_year():
    gdf = pas_frame([
        ('NP', box(0, 0, 10000, 10000), 'Denali', 'USA', 0),         # largest, unknown year
        ('SP', box(10000, 0, 11000, 1000), 'Denali', 'USA', 1970),   # small, oldest known year
    ])
    gdf['DESIG_ENG'] = ['National Park', 'State Park']
    gdf['IUCN_CAT'] = ['II', 'V']
    kept, _, _ = merge_same_name(gdf)
    row = kept.iloc[0]
    assert row['WDPA_PID'] == 'SP'                     # ID unchanged (matches uploaded points)
    assert row['STATUS_YR'] == 1970                    # oldest non-zero year
    assert (row['DESIG_ENG'], row['IUCN_CAT'], row['ATTR_PID']) == ('National Park', 'II', 'NP')


# ---------------------------------------------------------------- shape metric

def test_shape_ratio_is_size_neutral():
    # circles drawn in lon/lat at the equator (1° lon ≈ 1° lat there)
    small = lonlat_to_moll(shapely.Point(0, 0).buffer(0.05, quad_segs=64))
    large = lonlat_to_moll(shapely.Point(0, 0).buffer(0.5, quad_segs=64))
    gdf = pas_frame([('s', small, 's', 'X', 1), ('l', large, 'l', 'X', 1)])
    ratio = add_shape_metrics(gdf)['SHAPE_RATIO'].to_numpy()
    np.testing.assert_allclose(ratio, 2 * np.sqrt(np.pi), rtol=0.01)   # circle = 3.545


# ---------------------------------------------------------------- other-PA flag

def test_flag_other_pa_ignores_own_and_merged_pas():
    other = box(10, 0, 11, 1)
    merged_neighbor = box(20, 0, 21, 1)
    tree = shapely.STRtree([other, merged_neighbor])
    tree_pids = np.array(['OTHER', 'MERGED'])
    points = pd.DataFrame({
        'WDPA_PID': ['P'] * 10, 'transectID': [1] * 5 + [2] * 5,
        'pointID': [-2, -1, 0, 1, 2] * 2,
        'lon': [0, 0, 0, 10.5, 12, 0, 0, 0, 20.5, 22],   # t1 outer pt in OTHER, t2 in MERGED
        'lat': [0.5] * 10,
    })
    flags = flag_other_pa(points, tree, tree_pids, {'P': {'P', 'MERGED'}})
    assert list(flags[:5]) == [1] * 5 and list(flags[5:]) == [0] * 5


# ---------------------------------------------------------------- coverage filter

def test_generate_transects_removes_pa_with_coverage_gap(tmp_path):
    from modules.preprocessing import generate_transects
    pas = pas_frame([('U', u_shape(), 'u', 'X', 2000), ('SQ', box(100000, 0, 160000, 60000), 'sq', 'X', 2000)])
    kwargs = dict(src_crs=MOLL, sample_dist=500, transect_unit=2500, npts=2, buffer_dist=5500,
                  chunk_size=400, upload_dir=tmp_path, upload_prefix='t_', points_path=tmp_path / 'p.parquet',
                  tree=shapely.STRtree([box(1e7, 1e7, 1e7 + 1, 1e7 + 1)]), tree_pids=np.array(['far']),
                  own_pids={})
    strict = generate_transects(pas, max_gap_pct_limit=0, **kwargs)
    assert strict['status'] == {'U': 'coverage_gap', 'SQ': 'ok'}       # notch leaves a gap; square doesn't
    assert set(pd.read_parquet(tmp_path / 'p.parquet')['WDPA_PID']) == {'SQ'}
    assert strict['totals']['in_pas_removed_for_coverage_gap'] == strict['n_transects']['U'] > 0
    loose = generate_transects(pas, max_gap_pct_limit=None, **kwargs)
    assert loose['status'] == {'U': 'ok', 'SQ': 'ok'}


# ---------------------------------------------------------------- Earth Engine retries (script 03)

def test_retry_recovers_from_transient_errors(monkeypatch):
    import modules.remotesensing as rs
    monkeypatch.setattr(rs.time, 'sleep', lambda s: None)
    calls = {'n': 0}

    def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionError('network down')
        return 'COMPLETED'

    assert rs._retry(flaky, 'test') == 'COMPLETED' and calls['n'] == 3

    def always_fails():
        raise ConnectionError('still down')
    import pytest
    with pytest.raises(ConnectionError):
        rs._retry(always_fails, 'test', attempts=3)

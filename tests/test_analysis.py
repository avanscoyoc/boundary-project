"""Unit tests for modules/analysis.py (script 04 metrics and trends)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from modules.analysis import bh_adjust, add_trends, cohens_d, _grouped_cohens_d, PT_COLS


def test_bh_adjust_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0, 1, 200), rng.uniform(0, 0.001, 20), [np.nan, np.nan]]
    ours = bh_adjust(p)
    ref = multipletests(p[~np.isnan(p)], method='fdr_bh')[1]
    np.testing.assert_allclose(ours[~np.isnan(p)], ref)
    assert np.isnan(ours[-2:]).all()


def test_add_trends_classes_and_min_years():
    years = np.arange(2003, 2026)
    rows = []
    for pid, values in {'UP': 0.3 + 0.01 * (years - 2003),           # clear increase
                        'DOWN': 0.8 - 0.01 * (years - 2003),         # clear decrease
                        'FLAT': 0.5 + 0.001 * np.sin(years),         # no trend
                        'SHORT': [np.nan] * 19 + [0.1, 0.2, 0.3, 0.4]}.items():   # 4 years only
        for y, v in zip(years, values):
            rows.append({'WDPA_PID': pid, 'year': y, 'edge_extent': v})
    df = pd.DataFrame(rows)
    for col in ['edge_intensity', 'edge_extent_nowater', 'edge_intensity_nowater', 'edge_extent_nootherpa',
                'edge_intensity_nootherpa', 'edge_extent_noflags', 'edge_intensity_noflags']:
        df[col] = df['edge_extent']
    out = add_trends(df).groupby('WDPA_PID').first()
    assert out.loc['UP', 'trend_extent'] == 'increase'
    assert out.loc['DOWN', 'trend_extent'] == 'decrease'
    assert out.loc['FLAT', 'trend_extent'] == 'no_trend'
    assert pd.isna(out.loc['SHORT', 'trend_extent']) and np.isnan(out.loc['SHORT', 'p_adj_extent'])
    assert (out['p_adj_extent'].dropna() >= out['p_value_extent'].dropna()).all()


def test_grouped_cohens_d_matches_reference():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(60, 5)), columns=PT_COLS)
    df['pt_0'] += 1.0
    df['WDPA_PID'], df['year'] = 'A', 2010
    d = _grouped_cohens_d(df, ['WDPA_PID', 'year']).iloc[0]
    for name, col in {'D02': 'pt_2', 'D01': 'pt_1', 'D0m1': 'pt_m1', 'D0m2': 'pt_m2'}.items():
        assert np.isclose(d[name], cohens_d(df['pt_0'], df[col]))



def test_missing_transects():
    """Transects with a missing value in any year, as build_transect_years drops them (script 04 → 05)."""
    from modules.analysis import missing_transects
    years = [2003, 2004, 2005]

    def pts(pid, tid, missing=(), water=(), n=5):
        rows = []
        for p in (-2, -1, 0, 1, 2)[:n]:
            rows.append({'WDPA_PID': pid, 'transectID': tid, 'pointID': p, 'max_extent': int(p in water),
                         **{str(y): (np.nan if (p, y) in missing else 1.0) for y in years}})
        return rows
    df = pd.DataFrame(pts('a', 1)                                                     # complete
                      + pts('a', 2, missing={(2, y) for y in years}, water={1, 2})     # +5 km at water, every year
                      + pts('b', 3, missing={(0, 2004)})                               # boundary point, one year
                      + pts('b', 4, missing={(-2, 2005), (1, 2005)})                   # both sides, one year
                      + pts('b', 5, missing={(0, 2003)}, n=4))                         # 4 points: skipped
    out = missing_transects(df.sample(frac=1, random_state=1), years)
    assert out[['WDPA_PID', 'transectID']].values.tolist() == [['a', 2], ['b', 3], ['b', 4]]
    assert out['n_missing_years'].tolist() == [3, 1, 1]
    assert out['water_any'].tolist() == [1, 0, 0]
    assert out['n_missing_point_years'].tolist() == [3, 1, 2]
    assert out['n_missing_point_years_water'].tolist() == [3, 0, 0]
    assert out[[f'miss_{c}' for c in PT_COLS]].values.tolist() == [[0, 0, 0, 0, 3], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0]]

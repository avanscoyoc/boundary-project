"""Unit tests for modules/remotesensing.py (no Earth Engine calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


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
    with pytest.raises(ConnectionError):
        rs._retry(always_fails, 'test', attempts=3)

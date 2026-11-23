"""Golden fingerprints (scripts/fingerprint.py): what the live pipeline builds must not change during a refactor.

Needs the recordings in ML/subjects (marker ``data``); skipped elsewhere. Run with ``pytest -m data``.
"""
import pickle

import pytest

from scripts import fingerprint as FP

pytestmark = pytest.mark.data


@pytest.fixture
def no_cache(monkeypatch):
    """Any read of the processed-data cache fails the test: the check must rebuild from the recordings."""
    def refuse(*a, **k):
        raise AssertionError('the golden check read the processed-data cache')
    monkeypatch.setattr(pickle, 'load', refuse)


def test_bundles_bypass_cache(no_cache):
    assert FP.check('bundles') == []


def test_prepare_matches():
    assert FP.check('prepare') == []


def test_outputs_match():
    assert FP.check('outputs') == []


def test_confirm_provenance_still_passes():
    from EXPERIMENTS.analysis import confirm as CF
    from EXPERIMENTS.protocol import protocol as PT
    protocol = PT.load()
    runs, _, _ = CF.final_runs(protocol=protocol)
    assert CF.provenance_check(runs, protocol)['n_runs'] == len(runs)

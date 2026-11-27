"""ML.preprocessing.{trials, passive_map}: values computed with the old dataset_builder / active_torque (22 Nov 2026)."""
import numpy as np
import pandas as pd
import pytest

from ML.preprocessing import passive_map as PM, trials as TR


def test_lookup_extrapolates_like_before():
    e = [(-0.5, 8.0), (0.0, 1.0), (0.2, -3.0)]
    got = [PM.lookup_passive_torque(e, x) for x in (-0.7, -0.25, 0.1, 0.4)]
    assert got == pytest.approx([10.799999999999999, 4.5, -1.0, -7.0], abs=0, rel=0)
    assert PM.lookup_passive_torque([], 0.3) == 0.0
    assert PM.lookup_passive_torque([(0.1, 2.5)], -1.0) == 2.5


def test_classify_trials_by_comment():
    trials = []
    for c in ('MVC p1', 'passive retest p2', 'p3 test', 'Indx 4 retest'):
        df = pd.DataFrame({TR.POSITION_COL: [0.0], TR.TORQUE_COL: [0.0]})
        df.attrs['comment'] = c
        trials.append(df)
    mvc, passive, active = TR.classify_trials(trials)
    assert [len(mvc), len(passive), len(active)] == [1, 1, 2]
    assert [d.attrs['session'] for d in trials] == [None, 'retest', 'test', 'retest']


def test_split_at_device_step():
    pos = np.r_[np.zeros(3000), np.full(3000, 0.2)]
    df = pd.DataFrame({TR.POSITION_COL: pos, TR.TORQUE_COL: np.zeros(6000)})
    segs = TR.split_passive_segments(df)
    assert [len(s) for s in segs] == [3000, 2000]

"""The 6-day compute budget (deviation of 18 Nov): smaller search, 3 final seeds per cell winner, the 2 s check only,
and the run-time check that adds the pre-registered fallback steps in order."""
import json
import os

import pytest

from EXPERIMENTS.protocol import protocol as PT
from EXPERIMENTS.tuning import budget as BG, make_specs as MS, space as SPC

P = PT.load()
B = P['budget']
H = 3600.0


def test_protocol_budget():
    assert P['n_draws'] == 6 and P['fine']['n'] == 4 and P['final_seeds']['cell'] == [10, 11, 12]
    assert B['days'] == 6 and B['workers'] == 6
    assert B['fallback'] == ['drop_o75', 'skip_2s_check', 'fine_2'] and B['tune_gated'] is False
    assert B['screens'] == ['w200_full']


def test_fine_2_is_the_first_two_fine_draws():
    c = dict(SPC.ANCHOR)
    four, two = SPC.fine_configs('w20_full', c), SPC.fine_configs('w20_full', c, n=2)
    assert sorted(four) == ['f0', 'f1', 'f2', 'f3'] and two == {k: four[k] for k in ('f0', 'f1')}


def _r1_runs(rd, seconds, cells=None, start=1_000_000.0):
    """Fake finished R1 runs: one seed0.json per config and person, each taking ``seconds`` x the cell multiplier."""
    t = start
    for c in cells or MS.active_cells(rd):
        for name in SPC.configs(c):
            for p in P['people']:
                d = rd / f'TS2_R1_{c}_{name}' / p / 'S1toS2'
                d.mkdir(parents=True, exist_ok=True)
                s = seconds * BG.multiplier(c)
                f = d / 'seed0.json'
                f.write_text(json.dumps({'prep_s': 0.0, 'train_s': s}))
                t += s / B['workers']
                os.utime(f, (t, t))
    return t


def test_fast_runs_need_no_fallback(tmp_path):
    rd = tmp_path / 'results_v2'
    now = _r1_runs(rd, 60.0, cells=['w20_full', 'w20_o50'])
    out = BG.check(rd, now=now)
    assert out['applied'] == [] and out['fits'] is True
    assert json.loads((rd / 'budget.json').read_text())['applied'] == []


def test_slow_runs_add_steps_in_order(tmp_path):
    rd = tmp_path / 'results_v2'
    now = _r1_runs(rd, 1100.0, cells=['w20_full', 'w20_o50'])
    out = BG.check(rd, now=now)
    assert out['applied'] and out['applied'] == B['fallback'][:len(out['applied'])]
    slow = BG.check(rd, now=now + 400 * H)      # far behind: every step, still over
    assert slow['applied'] == B['fallback'] and slow['fits'] is False


def test_steps_are_never_removed(tmp_path):
    rd = tmp_path / 'results_v2'
    rd.mkdir()
    (rd / 'budget.json').write_text(json.dumps({'applied': ['drop_o75']}))
    now = _r1_runs(rd, 60.0, cells=['w20_full'])
    assert BG.check(rd, now=now)['applied'] == ['drop_o75']


def test_check_reads_run_times_only(tmp_path, monkeypatch):
    """The budget check opens seedN.json only: never validation or session-2 predictions."""
    rd = tmp_path / 'results_v2'
    now = _r1_runs(rd, 60.0, cells=['w20_full'])
    import numpy as np
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('budget check opened an npz'))
    BG.check(rd, now=now)


def test_active_cells_budget(tmp_path):
    rd = tmp_path / 'results_v2'
    rd.mkdir()
    cells = MS.active_cells(rd)
    assert len(cells) == 15 and all(c.endswith('_o75') for c in cells[-3:])     # o75 R1 runs queue last
    (rd / 'gate_w200_full.json').write_text(json.dumps({'continues': True}))
    assert len(MS.active_cells(rd)) == 15                                          # passed, but not tuned (budget)
    (rd / 'budget.json').write_text(json.dumps({'applied': ['drop_o75']}))
    assert not any(c.endswith('_o75') for c in MS.active_cells(rd)) and len(MS.active_cells(rd)) == 12


def test_fine_round_size_follows_budget(tmp_path):
    rd = tmp_path / 'results_v2'
    rd.mkdir()
    assert BG.fine_n(rd) == 4
    (rd / 'budget.json').write_text(json.dumps({'applied': ['drop_o75', 'skip_2s_check', 'fine_2']}))
    assert BG.fine_n(rd) == 2

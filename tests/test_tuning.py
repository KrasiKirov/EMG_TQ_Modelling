import json

import numpy as np
import pytest

from EXPERIMENTS.protocol import protocol as PT
from EXPERIMENTS.tuning import halving as HV, make_specs as MS, space as SPC

P6 = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']


def test_draws_reproducible():
    assert SPC.draws(8, 20261118) == SPC.draws(8, 20261118)
    assert SPC.draws(8, 20261118) != SPC.draws(8, 1)


def test_draws_within_space():
    for d in SPC.draws(50, 3):
        assert 3e-4 <= d['lr'] <= 3e-2
        assert d['units'] in (8, 16, 32, 64) and d['layers'] in (1, 2)
        assert 0.0 <= d['dropout'] <= 0.3 and d['batch'] in (32, 64, 128)


def test_same_draws_every_cell():
    cfgs = {c: SPC.configs(c) for c in ('w20_full', 'w100_o0', 'carry_5s')}
    keys = [sorted(v) for v in cfgs.values()]
    assert keys[0] == keys[1] == keys[2] and 'anchor' in keys[0] and len(keys[0]) == 9
    for name in keys[0]:
        a, b, c = (cfgs[k][name] for k in cfgs)
        assert a == b
        assert {k: v for k, v in a.items() if k != 'batch'} == c          # carried state: batch ignored


def test_overlap_cells_equal_updates():
    cells = SPC.cells()
    for w in (20, 50, 100, 200, 500, 1000):
        group = [cells[f'w{w}_{o}'] for o in ('full', 'o75', 'o50', 'o0')]
        assert all(g['epoch_windows'] == 'full' for g in group)
        assert [g['train_stride'] for g in group][0] == 1 and group[-1]['train_stride'] == w


def test_specs_valid(tmp_path):
    from EXPERIMENTS.pipeline import spec as SPEC
    ids = MS.write('R1', 'w20_o75', SPC.configs('w20_o75'), configs_dir=tmp_path)
    assert len(ids) == 9 and all(i.startswith('TS2_R1_w20_o75_') for i in ids)
    for i in ids:
        s = SPEC.load(i, tmp_path)
        assert s.changes['train_protocol'] == 'study60' and s.changes['train_stride'] == 5
        assert s.changes['shuffle'] is False and s.changes['clipnorm'] == 1.0


def _table(rng, configs, base=0.2):
    return {c: {p: float(base * np.exp(rng.normal(0, 0.1))) for p in P6} for c in configs}


def test_score_and_keep():
    t = {'a': dict.fromkeys(P6, 0.2), 'b': dict.fromkeys(P6, 0.1), 'c': dict.fromkeys(P6, 0.3)}
    assert np.isclose(HV.score(t)['b'], np.log(0.1))
    assert HV.keep(t, 2) == ['b', 'a']


def test_keep_ignores_test_files(tmp_path):
    from EXPERIMENTS.tuning import select as SEL
    rd = _fake_results(tmp_path, ['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'])
    t1 = SEL.load_val(['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'], seeds=[0], people=P6, results_dir=rd)[0]
    for f in rd.rglob('seed0.npz'):
        f.unlink()
    t2 = SEL.load_val(['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'], seeds=[0], people=P6, results_dir=rd)[0]
    assert HV.keep(t1['full'], 1) == HV.keep(t2['full'], 1)


def _fake_results(tmp, ids, seeds=(0,), err=None, n_params=None):
    rd = tmp / 'results_v2'
    rng = np.random.default_rng(0)
    for k, e in enumerate(ids):
        for p in P6:
            for s in seeds:
                d = rd / e / p / 'S1toS2'
                d.mkdir(parents=True, exist_ok=True)
                y = rng.normal(0, 5, 400)
                scale = (err or {}).get(e, 0.1 * (k + 1))
                yp = y + rng.normal(0, scale, 400)
                np.savez(d / f'seed{s}_val.npz', y=y, y_pred=yp, trial=np.repeat([0, 2], 200), last=np.arange(400),
                         common=np.arange(400) % 2 == 0, session=np.zeros(400, np.int8))
                np.savez(d / f'seed{s}.npz', y_own_target=y, y_pred=y + 99)
                (d / f'seed{s}.json').write_text(json.dumps({'train_info': {'n_params': (n_params or {}).get(e, 100 + k)}}))
    return rd


def test_gate_stops_10pct_4of6():
    short = {'s': dict.fromkeys(P6, 0.2)}
    screen = {'x': {p: 0.2 * (1.15 if i < 4 else 1.0) for i, p in enumerate(P6)}}
    cont, ratios = HV.gate(screen, [short])
    assert cont is False and len(ratios[0]) == 6


def test_gate_passes_within_margin():
    short = {'s': dict.fromkeys(P6, 0.2)}
    assert HV.gate({'x': dict.fromkeys(P6, 0.2 * 1.08)}, [short])[0] is True
    # 15 % worse for only 3 people: continues
    assert HV.gate({'x': {p: 0.2 * (1.15 if i < 3 else 1.0) for i, p in enumerate(P6)}}, [short])[0] is True


def test_gate_against_any_shorter_length():
    a, b = {'s': dict.fromkeys(P6, 0.3)}, {'s': dict.fromkeys(P6, 0.2)}
    assert HV.gate({'x': dict.fromkeys(P6, 0.25)}, [a, b])[0] is False


def test_stopped_length_stops_longer():
    assert HV.gated_lengths_to_run({'w200_full': False}) == []
    assert HV.gated_lengths_to_run({'w200_full': True, 'w500_full': False}) == ['w200_full']
    assert HV.gated_lengths_to_run({'w200_full': True}) == ['w200_full', 'w500_full']
    assert HV.gated_lengths_to_run({}) == ['w200_full']

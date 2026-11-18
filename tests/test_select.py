import json

import numpy as np
import pytest

from EXPERIMENTS.protocol import lock as LK
from EXPERIMENTS.tuning import select as SEL
from tests.test_tuning import P6, _fake_results

IDS = ['TS2_R3_w20_full_anchor', 'TS2_R3_w20_full_d0', 'TS2_R3_w20_full_d1']


def test_select_without_test_files(tmp_path):
    rd = _fake_results(tmp_path, IDS, seeds=(0, 1))
    for f in rd.rglob('seed?.npz'):
        f.unlink()
    tables, cov = SEL.load_val(IDS, seeds=[0, 1], people=P6, results_dir=rd)
    assert SEL.cut(tables['full'], 2) and SEL.winner(tables['full'], SEL.n_params(IDS, P6, rd)) in IDS
    assert all(cov[i][p] == 200 for i in IDS for p in P6)


def test_choose_ignores_test_files(tmp_path):
    rd = _fake_results(tmp_path, IDS, seeds=(0, 1))
    w1 = SEL.winner(SEL.load_val(IDS, [0, 1], P6, rd)[0]['full'], SEL.n_params(IDS, P6, rd))
    for f in rd.rglob('seed?.npz'):
        np.savez(f, y_own_target=np.zeros(3), y_pred=np.ones(3))
    assert SEL.winner(SEL.load_val(IDS, [0, 1], P6, rd)[0]['full'], SEL.n_params(IDS, P6, rd)) == w1


def test_test_locked_before_a0(tmp_path):
    rd = _fake_results(tmp_path, IDS[:1])
    f = next(rd.rglob('seed0.npz'))
    assert not LK.test_open(rd)
    with pytest.raises(LK.TestLocked):
        LK.load_test(f, rd)
    (rd / 'choice.json').write_text('{}')
    assert LK.test_open(rd) and 'y_pred' in LK.load_test(f, rd)


def test_missing_run_raises(tmp_path):
    rd = _fake_results(tmp_path, IDS[:2])
    with pytest.raises(FileNotFoundError):
        SEL.load_val(IDS, [0], P6, rd)


def _tab(best, diffs):
    return {'best': dict.fromkeys(P6, 0.2),
            'cand': {p: 0.2 * np.exp(d) for p, d in zip(P6, diffs)}} if best else None


def test_paired_se():
    assert SEL.within_one_se(_tab(1, [0.01] * 6)) == ['best']
    assert sorted(SEL.within_one_se(_tab(1, [0.05, -0.03, 0.04, -0.02, 0.03, -0.01]))) == ['best', 'cand']


def test_one_se_fewest_params():
    t = _tab(1, [0.05, -0.03, 0.04, -0.02, 0.03, -0.01])
    assert SEL.winner(t, {'best': 500, 'cand': 100}) == 'cand'
    assert SEL.winner(t, {'best': 100, 'cand': 100}) == 'best'          # tie in size: lower score


def test_a0_on_common_targets(tmp_path):
    full = {'a': dict.fromkeys(P6, 0.1), 'b': dict.fromkeys(P6, 0.2)}
    common = {'a': dict.fromkeys(P6, 0.3), 'b': dict.fromkeys(P6, 0.2)}
    out = tmp_path / 'choice.json'
    a0 = SEL.choose_a0({'full': full, 'common': common}, {'a': 10, 'b': 10}, out=out, cell_winners={'w': 'b'})
    assert a0 == 'b' and json.loads(out.read_text())['A0'] == 'b'


def test_cli_rounds_and_gate(tmp_path, monkeypatch, capsys):
    """coarse (R1) -> fine draws around the best coarse config (R2) -> best 2 of both with seed 1 (R3) -> gate."""
    from EXPERIMENTS.pipeline import spec as SPEC
    from EXPERIMENTS.tuning import make_specs as MS, space as SPC
    cfg = tmp_path / 'configs'
    cfg.mkdir()
    monkeypatch.setattr(SPEC, 'CONFIGS', cfg)
    r1 = [i for c in ('w20_full', 'w50_full', 'w100_full', 'w200_full') for i in MS.write('R1', c, SPC.configs(c), cfg)]
    rd = _fake_results(tmp_path, r1)
    monkeypatch.setattr(SEL, 'RESULTS_V2', rd)
    SEL.main(['--round', 'R1', '--cell', 'w20_full'])
    assert json.loads(capsys.readouterr().out)['best'] == 'TS2_R1_w20_full_anchor'   # fake error grows with position
    r2 = MS.for_round('R2', 'w20_full')
    assert r2 == [f'TS2_R2_w20_full_f{i}' for i in range(8)]
    anchor = SPC.ANCHOR
    for i in r2:
        ch = SPEC.load(i, cfg).changes
        assert ch['layers'] == anchor['layers'] and ch['units'] in (8, 16) and ch['train_protocol'] == 'study400'
    _fake_results(tmp_path, r2, seeds=(0, 1), err={i: 0.05 for i in r2[:2]})
    cands = SEL.r3_ids('w20_full', P6)
    assert sorted(cands) == sorted(r2[:2])
    assert MS.for_round('R3', 'w20_full') == cands
    SEL.main(['--round', 'R3', '--cell', 'w20_full'])
    assert json.loads(capsys.readouterr().out)['winner'] in cands
    SEL.main(['--gate', 'w200_full'])
    g = json.loads(capsys.readouterr().out)
    assert g['continues'] is False and list(g['ratios']) == ['w20_full', 'w50_full', 'w100_full']   # w200 last: worst


def test_choose_ridge_alpha():
    assert SEL.choose_ridge_alpha({0.1: 0.3, 1.0: 0.2, 10.0: 0.25}) == 1.0

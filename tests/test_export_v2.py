import json

import numpy as np
import pytest

from EXPERIMENTS.analysis import confirm as CF, study_v2_export as EX
from EXPERIMENTS.pipeline import spec as SPEC
from EXPERIMENTS.protocol import protocol as PT

PEOPLE = PT.load()['people']


def _runs(rd, ids, seeds=(0,), n=8000):
    rng = np.random.default_rng(0)
    for e in ids:
        for p in PEOPLE:
            for s in seeds:
                d = rd / e / p / 'S1toS2'
                d.mkdir(parents=True, exist_ok=True)
                y = rng.normal(0, 5, n)
                trial = np.repeat(np.arange(8), n // 8)
                t = np.tile(1.0 + np.arange(n // 8) / 100, 8)
                np.savez(d / f'seed{s}.npz', y_own_target=y, y_pred=y + 0.3, trial=trial, t=t, angle=trial / 10)
                np.savez(d / f'seed{s}_val.npz', y=y[:400], y_pred=y[:400] + 0.1, common=np.ones(400, bool))
                (d / f'seed{s}.json').write_text(json.dumps({'config': {'units': 8}, 'config_hash': 'a' * 16,
                                                             'provenance': {'git_commit': 'c', 'host': 'box.example.home'},
                                                             'train_info': {'n_params': 10}}))


def test_export_before_a0_has_no_session2(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'])
    out = EX.build(rd)
    assert out['confirm'] is None
    assert all('positions' not in r for r in out['runs'].values())
    assert out['tuning']['w20_full']['R1']['d0']['val_nmae']['CH'] > 0


def test_every_run_has_reproduce_fields(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0'], seeds=(0, 1))
    out = EX.build(rd)
    assert len(out['runs']) == 2 * len(PEOPLE)
    for r in out['runs'].values():
        assert {'spec', 'subject', 'direction', 'seed', 'config_hash', 'command'} <= set(r)
        assert r['command'].startswith('python -m EXPERIMENTS.pipeline.run --exp TS2_R1_w20_full_d0')


def test_export_leaves_out_host(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0'])
    out = EX.build(rd)
    for r in out['runs'].values():
        assert 'host' not in r['provenance'] and r['provenance']['git_commit'] == 'c'


def _final(rd, cfg_dir, bad=None):
    """A complete synthetic set of study v2 final runs (what confirm needs), A0 = the carry_2s winner, whose seeds
    10-12 live in FE_cell_carry_2s. Trials start at 1.0 s; a w-step window first predicts (w - 1) x 10 ms later.
    ``bad``: a trial whose predictions are 5x noisier, for every run."""
    P = PT.load()
    fs, cells = P['final_seeds'], {c: v for c, v in P['cells'].items() if not v['gated']}
    net = {'units': 64, 'layers': 1, 'dropout': 0.02, 'lr': 0.005}
    arms = {f'FE_cell_{c}': (dict(net, window=v['window'], **({'state': 'carry', 'chunk': v['chunk']} if 'state' in v
                                                             else {'train_stride': v['train_stride'], 'batch': 64})),
                             fs['cell']) for c, v in cells.items()}
    arms['FE_A0'] = (dict(arms['FE_cell_carry_2s'][0]), [s for s in fs['A0'] if s not in fs['cell']])
    arms['FE_A1'] = ({'model': 'ridge_lagged', 'window': 51}, [10])
    arms['FE_A2'] = ({'window': 20, 'train_stride': 1, 'units': 8, 'layers': 2, 'dropout': 0.3, 'lr': 0.003,
                      'batch': 64}, fs['A2'])
    arms['FE_A3'] = (dict(arms['FE_A0'][0], passive='none'), fs['A3'])
    rng = np.random.default_rng(1)
    t = np.tile(1.0 + np.arange(600) / 100, 4)
    trial = np.repeat([0, 2, 4, 6], 600)
    angle = np.repeat([-0.3, 0.1, -0.1, 0.2], 600)
    y = 10 * np.sin(t / 2 + trial)
    for e, (changes, seeds) in arms.items():
        changes = dict({'model': 'lstm', 'passive': 'static'}, **changes)
        (cfg_dir / f'{e}.json').write_text(json.dumps({'id': e, 'title': e, 'changes': changes}))
        keep = t >= 1.0 + (changes['window'] - 1) / 100 - 1e-6
        for p in PEOPLE:
            for s in seeds:
                d = rd / e / p / 'S1toS2'
                d.mkdir(parents=True, exist_ok=True)
                yp = y + rng.normal(0, 1, len(y)) * np.where(trial == bad, 5.0, 1.0)
                np.savez(d / f'seed{s}.npz', y_own_target=y[keep], y_pred=yp[keep], trial=trial[keep], t=t[keep],
                         angle=angle[keep])
                np.savez(d / f'seed{s}_val.npz', y=y[:400], y_pred=yp[:400], common=np.ones(400, bool))
                (d / f'seed{s}.json').write_text(json.dumps({
                    'config_hash': 'a' * 16, 'train_s': 100.0 + s, 'versions': {'numpy': '2'},
                    'provenance': {'git_commit': 'c', 'host': 'box.example.home', 'versions': {'numpy': '2'}}}))
    (rd / 'choice.json').write_text(json.dumps({'A0': 'TS2_R1_carry_2s_d5',
                                                'cell_winners': {'carry_2s': 'TS2_R1_carry_2s_d5'}}))
    return rd


@pytest.fixture
def final(tmp_path, monkeypatch):
    cfg_dir = tmp_path / 'configs'
    cfg_dir.mkdir()
    monkeypatch.setattr(SPEC, 'CONFIGS', cfg_dir)
    rd = _final(tmp_path / 'results_v2', cfg_dir)
    return rd, EX.build(rd, with_confirm=False)


def test_final_scores_use_confirm_samples(final):
    rd, out = final
    runs, (start, _, scale), sfrom = CF.final_runs(rd)
    e, p, s = 'FE_cell_w100_full', 'EG', 11
    z = runs[e, p, 'S1toS2', s][1]
    y, yp, t, tr = (np.asarray(z[k], float) for k in ('y_own_target', 'y_pred', 't', 'trial'))
    f = out['final'][e][p]
    j = f['seeds'].index(s)
    assert f['positions'] == sorted(f['positions']) == [-0.3, -0.1, 0.1, 0.2]
    for k, i in enumerate([0, 4, 2, 6]):                                   # trials in ascending angle
        m = CF.scored_mask(t, tr, start, p, 'S1toS2', sfrom[p]['S1toS2']) & (tr == i)
        assert m.sum() < (tr == i).sum()
        err = yp[m] - y[m]
        assert abs(f['nmae'][k][j] - np.mean(np.abs(err)) / scale[p, 'S1toS2']) < 1e-12
        assert abs(f['mae'][k][j] - np.mean(np.abs(err))) < 1e-12
        assert abs(f['r2'][k][j] - (1 - np.sum(err ** 2) / np.sum((y[m] - y[m].mean()) ** 2))) < 1e-12
        assert abs(f['peak'][k] - np.max(np.abs(y[m]))) < 1e-12
    assert out['score_from_s'] == sfrom
    assert f['train_s'] == {'10': 110.0, '11': 111.0, '12': 112.0}


def test_a0_is_one_setting_with_cell_seeds(final):
    _, out = final
    assert 'FE_A0' in out['settings'] and 'FE_cell_carry_2s' not in out['settings']
    assert 'FE_cell_carry_2s' not in out['final']
    assert len(out['settings']) == 18 and set(out['final']) == set(out['settings'])
    assert all(out['final']['FE_A0'][p]['seeds'] == list(range(10, 20)) for p in PEOPLE)
    assert out['final']['FE_A1']['CH']['seeds'] == [10]


def test_settings_params(final):
    st = final[1]['settings']
    w = st['FE_cell_w50_o75']
    assert (w['kind'], w['cell'], w['model'], w['window_s'], w['overlap'], w['state'], w['target']) == \
        ('cell', 'w50_o75', 'lstm', 0.5, 0.74, 'reset', 'active')
    assert (w['units'], w['layers'], w['batch']) == (64, 1, 64)
    c = st['FE_cell_carry_5s']
    assert (c['state'], c['chunk_s'], c['overlap'], c['window_s']) == ('carry', 5.0, None, 0.2)
    assert st['FE_cell_w20_full']['overlap'] == 'full'
    a0 = st['FE_A0']
    assert (a0['kind'], a0['cell'], a0['state'], a0['chunk_s'], a0['batch']) == ('A0', 'carry_2s', 'carry', 2.0, None)
    assert st['FE_A3']['target'] == 'total' and st['FE_A3']['kind'] == 'A3'
    a1 = st['FE_A1']
    assert (a1['model'], a1['cell'], a1['state'], a1['overlap'], a1['units']) == ('ridge_lagged', None, None, None,
                                                                                  None)
    a2 = st['FE_A2']
    assert (a2['cell'], a2['overlap'], a2['state'], a2['units'], a2['layers']) == (None, 'full', 'reset', 8, 2)
    assert all(v['title'] for v in st.values())


def test_no_legacy_and_no_host(final):
    _, out = final
    assert 'legacy' not in out
    assert all('host' not in r['provenance'] and 'positions' not in r for r in out['runs'].values())
    assert out['versions'] == [{'numpy': '2'}]


def test_tuning_has_params(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0'])
    p = EX.build(rd)['tuning']['w20_full']['R1']['d0']['params']
    assert set(p) == {'units', 'layers', 'dropout', 'lr', 'batch'} and p['units'] > 0


def test_size_budget(tmp_path):
    rd = tmp_path / 'results_v2'
    (rd).mkdir()
    (rd / 'choice.json').write_text('{}')
    ids = [f'TS2_R1_w20_full_d{i}' for i in range(9)] + [f'FE_cell_w{w}_full' for w in (20, 50, 100)]
    _runs(rd, ids, seeds=(10, 11, 12, 13, 14))
    out = tmp_path / 'study_v2.json'
    EX.write(EX.build(rd, with_confirm=False), out)
    size = out.stat().st_size
    # this is ~1/6 of the real run count; scaled up x6 it must stay within the 12 MB budget
    assert size * 6 <= EX.SIZE_BUDGET, size


def test_export_during_tuning_skips_unfinished(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'])
    for f in (rd / 'TS2_R1_w20_full_d1' / 'YES').rglob('*'):
        if f.is_file():
            f.unlink()
    out = EX.build(rd)
    assert out['tuning']['w20_full']['R1']['d1']['incomplete'] is True
    assert out['tuning']['w20_full']['R1']['d0']['val_nmae']['YES'] > 0

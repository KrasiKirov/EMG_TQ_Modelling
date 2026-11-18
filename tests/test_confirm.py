import json

import numpy as np
import pytest

from EXPERIMENTS.analysis import confirm as CF
from EXPERIMENTS.pipeline import provenance as PV, run as PR, spec as SPEC
from EXPERIMENTS.protocol import lock as LK, protocol as PT

P = PT.load()
PEOPLE = P['people']
CELLS = [c for c, v in P['cells'].items() if not v['gated']]


def _arms():
    fs = P['final_seeds']
    arms = {f'FE_cell_{c}': (['S1toS2'], fs['cell']) for c in CELLS}
    arms['FE_A0'] = (['S1toS2'], fs['A0'])
    arms['FE_A1'] = (['S1toS2'], [10])
    arms['FE_A2'] = (['S1toS2'], fs['A2'])
    arms['FE_A3'] = (['S1toS2'], fs['A3'])
    return arms


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Synthetic final runs: error 0.1 everywhere; overrides set an arm's error factor."""
    cfg_dir, rd = tmp_path / 'configs', tmp_path / 'results_v2'
    cfg_dir.mkdir()
    monkeypatch.setattr(SPEC, 'CONFIGS', cfg_dir)

    def make(factor=None, choice=True, skip=None, bad_hash=None, drift=None, data_mix=None, window=None):
        factor, window = factor or {}, window or {}
        rng = np.random.default_rng(0)
        late = [e for e in factor if e.endswith('late')]
        t = np.tile(1.0 + np.arange(900) / 100, 4)
        trial = np.repeat([0, 2, 4, 6], 900)
        for e, (dirs, seeds) in _arms().items():
            changes = {'units': 8, 'passive': 'none' if e == 'FE_A3' else 'static', 'window': window.get(e, 20)}
            if e == 'FE_A0':
                changes['window'] = 20
            (cfg_dir / f'{e}.json').write_text(json.dumps({'id': e, 'title': e, 'changes': changes}))
            cfg = dict(PR.BASE, **changes)
            for p in PEOPLE:
                for d in dirs:
                    for s in seeds:
                        if skip == (e, p, d, s) or (e == 'FE_A0' and d == 'S1toS2' and s < 15):
                            continue
                        out = rd / e / p / d
                        w = changes['window']
                        keep = (t >= 1.0 + (w - 1) / 100 - 1e-6)            # first prediction after one span
                        out.mkdir(parents=True, exist_ok=True)
                        y = 10 * np.sin(t / 3 + trial)
                        err = 0.1 * factor.get(e, 1.0) * (1 + 0.01 * rng.normal())
                        yp = y + err * 10 * rng.choice([-1, 1], len(y))
                        if e in late:                                     # big errors before 4 s
                            yp = np.where(t < 4.0, y + 50, yp)
                        np.savez(out / f'seed{s}.npz', y_own_target=y[keep], y_pred=yp[keep], trial=trial[keep],
                                 t=t[keep], angle=(trial / 10.0)[keep], session=np.ones_like(trial)[keep])
                        yv = y[:400]
                        np.savez(out / f'seed{s}_val.npz', y=yv, y_pred=yv + err, common=np.ones(400, bool),
                                 trial=trial[:400], t=t[:400])
                        prov = dict(git_commit='c1', git_dirty=False, protocol_hash=P['hash'],
                                    data_sha256={f'{p}_S1.csv': 'h1'})
                        if bad_hash == (e, p):
                            prov['protocol_hash'] = 'x' * 64
                        if data_mix == (e, p, s):
                            prov['data_sha256'] = {f'{p}_S1.csv': 'h2'}
                        h = 'f' * 16 if drift == (e, p) else PV.config_hash(cfg)
                        (out / f'seed{s}.json').write_text(json.dumps(
                            {'config': cfg, 'config_hash': h, 'provenance': prov, 'train_info': {'n_params': 100}}))
        if choice:
            (rd / 'choice.json').write_text(json.dumps({'A0': 'TS2_R2_w20_full_d0',
                                                        'cell_winners': {'w20_full': 'TS2_R2_w20_full_d0'}}))
        return rd
    return make


def test_refuses_before_a0(study):
    rd = study(choice=False)
    with pytest.raises(LK.TestLocked):
        CF.run(rd, P)


def test_incomplete_arm_refused(study):
    rd = study(skip=('FE_A2', 'JM', 'S1toS2', 13))
    with pytest.raises(CF.Refused, match=r'FE_A2.*JM.*S1toS2.*13'):
        CF.run(rd, P)


def test_wrong_protocol_hash_refused(study):
    with pytest.raises(CF.Refused, match='protocol'):
        CF.run(study(bad_hash=('FE_A1', 'CH')), P)


def test_config_hash_drift_refused(study):
    with pytest.raises(CF.Refused, match='FE_A0'):
        CF.run(study(drift=('FE_A0', 'EG')), P)


def test_data_hash_mixed_refused(study):
    with pytest.raises(CF.Refused, match=r'HM_S1\.csv.*FE_A3'):
        CF.run(study(data_mix=('FE_A3', 'HM', 12)), P)


def test_end_to_end_synthetic(study):
    out = CF.run(study(factor={'FE_A3': 1.3}), P)
    f3 = out['families']['F3']
    assert f3['A3 vs A0']['decision'] == 'worse'
    assert f3['A0 vs A1']['decision'] == 'no clear difference'
    assert set(f3['A3 vs A0']['per_person']) == set(PEOPLE)
    assert out['provenance_check']['commits'] == ['c1']
    assert out['limitations'] == P['limitations'] and len(out['limitations']) == 4
    f1 = out['families']['F1']
    assert f1['w200_full']['decision'] == 'stopped' and f1['w50_full']['decision'] == 'no clear difference'


def test_overlap_effect_averaged_over_lengths(study):
    out = CF.run(study(factor={'FE_cell_w20_o0': 1.4, 'FE_cell_w50_o0': 1.0}), P)
    f2 = out['families']['F2']
    assert f2['o0']['lengths'] == ['w20', 'w50', 'w100']
    assert all(np.isclose(v, np.log(1.4) / 3, atol=0.03) for v in f2['o0']['effects'].values())
    assert f2['o0']['decision'] == 'worse' and f2['o75']['decision'] == 'no clear difference'


def test_a0_seeds_from_cell_winner(study):
    out = CF.run(study(), P)
    assert set(out['families']['F3']['A3 vs A0']['comparator_seeds']['CH']) == set(range(10, 20))


def test_comparison_scored_on_samples_both_arms_predict(study):
    """An arm with a 300-step window first predicts 2.99 s into each trial: its own comparison (vs 0.2 s) is scored
    from 2.99 s, every other comparison keeps the 1 s start; the descriptive scores and D2b put every arm on one
    shared start (2.99 s), so they line up as one curve."""
    rd = study(window={'FE_cell_w50_full': 300})
    out = CF.run(rd, P)
    f1, f3 = out['families']['F1'], out['families']['F3']
    assert f1['w50_full']['score_from_s'] == dict.fromkeys(PEOPLE, 2.99)
    assert f1['w100_full']['score_from_s'] == dict.fromkeys(PEOPLE, 1.0)
    assert f3['A0 vs A1']['score_from_s'] == dict.fromkeys(PEOPLE, 1.0)
    assert out['score_from_s'] == {p: {'S1toS2': 2.99} for p in PEOPLE}
    runs = CF._load_runs(rd, P)
    shared = CF._scores(runs, P, {e for e, _, _, _ in runs}, CF._timing(runs))[0]
    assert out['descriptive']['FE_A0 S1toS2'] == {p: shared['FE_A0', p, 'S1toS2'] for p in PEOPLE}


def test_same_angle_trials_kept_apart(study, monkeypatch):
    out = CF.run(study(), P)
    pp = out['families']['F3']['A0 vs A1']['per_position']['CH']
    assert len(pp) == 4 and all('angle' in v for v in pp.values())


def test_no_session2_to_1_expected(study):
    out = CF.run(study(), P)
    assert all(not k.endswith('S2toS1') for k in out['descriptive'])

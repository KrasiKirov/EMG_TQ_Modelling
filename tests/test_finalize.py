import numpy as np

from EXPERIMENTS.pipeline import spec as SPEC
from EXPERIMENTS.tuning import finalize as FN, make_specs as MS, space as SPC
from ML.models.baselines import lagged_features, LaggedRidge


def _choice(tmp_path, monkeypatch):
    cfg = tmp_path / 'configs'
    cfg.mkdir()
    monkeypatch.setattr(SPEC, 'CONFIGS', cfg)
    winners = {}
    for c in ('w20_full', 'w100_full', 'w200_full', 'carry_5s'):
        ids = MS.write('R2', c, {'d3': SPC.configs(c)['d3']}, cfg)
        winners[c] = ids[0]
    return {'A0': winners['w100_full'], 'cell_winners': winners}, cfg


def test_final_seeds_fresh(tmp_path, monkeypatch):
    choice, cfg = _choice(tmp_path, monkeypatch)
    plan = FN.write(choice, cfg)
    assert plan and all(min(seeds) >= 10 for _, _, _, seeds in plan)
    assert {d for _, _, d, _ in plan} == {'S1toS2'}                 # D1: no session 2 -> 1 runs


def test_passed_gated_cell_gets_final_seeds(tmp_path, monkeypatch):
    choice, cfg = _choice(tmp_path, monkeypatch)
    plan = FN.write(choice, cfg)
    assert {tuple(s) for i, _, _, s in plan if i == 'FE_cell_w200_full'} == {(10, 11, 12)}


def test_a3_differs_from_a0_only_in_passive(tmp_path, monkeypatch):
    choice, cfg = _choice(tmp_path, monkeypatch)
    FN.write(choice, cfg)
    a0, a3 = SPEC.load('FE_A0', cfg).changes, SPEC.load('FE_A3', cfg).changes
    assert {k for k in set(a0) | set(a3) if a0.get(k) != a3.get(k)} == {'passive'} and a3['passive'] == 'none'
    a2 = SPEC.load('FE_A2', cfg).changes
    assert (a2['units'], a2['layers'], a2['dropout'], a2['lr'], a2['window']) == (8, 2, 0.3, 0.003, 20)


def test_ridge_alpha_per_person():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 60, 5))
    y = 3 * X[:, -1, 0] + 2 * X[:, -11, 1] + rng.normal(0, 0.1, 600)
    m = LaggedRidge([1e-3, 1e3]).fit(X[:400], y[:400], X[400:], y[400:])
    assert m.alpha == 1e-3 and set(m.val_mae) == {1e-3, 1e3}
    assert np.mean(np.abs(m.predict(X[400:]) - y[400:])) < 0.2


def test_lagged_features_shape_and_causal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 60, 5))
    F = lagged_features(X)
    assert F.shape == (10, 49)
    X2 = X.copy()
    X2[:, :-51] = 99.0                     # samples older than the 50-step lag
    assert np.allclose(lagged_features(X2), F)


def test_a0_reuses_cell_winner_seeds(tmp_path, monkeypatch):
    choice, cfg = _choice(tmp_path, monkeypatch)
    plan = FN.write(choice, cfg)
    a0 = {d: s for i, _, d, s in plan if i == 'FE_A0'}
    assert a0 == {'S1toS2': [13, 14, 15, 16, 17, 18, 19]}
    assert SPEC.load('FE_A0', cfg).changes == SPEC.load('FE_cell_w100_full', cfg).changes

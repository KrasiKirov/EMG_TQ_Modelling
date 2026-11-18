import json

import numpy as np

from EXPERIMENTS.analysis import study_v2_export as EX
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
                                                             'provenance': {'git_commit': 'c'},
                                                             'train_info': {'n_params': 10}}))


def test_export_before_a0_has_no_session2(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'])
    out = EX.build(rd, tmp_path / 'v1')
    assert out['confirm'] is None
    assert all('positions' not in r for r in out['runs'].values())
    assert out['tuning']['w20_full']['R1']['d0']['val_nmae']['CH'] > 0


def test_every_run_has_reproduce_fields(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0'], seeds=(0, 1))
    out = EX.build(rd, tmp_path / 'v1')
    assert len(out['runs']) == 2 * len(PEOPLE)
    for r in out['runs'].values():
        assert {'spec', 'subject', 'direction', 'seed', 'config_hash', 'command'} <= set(r)
        assert r['command'].startswith('python -m EXPERIMENTS.pipeline.run --exp TS2_R1_w20_full_d0')


def test_legacy_seeds_0_4(tmp_path):
    v1 = tmp_path / 'results'
    _runs(v1, ['W00_reference', 'WG_2s_o00_10ms'], seeds=range(8))
    _runs(v1, ['HF_u8_l2_d00_lr01'], seeds=(0,))
    out = EX.build(tmp_path / 'results_v2', v1)
    leg = out['legacy']
    assert set(leg['experiments']) == {'W00_reference', 'WG_2s_o00_10ms'} and 'exploratory' in leg['note']
    assert all(set(map(int, seeds)) == set(range(5)) for e in leg['experiments'].values() for seeds in e.values())


def test_size_budget(tmp_path):
    rd = tmp_path / 'results_v2'
    (rd).mkdir()
    (rd / 'choice.json').write_text('{}')
    ids = [f'TS2_R1_w20_full_d{i}' for i in range(9)] + [f'FE_cell_w{w}_full' for w in (20, 50, 100)]
    _runs(rd, ids, seeds=(10, 11, 12, 13, 14))
    out = tmp_path / 'study_v2.json'
    EX.write(EX.build(rd, tmp_path / 'v1', with_confirm=False), out)
    size = out.stat().st_size
    # this is ~1/6 of the real run count; scaled up x6 it must stay within the 12 MB budget
    assert size * 6 <= EX.SIZE_BUDGET, size


def test_export_during_tuning_skips_unfinished(tmp_path):
    rd = tmp_path / 'results_v2'
    _runs(rd, ['TS2_R1_w20_full_d0', 'TS2_R1_w20_full_d1'])
    for f in (rd / 'TS2_R1_w20_full_d1' / 'YES').rglob('*'):
        if f.is_file():
            f.unlink()
    out = EX.build(rd, tmp_path / 'v1')
    assert out['tuning']['w20_full']['R1']['d1']['incomplete'] is True
    assert out['tuning']['w20_full']['R1']['d0']['val_nmae']['YES'] > 0

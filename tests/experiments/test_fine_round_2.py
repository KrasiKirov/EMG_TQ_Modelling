"""Second fine round (EXPERIMENTS/protocol/fine_round_2.json): specs, jobs and the cell-winner rule."""
import json

import numpy as np
import pytest

from EXPERIMENTS.pipeline import spec as SPEC
from EXPERIMENTS.pipeline.settings import strip_retired
from EXPERIMENTS.tuning import fine2 as F2, make_specs as MS, space as SPC

P6 = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
WINNERS = {'carry_2s': 'TS2_R1_carry_2s_d5', 'w50_full': 'TS2_R2_w50_full_f3', 'w100_full': 'TS2_R2_w100_full_f3'}


def test_protocol_pins_the_design():
    assert F2.P['cells'] == ['carry_2s', 'w50_full', 'w100_full']
    assert F2.P['draws']['n'] == 4 and F2.P['draws']['seed'] == 20261125
    assert F2.P['seeds'] == {'draws': [0, 1, 2], 'winner': [2]}


def test_draws_stay_at_or_below_64_units_and_one_layer():
    centre = {'units': 64, 'layers': 1, 'dropout': 0.004, 'lr': 0.00556, 'batch': 64}
    g = F2.configs('w50_full', centre)
    assert list(g) == ['g0', 'g1', 'g2', 'g3']
    assert all(c['units'] in (32, 64) and c['layers'] == 1 and c['batch'] in (32, 64, 128) for c in g.values())
    assert all(0.00556 / 3 <= c['lr'] <= 0.00556 * 3 and 0 <= c['dropout'] <= 0.104 for c in g.values())
    assert g == F2.configs('w50_full', centre)                              # reproducible
    assert list(g.values()) != list(SPC.fine_configs('w50_full', centre).values())   # not study_v2's R2 draws


def test_carried_state_draws_have_no_batch():
    g = F2.configs('carry_2s', {'units': 64, 'layers': 1, 'dropout': 0.017, 'lr': 0.00754})
    assert all('batch' not in c for c in g.values())


def _winners(d):
    """Winner specs as committed (windowed ones still hold the retired keys at their full-overlap values)."""
    for cell, wid in WINNERS.items():
        rnd, name = wid.split('_')[1], wid.rsplit('_', 1)[1]
        centre = {'units': 64, 'layers': 1, 'dropout': 0.01, 'lr': 0.005, **({} if 'carry' in cell else {'batch': 64})}
        (i,) = MS.write(rnd, cell, {name: centre}, d)
        if 'carry' not in cell:
            raw = json.loads((d / f'{i}.json').read_text())
            raw['changes'].update(train_stride=1, epoch_windows='full')
            (d / f'{i}.json').write_text(json.dumps(raw, indent=1) + '\n')
    return dict(WINNERS)


def test_write_makes_four_draws_and_a_winner_copy_per_cell(tmp_path):
    ids = F2.write(_winners(tmp_path), tmp_path)
    assert len(ids) == 15
    for cell, wid in WINNERS.items():
        w = SPEC.load(f'TS2_R4_{cell}_w', tmp_path).changes
        assert w == strip_retired(dict(SPEC.load(wid, tmp_path).changes))      # the winner's settings unchanged
        for k in range(4):
            g = SPEC.load(f'TS2_R4_{cell}_g{k}', tmp_path).changes
            assert g['train_protocol'] == 'study400' and g['shuffle'] is False
            assert {k2: g[k2] for k2 in ('window', 'state', 'chunk') if k2 in g} == SPC.changes(cell)
            assert 'train_stride' not in g and g['units'] <= 64
    assert F2.write(_winners(tmp_path), tmp_path) == ids                        # re-runnable


def test_jobs_are_234_and_the_copy_runs_seed_2_only(tmp_path):
    ids = F2.write(_winners(tmp_path), tmp_path)
    j = F2.jobs(ids, P6)
    assert len(j) == 234 and {d for _, _, d, _ in j} == {'S1toS2'}
    assert {s for i, _, _, s in j if i.endswith('_w')} == {2}
    assert {s for i, _, _, s in j if '_g' in i} == {0, 1, 2}
    assert len({tuple(x) for x in j}) == 234


def _val(rd, exp, seed, err, n_params=1000):
    """One run's session-1 validation file: y_pred = y + err, so nMAE is proportional to err."""
    for p in P6:
        d = rd / exp / p / 'S1toS2'
        d.mkdir(parents=True, exist_ok=True)
        y = np.linspace(-1, 1, 200)
        np.savez(d / f'seed{seed}_val.npz', y=y, y_pred=y + err, common=np.ones(200, bool))
        (d / f'seed{seed}.json').write_text(json.dumps({'train_info': {'n_params': n_params}}))


def _cell(rd, winner_errs, g_err):
    w = 'TS2_R1_carry_2s_d5'
    for s, e in zip((0, 1), winner_errs[:2]):
        _val(rd, w, s, e)
    _val(rd, 'TS2_R4_carry_2s_w', 2, winner_errs[2])
    for k in range(4):
        for s in (0, 1, 2):
            _val(rd, f'TS2_R4_carry_2s_g{k}', s, g_err + 0.01 * k)
    return w


def test_winner_seed_2_is_read_from_the_copy(tmp_path):
    # winner seeds 0, 1, 2 = 1.0, 0.2, 0.2 -> median 0.2 beats the draws (0.5); without seed 2 it would be 0.6
    w = _cell(tmp_path, (1.0, 0.2, 0.2), 0.5)
    assert F2.cell_winner('carry_2s', w, P6, tmp_path) == w


def test_a_clearly_better_draw_wins(tmp_path):
    w = _cell(tmp_path, (1.0, 1.0, 1.0), 0.5)
    assert F2.cell_winner('carry_2s', w, P6, tmp_path) == 'TS2_R4_carry_2s_g0'


def test_missing_copy_run_raises(tmp_path):
    w = _cell(tmp_path, (1.0, 1.0, 1.0), 0.5)
    for f in (tmp_path / 'TS2_R4_carry_2s_w').rglob('seed2_val.npz'):
        f.unlink()
    with pytest.raises(FileNotFoundError):
        F2.cell_winner('carry_2s', w, P6, tmp_path)


def test_a0_kept_when_no_winner_changes(tmp_path):
    choice = {'A0': 'TS2_R1_carry_2s_d5', 'cell_winners': dict(WINNERS)}
    assert F2.choose_a0(choice, dict(WINNERS), P6, tmp_path) == 'TS2_R1_carry_2s_d5'     # reads no file

import json

import numpy as np
import pytest

from EXPERIMENTS.analysis import study_scores as SS
from EXPERIMENTS.pipeline.run import BASE


def cfg(**changes):
    return dict(BASE, **changes)


def test_facets_of_the_reference():
    f = SS.facets(cfg(passive='static', window=20, shuffle=False, train_protocol='fast'))
    assert f == {'length_s': 0.2, 'step_ms': 10, 'overlap_pct': 95, 'target': 'active', 'state': 'reset',
                 'chunk_s': None, 'shuffle': False, 'epochs': 150}


def test_facets_of_a_grid_cell_and_total_torque():
    # 500 ms = 10 steps of 50 ms; a training window every 120 ms -> 76 % overlap
    f = SS.facets(cfg(passive='none', window=10, input_step=5, train_stride=12, train_protocol='fast400'))
    assert (f['length_s'], f['step_ms'], f['overlap_pct'], f['target'], f['epochs']) == (0.5, 50, 76, 'total', 400)


def test_facets_of_carried_state():
    f = SS.facets(cfg(passive='static', state='carry', chunk=50))
    assert (f['state'], f['chunk_s']) == ('carry', 0.5)


def test_question_and_pairing_from_the_id():
    assert SS.question('W00_reference') == ('reference', None)
    assert SS.question('WT1_total') == ('reference', 'W00_reference')
    assert SS.question('WT1_total_400') == ('epochs', 'WT1_total')
    assert SS.question('WO1_90') == ('overlap', None)
    assert SS.question('WO1_90_tot') == ('overlap', 'WO1_90')
    assert SS.question('WL2c_1s_10ms') == ('step', None)
    assert SS.question('WL3_2s') == ('length', None)
    assert SS.question('WG_1s_o50_tot') == ('grid', 'WG_1s_o50')
    assert SS.question('WS1_shuffled') == ('shuffle', None)
    assert SS.question('WC3_carry_1s') == ('carry', None)
    assert SS.question('WC4_carry_10s') == ('carry', None)
    assert SS.question('WO4_00_eq') == ('overlap_eq', None)   # equal updates per epoch, against the reference
    assert SS.question('W00_reference_lp4') is None          # earlier rounds are not part of the study page
    assert SS.question('WF1_lowpass2') is None


def test_position_scores_are_within_each_position():
    # two positions with different torque levels; the prediction tracks the shape but is 1 Nm off at the second
    t = np.arange(0, 20, .01)
    y = np.concatenate([np.sin(t), 10 + np.sin(t)])
    p = np.concatenate([np.sin(t), 11 + np.sin(t)])
    a = np.repeat([-0.5, 0.1], len(t))
    tt = np.concatenate([t, t + 100])
    rows = SS.position_scores(y, p, a, tt)
    assert len(rows) == 2
    first, second = (dict(zip(SS.METRICS, r)) for r in rows)
    assert first['mae'] == pytest.approx(0) and first['r2'] == pytest.approx(1)
    assert second['mae'] == pytest.approx(1) and second['rmse'] == pytest.approx(1)
    # within the second position the 1 Nm offset is larger than the spread -> R2 far below 0, shape still exact
    assert second['r2'] < 0 and second['r2_demeaned'] == pytest.approx(1)
    assert second['bias'] == pytest.approx(1)


def test_build_writes_rows_per_seed_and_position(tmp_path):
    results = tmp_path / 'results'
    t = np.arange(0, 20, .01)
    for exp, cfg_ in [('W00_reference', {'passive': 'static'}), ('WT1_total', {'passive': 'none'}),
                      ('W00_reference_lp4', {'passive': 'static'})]:
        for seed in (0, 1):
            d = results / exp / 'EG' / 'S1toS2'; d.mkdir(parents=True, exist_ok=True)
            y = np.concatenate([np.sin(t), 5 + np.sin(t)])
            np.savez(d / f'seed{seed}.npz', y_own_target=y.astype(np.float32), y_pred=(y + .1 * seed).astype(np.float32),
                     angle=np.repeat([-0.5, 0.1], len(t)).astype(np.float32),
                     t=np.concatenate([t, t + 100]).astype(np.float32), trial=np.repeat([1, 2], len(t)))
            (d / f'seed{seed}.json').write_text(json.dumps({'config': cfg_, 'train_s': 60.0,
                                                            'train_info': {'epochs': 100}, 'n_windows': {'train': 7}}))
    titles = {'W00_reference': 'Reference', 'WT1_total': 'Reference (total)'}
    out = SS.build(results, titles)
    assert set(out['settings']) == {'W00_reference', 'WT1_total'}          # the earlier-round setting is left out
    assert out['settings']['WT1_total']['pair'] == 'W00_reference'
    assert out['settings']['WT1_total']['target'] == 'total'
    assert out['settings']['W00_reference']['title'] == 'Reference'
    runs = out['runs']['W00_reference']['EG']
    assert sorted(runs) == ['0', '1'] and len(runs['0']) == 2 and len(runs['0'][0]) == len(SS.METRICS)
    assert out['positions']['EG'] == [-0.5, 0.1]
    assert out['epochs']['W00_reference']['EG'] == {'0': 100, '1': 100}

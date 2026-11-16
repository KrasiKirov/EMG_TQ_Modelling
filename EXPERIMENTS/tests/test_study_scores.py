import json

import numpy as np
import pytest

from EXPERIMENTS.analysis import study_scores as SS
from EXPERIMENTS.pipeline.run import BASE


VERSIONS = {'python': '3.13.2', 'tensorflow': '2.21.0', 'numpy': '2.4.3'}


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
            (d / f'seed{seed}.json').write_text(json.dumps({'config': cfg_, 'train_s': 60.0, 'train_info': {'epochs': 100},
                                                            'n_windows': {'train': 7}, 'versions': VERSIONS}))
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
    assert out['times']['W00_reference']['EG'] == {'0': 60.0, '1': 60.0}
    assert out['versions'] == [VERSIONS]                                   # the software every run reported


def test_peaks_are_the_largest_measured_torque_at_each_position(tmp_path):
    results = tmp_path / 'results'
    t = np.arange(0, 20, .01)
    for exp, passive, level in [('W00_reference', 'static', 1.0), ('WL3_2s', 'static', 1.0), ('WT1_total', 'none', 3.0)]:
        d = results / exp / 'EG' / 'S1toS2'; d.mkdir(parents=True)
        keep = t >= (10 if exp == 'WL3_2s' else 0)            # a longer window starts later; its peak cannot be lower
        y = level * np.concatenate([np.sin(t[keep]), -4 + np.sin(t[keep])])
        np.savez(d / 'seed0.npz', y_own_target=y.astype(np.float32), y_pred=y.astype(np.float32),
                 angle=np.repeat([-0.5, 0.1], keep.sum()).astype(np.float32),
                 t=np.concatenate([t[keep], t[keep] + 100]).astype(np.float32), trial=np.repeat([1, 2], keep.sum()))
        (d / 'seed0.json').write_text(json.dumps({'config': {'passive': passive}}))
    out = SS.build(results, {})
    # max |torque| per position, by target: -4 + sin reaches -5 (5 Nm); total torque here is 3 times larger
    assert out['peaks']['active']['EG'] == [pytest.approx(1.0, abs=.01), pytest.approx(5.0, abs=.01)]
    assert out['peaks']['total']['EG'] == [pytest.approx(3.0, abs=.01), pytest.approx(15.0, abs=.01)]


def test_machine_names_the_processor_memory_and_how_runs_were_spread():
    raw = {'machine_name': 'MacBook Air', 'chip_type': 'Apple M3', 'hw.physicalcpu': '8', 'hw.perflevel0.physicalcpu': '4',
           'hw.perflevel1.physicalcpu': '4', 'hw.memsize': str(16 * 2 ** 30), 'os': '27.2.1', 'gpus': []}
    assert SS.machine(raw, workers=6, threads=1) == {
        'computer': 'MacBook Air', 'processor': 'Apple M3', 'cores': 8, 'performance_cores': 4, 'efficiency_cores': 4,
        'ram_gb': 16, 'os': 'macOS 27.2.1', 'accelerator': 'none (CPU only)', 'parallel_runs': 6, 'threads_per_run': 1}


def _study(tmp_path, errors, seeds=5, start=None):
    """Fake results: per setting, the prediction is the target + error × (seed + 1) at both positions (people EG, JM).
    start[exp] = first time stamp of that setting's test samples (a longer window starts later)."""
    results = tmp_path / 'results'
    t = np.arange(0, 20, .01)
    for exp, err in errors.items():
        for subj in ('EG', 'JM'):
            d = results / exp / subj / 'S1toS2'; d.mkdir(parents=True, exist_ok=True)
            keep = t >= (start or {}).get(exp, 0)
            tt = np.concatenate([t[keep], t[keep] + 100])
            y = np.concatenate([np.sin(t[keep]), 5 + np.sin(t[keep])])
            for seed in range(seeds):
                np.savez(d / f'seed{seed}.npz', y_own_target=y.astype(np.float32),
                         y_pred=(y + err * (seed + 1)).astype(np.float32),
                         angle=np.repeat([-0.5, 0.1], keep.sum()).astype(np.float32), t=tt.astype(np.float32),
                         trial=np.repeat([1, 2], keep.sum()))
                (d / f'seed{seed}.json').write_text(json.dumps({'config': {'passive': 'static'}, 'train_s': 60.0 + seed,
                                                                'train_info': {'epochs': 100}}))
    return results


def test_best_and_worst_are_the_lowest_and_highest_median_error_ratio(tmp_path):
    results = _study(tmp_path, {'W00_reference': .2, 'WL2_1s': .1, 'WO4_00': .4, 'WL3_2s': .25})
    out = SS.build(results, {})
    assert SS.extremes(out) == {'best': 'WL2_1s', 'worst': 'WO4_00', 'clearly_better': ['WL2_1s']}


def test_settings_with_too_few_seeds_cannot_be_best_or_worst(tmp_path):
    results = _study(tmp_path, {'W00_reference': .2, 'WL2_1s': .1, 'WO4_00': .4})
    for subj in ('EG', 'JM'):
        for seed in (1, 2, 3, 4):
            (results / 'WO4_00' / subj / 'S1toS2' / f'seed{seed}.npz').unlink()
    assert SS.extremes(SS.build(results, {}))['worst'] == 'WL2_1s'


def test_waves_keep_the_typical_seed_and_each_settings_own_start(tmp_path):
    results = _study(tmp_path, {'W00_reference': .2, 'WL2_1s': .1, 'WO4_00': .4}, start={'WO4_00': 10})
    out = SS.build(results, {})
    w = SS.waves(results, out, SS.extremes(out), hz=20)
    assert w['hz'] == 20 and (w['best'], w['worst'], w['reference']) == ('WL2_1s', 'WO4_00', 'W00_reference')
    eg = w['people']['EG']
    assert eg['positions'] == [-0.5, 0.1]
    target = eg['target'][1]                        # second position, from the setting that starts earliest
    assert target['t0'] == pytest.approx(100.02) and len(target['y']) == 400 and target['y'][0] == pytest.approx(5.02, abs=.01)
    worst = eg['pred']['WO4_00'][0]                 # 5 seeds: errors .4 .8 1.2 1.6 2.0 -> the middle one is seed 2
    assert worst['seed'] == 2 and worst['t0'] == pytest.approx(10.02) and len(worst['y']) == 200
    assert worst['y'][0] == pytest.approx(np.sin(10.02) + 1.2, abs=.02)


def test_run_versions_cover_every_results_folder_not_only_the_study(tmp_path):
    results = tmp_path / 'results'
    old = {'python': '3.11.1', 'tensorflow': '2.21.0', 'numpy': '2.4.4'}
    for exp, subj, seed, v in [('W00_reference', 'EG', 0, VERSIONS), ('W00_reference', 'JM', 0, VERSIONS),
                               ('N0_total_trainmax', 'EG', 0, old), ('X00_baseline', 'EG', 0, None)]:
        d = results / exp / subj / 'S1toS2'; d.mkdir(parents=True, exist_ok=True)
        (d / f'seed{seed}.json').write_text(json.dumps({'versions': v} if v else {}))
    assert SS.run_versions(results) == {'W00_reference': [VERSIONS], 'N0_total_trainmax': [old], 'X00_baseline': []}

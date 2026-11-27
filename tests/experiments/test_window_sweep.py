"""Window sweep (EXPERIMENTS/protocol/window_sweep.json): fixed configs, only the length changes."""
import json

from EXPERIMENTS.pipeline import spec as SPEC
from EXPERIMENTS.pipeline.settings import strip_retired
from EXPERIMENTS.tuning import window_sweep as WS

P6 = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
RESET = {'model': 'lstm', 'window': 50, 'train_stride': 1, 'epoch_windows': 'full', 'units': 64, 'layers': 1,
         'dropout': 0.004, 'lr': 0.00556, 'batch': 64, 'shuffle': False, 'train_protocol': 'study400'}
CARRY = {'model': 'lstm', 'state': 'carry', 'chunk': 200, 'window': 20, 'units': 64, 'layers': 1, 'dropout': 0.017,
         'lr': 0.00754, 'shuffle': False, 'train_protocol': 'study400'}


def _refs(d):
    for i, ch in (('FE_cell_w50_full', RESET), ('FE_A0', CARRY)):
        (d / f'{i}.json').write_text(json.dumps({'id': i, 'title': i, 'changes': ch}, indent=1) + '\n')


def test_protocol_pins_the_design():
    assert WS.P['seeds'] == [10, 11, 12, 13, 14]
    assert WS.P['arms']['reset'] == {'w50': 50, 'w100': 100, 'w200': 200, 'w500': 500, 'w1000': 1000}
    assert WS.P['arms']['carry'] == {'carry50': 50, 'carry100': 100, 'carry200': 200, 'carry500': 500,
                                     'carry1000': 1000}


def test_only_the_length_differs_from_the_reference(tmp_path):
    _refs(tmp_path)
    ids = WS.write(tmp_path)
    assert sorted(ids) == sorted(f'FE_win_{a}' for a in ('w50', 'w100', 'w200', 'w500', 'w1000',
                                                          'carry50', 'carry100', 'carry500', 'carry1000'))
    for i in ids:
        ch = dict(SPEC.load(i, tmp_path).changes)
        ref, key = (strip_retired(CARRY), 'chunk') if 'carry' in i else (strip_retired(RESET), 'window')
        assert {k for k in set(ch) | set(ref) if ch.get(k) != ref.get(k)} <= {key}
        assert ch[key] == int(i.rsplit('carry' if 'carry' in i else 'w', 1)[1])
        assert 'train_stride' not in ch
    assert SPEC.load('FE_win_carry50', tmp_path).changes['window'] == 20
    assert WS.write(tmp_path) == ids                                       # re-runnable


def test_jobs_are_252_longest_first_and_reuse_existing_seeds(tmp_path):
    _refs(tmp_path)
    j = WS.jobs(WS.write(tmp_path), P6)
    assert len(j) == 252 and len({tuple(x) for x in j}) == 252
    assert {d for _, _, d, _ in j} == {'S1toS2'}
    assert {s for i, _, _, s in j if i == 'FE_win_w50'} == {13, 14}           # 10-12 are FE_cell_w50_full's
    assert {s for i, _, _, s in j if i == 'FE_win_w1000'} == {10, 11, 12, 13, 14}
    order = list(dict.fromkeys(i for i, *_ in j))
    assert order[:4] == ['FE_win_w1000', 'FE_win_w500', 'FE_win_w200', 'FE_win_w100'] and order[-1] == 'FE_win_w50'


def test_sources_name_the_reused_runs():
    s = WS.sources()
    assert s['w50'] == [('FE_cell_w50_full', x) for x in (10, 11, 12)] + [('FE_win_w50', 13), ('FE_win_w50', 14)]
    assert s['carry200'] == [('FE_cell_carry_2s', x) for x in (10, 11, 12)] + [('FE_A0', 13), ('FE_A0', 14)]
    assert s['w1000'] == [('FE_win_w1000', x) for x in (10, 11, 12, 13, 14)]

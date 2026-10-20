"""One exploratory YES run: low-pass order 2, bandpass order 4.

Overrides only this process's filter-design function during data loading.
Never changes shared defaults, benchmark outputs, or presentation inputs.
Run from the repository parent with python -m ML.training.isolated_filter_probe.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ML.preprocessing import emg_envelope
from ML.training import load_subject, train_model, evaluate_model

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'reports/yes_eight_positions_reconstructed_20261013/YES/metrics.partial.json'


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def lowpass_only(calls):
    def design(order, wn, *args, **kwargs):
        kind = kwargs.get('btype', 'low')
        assert order == 4, (order, kind)
        actual = 2 if kind == 'low' else order
        calls[(kind, actual)] += 1
        return butter(actual, wn, *args, **kwargs)
    return design


def verify_override():
    signal = np.random.default_rng(42).normal(size=6000)
    original, original_stages = emg_envelope.extract_envelope(signal, return_intermediates=True)
    calls = Counter()
    with patch.object(emg_envelope, 'butter', lowpass_only(calls)):
        changed, changed_stages = emg_envelope.extract_envelope(signal, return_intermediates=True)
    np.testing.assert_array_equal(original_stages['filtered'], changed_stages['filtered'])
    expected = np.clip(sosfiltfilt(butter(2, 2 / 500, output='sos'), original_stages['rectified']), 0, None)
    np.testing.assert_allclose(changed, expected, rtol=0, atol=0)
    assert not np.allclose(original, changed)
    assert calls == {('band', 4): 1, ('low', 2): 1}
    assert emg_envelope.butter is butter and emg_envelope.DEFAULT_ORDER == 4
    print('PASS: only low-pass order changes; bandpass samples and shared defaults stay identical.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify_override()
    if args.check_only:
        return
    if args.output is None:
        parser.error('--output is required for training')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / 'status.json'
    status = {'state': 'loading', 'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        baseline = next(r for r in json.loads(BASELINE.read_text()) if r['seed'] == 11)
        assert baseline['window_ms'] == 500 and len(baseline['per_position']) == 8
        write_json(output / 'baseline_seed11.json', baseline)
        config = {
            'purpose': 'Exploratory single-seed comparison only. Excluded from benchmark and presentation.',
            'subject': 'YES', 'seed': 11, 'runs': 1,
            'bandpass_design_order': 4, 'bandpass_hz': [30, 300],
            'lowpass_design_order': 2, 'lowpass_hz': 2,
            'filter_application': 'zero-phase forward-backward SOS, both stages',
            'window_ms': 500, 'stride_ms': 10, 'batch_size': 64,
            'max_epochs': 200, 'learning_rate': 0.003,
            'model': 'same shared two-layer 8-unit LSTM, dropout 0.3',
            'baseline_source': str(BASELINE),
            'caveats': [
                'One seed cannot establish an order effect beyond seed variation.',
                'Four YES active-trial identities are historically reconstructed, not independently verified.',
                'Concurrent training can affect timings; this is not a speed benchmark.',
            ],
            'shared_source_sha256': {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT / 'preprocessing/emg_envelope.py', ROOT / 'preprocessing/dataset_builder.py', ROOT / 'training/__init__.py', ROOT / 'models/lstm.py', ROOT / 'config.py']
            },
        }
        write_json(output / 'config.json', config)
        load_started = time.perf_counter()
        calls = Counter()
        with patch.object(emg_envelope, 'butter', lowpass_only(calls)):
            data = load_subject('YES', yes_eight_positions=True)
        preprocess_s = time.perf_counter() - load_started
        assert emg_envelope.butter is butter and emg_envelope.DEFAULT_ORDER == 4
        assert set(calls) == {('band', 4), ('low', 2)}
        assert calls[('band', 4)] == calls[('low', 2)] > 0
        assert data['X_test'].shape == (70888, 50, 5)
        assert data['X_train'].shape == (63792, 50, 5)
        assert data['X_val'].shape == (15948, 50, 5)
        assert data['reconstruction'] == baseline['reconstruction']
        status.update(state='training', filter_calls={str(k): v for k, v in calls.items()})
        write_json(status_path, status)
        started = time.perf_counter()
        model, history = train_model(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            label='isolated YES low-pass order 2, seed 11',
            seed=11, lr=0.003, epochs=200, batch_size=64, verbose=2,
        )
        train_s = time.perf_counter() - started
        eval_started = time.perf_counter()
        result = evaluate_model(model, data['X_test'], data['y_test'], data['operating_positions'], mvc_tq_test=data['mvc_tq_test'])
        eval_s = time.perf_counter() - eval_started
        assert len(result['per_position']) == 8
        assert all(m['n'] == 8861 for m in result['per_position'].values())
        assert set(map(float, result['per_position'])) == set(map(float, baseline['per_position']))
        metrics = {
            'subject': 'YES', 'seed': 11, 'window_ms': 500, 'n_test': 70888,
            'preprocess_s': preprocess_s, 'train_s': train_s, 'eval_s': eval_s,
            'duration_s': train_s + eval_s, 'epochs_completed': len(history.history['loss']),
            'overall': result['overall'], 'per_position': result['per_position'],
            'reconstruction': data['reconstruction'],
        }
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'history.json', history.history)
        rows = ['# Isolated low-pass order comparison', '',
                'YES, reconstructed eight-position protocol, seed 11. Only the envelope low-pass design order changes (4 to 2). The bandpass remains order 4. Both use forward-backward filtering.', '',
                'Exploratory only: no presentation or primary benchmark update. One seed does not establish a consistent improvement. Four active-trial identities are inferred, not independently verified.', '',
                '| Scope | Metric | Order 4 baseline | Order 2 probe | Change (2 minus 4) |',
                '|---|---|---:|---:|---:|']
        pairs = [('Pooled', baseline['overall'], result['overall'])]
        for pos, values in sorted(result['per_position'].items()):
            pairs.append((f'{float(pos):+.3f} rad', baseline['per_position'][str(pos)], values))
        for scope, before, after in pairs:
            for metric in ['r2', 'vaf', 'rmse', 'mae']:
                rows.append(f'| {scope} | {metric} | {before[metric]:.4f} | {after[metric]:.4f} | {after[metric] - before[metric]:+.4f} |')
        rows.extend(['', 'R² is unitless; VAF is percent; RMSE and MAE are Nm. Higher R²/VAF and lower errors are better.', '',
                     f'Probe: {metrics["epochs_completed"]} epochs, training {train_s / 60:.2f} min, evaluation {eval_s:.2f} s, preprocessing {preprocess_s:.2f} s.',
                     f'Baseline: {baseline["epochs_completed"]} epochs, training {baseline["train_s"] / 60:.2f} min. Concurrent workload makes speed comparisons unreliable.'])
        (output / 'comparison.md').write_text('\n'.join(rows) + '\n')
        status.update(state='complete', completed_utc=datetime.now(timezone.utc).isoformat(), completed_runs=1)
        write_json(status_path, status)
        print(json.dumps(safe(metrics['overall'])), flush=True)
        print(f'COMPLETE: {output}', flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()

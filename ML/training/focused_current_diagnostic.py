"""One-seed current-pipeline diagnostic for EG, IES and JM.

This is deliberately separate from the repeated-run benchmark and presentation.
It saves train/validation/retest position summaries, including signed bias,
without merging any output into reports/repeated_runs.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from ML.training import load_subject, train_model
from ML.evaluation.metrics import compute_metrics, snap_to_operating_points
from ML.config import N_STEPS, MODEL_DIR


SUBJECTS = {
    'EG': (-0.7393509903, -0.0382968193),
    'IES': (-0.5907963621, 0.1437923659),
    'JM': (-0.7867236040, 0.0547125974),
}
SEED = 11
EPOCHS = 200
BATCH = 64


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


def summarize(X, y, pred, positions, target_positions):
    snapped, _ = snap_to_operating_points(X[:, -1, 4], positions)
    output = {}
    for label, target in target_positions.items():
        position = min(positions, key=lambda p: abs(float(p) - target))
        mask = np.isclose(snapped, position)
        metric = compute_metrics(y[mask], pred[mask])
        residual = pred[mask] - y[mask]
        output[label] = {
            'angle_rad': float(position),
            'n': int(mask.sum()),
            **metric,
            'bias': float(np.mean(residual)),
            'residual_sd': float(np.std(residual, ddof=1)),
            'target_sd': float(np.std(y[mask], ddof=1)),
            'target_iqr': float(np.percentile(y[mask], 75) - np.percentile(y[mask], 25)),
        }
    return output


def main():
    root = Path('/private/tmp/focused_current_diagnostic_20261013')
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        'purpose': 'Focused confirmation only; excluded from repeated_runs and presentation.',
        'seed': SEED,
        'epochs': EPOCHS,
        'batch_size': BATCH,
        'window_steps': N_STEPS,
        'window_ms': N_STEPS * 10,
        'subjects': list(SUBJECTS),
        'started_utc': time.time(),
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    for subject, (weak, best) in SUBJECTS.items():
        started = time.perf_counter()
        data = load_subject(subject)
        model, history = train_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            label=f'focused {subject} seed {SEED}',
            seed=SEED, epochs=EPOCHS, batch_size=BATCH, lr=0.003,
            verbose=0,
        )
        results = {}
        for split, X, y in [
            ('train', data['X_train'], data['y_train']),
            ('validation', data['X_val'], data['y_val']),
            ('retest', data['X_test'], data['y_test']),
        ]:
            pred = model.predict(X, verbose=0).ravel()
            results[split] = {
                'pooled': {**compute_metrics(y, pred), 'bias': float(np.mean(pred - y)), 'n': int(len(y))},
                'positions': summarize(X, y, pred, data['operating_positions'], {'weak': weak, 'strong': best}),
            }
        payload = {
            'subject': subject,
            'seed': SEED,
            'epochs_completed': len(history.history['loss']),
            'duration_s': time.perf_counter() - started,
            'passive_map_session': data['passive_session'],
            'passive_map_source': data['passive_map_source'],
            'passive_entries': data['passive_entries'],
            'results': results,
        }
        (root / f'{subject}.json').write_text(json.dumps(safe(payload), indent=2))
        print(json.dumps({'subject': subject, 'duration_s': payload['duration_s'], 'epochs': payload['epochs_completed']}), flush=True)
        tf.keras.backend.clear_session()

    manifest['completed_utc'] = time.time()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'COMPLETE {root}', flush=True)


if __name__ == '__main__':
    main()

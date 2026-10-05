"""Reproducible within-subject, LOSO, and fixed-budget adaptation experiments.

Run from repository root: python -m ML.training.benchmark --help
Every invocation creates a new directory. Retest evaluation is an explicit switch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from ML.config import DATA_DIR, SUBJECTS
from ML.evaluation.report import aggregate_runs, evaluate, plot_predictions, write_json
from ML.models.baselines import PositionModel, RidgeModel
from ML.preprocessing.pipeline import PipelineConfig, SplitData, load_bundle, source_normalize
from ML.training.run_artifacts import create_run


def predict(model, kind, X, offset=0., scale=1.):
    if not len(X):
        return np.array([], dtype=float)
    if kind in ('ridge', 'position'):
        return model.predict(X)
    return np.concatenate([np.asarray(model(X[i:i + 512], training=False)).ravel()
                           for i in range(0, len(X), 512)]) * scale + offset


def weights_for_positions(data):
    groups = np.char.add(np.char.add(data.metadata['subject'], '/'), data.metadata['position_id'])
    scales = {g: max(float(np.subtract(*np.percentile(data.y[groups == g], [75, 25]))), .1)
              for g in np.unique(groups)}
    mean = np.mean(list(scales.values()))
    weight = np.array([np.clip(mean / scales[g], .2, 5.) for g in groups], np.float32)
    return weight / weight.mean()


def train_candidate(kind, train, val, args, seed):
    if not len(train.y) or not len(val.y):
        raise ValueError('Training and validation require eligible, calibrated windows')
    start = time.perf_counter()
    offset, scale = 0., 1.
    history = {}
    if kind == 'position':
        groups = np.char.add(np.char.add(train.metadata['subject'], '/'), train.metadata['position_id'])
        model = PositionModel().fit(train.X, train.y, groups)
    elif kind == 'ridge':
        candidates = [RidgeModel(a).fit(train.X, train.y) for a in (.1, 1., 10.)]
        scores = [evaluate(val, m.predict(val.X))['mean_subject_position_rmse'] for m in candidates]
        model = candidates[int(np.argmin(scores))]
        history = {'selected_alpha': model.alpha, 'validation_scores': scores}
    else:
        import tensorflow as tf
        from ML.models.lstm import build_model
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        shape = train.X.shape[1:]
        if kind == 'lstm':
            model = build_model(*shape, n_units=8, dropout=args.dropout)
        elif kind == 'mlp':
            # Cropping avoids a nonportable Lambda in the saved model.
            model = tf.keras.Sequential([tf.keras.layers.Input(shape=shape),
                     tf.keras.layers.Cropping1D((shape[0] - 1, 0)), tf.keras.layers.Flatten(),
                     tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(1)])
        else:
            raise ValueError(f'Unknown model {kind}')
        offset, scale = float(train.y.mean()), max(float(train.y.std()), 1e-6)
        model.compile(optimizer=tf.keras.optimizers.Nadam(args.lr, clipnorm=1.), loss='mse')
        fitted = model.fit(train.X, (train.y - offset) / scale,
                  validation_data=(val.X, (val.y - offset) / scale),
                  sample_weight=weights_for_positions(train) if args.position_weights else None,
                  batch_size=args.batch, epochs=args.epochs, shuffle=True, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                                              restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5,
                                                                  factor=.7, min_lr=1e-6)])
        history = fitted.history
        history['best_epoch_one_based'] = int(np.argmin(history['val_loss'])) + 1
        history['loss_units'] = 'squared standardized training torque'
        history['epochs_executed'] = len(history['loss'])
        history['optimizer_updates'] = int(model.optimizer.iterations.numpy())
    return model, offset, scale, history, time.perf_counter() - start


def omit_channel(data, omit):
    if omit is None:
        return data
    from ML.preprocessing.trial_manifest import CHANNELS
    X = data.X.copy()
    X[:, :, CHANNELS.index(omit)] = 0
    return SplitData(X, data.y, data.measured, data.metadata)


def save_result(root, name, kind, seed, data, preprocess, args, fitted=None):
    folder = root / name
    folder.mkdir()
    data = {s: omit_channel(d, args.omit_muscle) for s, d in data.items()}
    model, offset, scale, history, elapsed = fitted or train_candidate(kind, data['train'], data['val'], args, seed)
    if kind in ('ridge', 'position'):
        model.save(folder / 'model.npz')
    else:
        model.save(folder / 'model.keras')
    package = dict(schema_version=1, kind=kind, seed=seed, target_offset=offset, target_scale=scale,
                   omit_muscle=args.omit_muscle, preprocessing=preprocess,
                   observed_training_feature_ranges={
                       'minimum': data['train'].X.min(axis=(0, 1)).tolist(),
                       'maximum': data['train'].X.max(axis=(0, 1)).tolist(),
                       'interpretation': 'Observed range, not a guarantee of physiological validity'},
                   input_shape=list(data['train'].X.shape[1:]))
    write_json(folder / 'package.json', package)
    write_json(folder / 'history.json', history)
    results = dict(name=name, kind=kind, seed=seed, training_seconds=elapsed,
                   parameter_count=int(model.count_params()) if hasattr(model, 'count_params') else None,
                   target_mode=args.target_mode, protocol=args.protocol, splits={})
    for split in ('train', 'val', 'test'):
        if split == 'test' and not args.evaluate_retest:
            continue
        d = data[split]
        yp = predict(model, kind, d.X, offset, scale)
        results['splits'][split] = evaluate(d, yp)
        np.savez_compressed(folder / f'{split}_predictions.npz', y_true=d.y, y_pred=yp,
                            measured_torque=d.measured, **d.metadata)
        if split == 'test' and args.plots:
            plot_predictions(d, yp, folder / 'test_predictions.png')
    # Warmed-up single-window inference, independent of batch throughput.
    sample = data['train'].X[:1]
    predict(model, kind, sample, offset, scale)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        predict(model, kind, sample, offset, scale)
        times.append((time.perf_counter() - t) * 1000)
    results['inference_ms'] = {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95))}
    results['inference_execution'] = 'NumPy' if kind in ('ridge', 'position') else 'eager; use profile_inference for packaged graph timings'
    write_json(folder / 'metrics.json', results)
    print(f"{name}: val macro RMSE={results['splits']['val']['mean_subject_position_rmse']:.3f} Nm; {elapsed:.1f}s", flush=True)
    return results


def calibration_subset(target, seconds, sample_rate, max_positions=2):
    """Bound raw filter support as well as prediction count; never use the retest set.

    This function selects trial/position identities; raw-prefix rebuilding below
    prevents zero-phase preprocessing from borrowing future calibration samples.
    """
    records = [r for r in target.manifest['trials'] if r['kind'] == 'active'
               and r['status'] == 'included' and r['session'] == 'test']
    positions = sorted({r['position_id'] for r in records},
                       key=lambda p: np.mean([r['mean_position_rad'] for r in records if r['position_id'] == p]))
    if seconds <= 0 or sample_rate <= 0 or max_positions < 1:
        raise ValueError('Calibration budget, sample rate, and position count must be positive')
    if not positions:
        return {}
    selected = [positions[i] for i in np.unique(np.linspace(0, len(positions)-1,
                                                          min(max_positions, len(positions))).round().astype(int))]
    overrides = {}
    for pos in selected:
        r = next(r for r in records if r['position_id'] == pos)
        start, end = r['valid_intervals'][0]
        stop = min(end, start + int(seconds / len(selected) * sample_rate))
        overrides[str(r['trial'])] = [[start, stop]]
    return overrides


def run(args):
    if min(args.epochs, args.batch, args.patience) < 1 or not 0 <= args.dropout < 1 or args.lr <= 0:
        raise ValueError('Invalid training configuration')
    if any(n < 0 for n in args.calibration_seconds):
        raise ValueError('Calibration budgets cannot be negative')
    for key in ('subjects', 'models', 'seeds', 'calibration_seconds'):
        values = getattr(args, key)
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate {key} would overwrite run identities')
    config = PipelineConfig(window=args.window, train_stride=args.train_stride, target_mode=args.target_mode,
                            edge_trim_s=args.edge_trim, gap_s=args.gap, lp_cutoff=args.lp_cutoff)
    root = create_run(args.output, vars(args))
    bundles = {}
    preprocessing_profile = {}
    for subject in args.subjects:
        t = time.perf_counter()
        bundle = load_bundle(subject, config, args.data_dir, args.overrides)
        bundles[subject] = bundle
        write_json(root / f'{subject}_manifest.json', bundle.manifest)
        write_json(root / f'{subject}_coverage.json', bundle.coverage)
        write_json(root / f'{subject}_preprocessing.json', bundle.preprocessing)
        preprocessing_profile[subject] = dict(seconds=time.perf_counter()-t,
            feature_target_bytes=sum(d.X.nbytes + d.y.nbytes + d.measured.nbytes for d in bundle.splits.values()))
        print(f'{subject}: {[len(bundle[s].y) for s in ("train", "val", "test")]} windows; build {time.perf_counter()-t:.1f}s', flush=True)
    write_json(root / 'preprocessing_performance.json', preprocessing_profile)
    results = []
    for subject, target in bundles.items():
        if args.protocol == 'within':
            data = target.splits
            prep = target.preprocessing
        else:
            sources = [b for s, b in bundles.items() if s != subject]
            if not sources:
                raise ValueError('Transfer requires at least two subjects')
            source, destination, scales = source_normalize(sources, target)
            data = dict(source, test=destination['test'])
            prep = dict(target.preprocessing, emg_scales=scales.tolist(), normalization='source-training max',
                        source_subjects=[b.manifest['subject'] for b in sources],
                        calibration_protocol='target passive calibration' if args.target_mode == 'active_torque'
                        else 'no target labels or target-fitted EMG normalization')
        for kind in args.models:
            seeds = args.seeds if kind in ('lstm', 'mlp') else [args.seeds[0]]
            for seed in seeds:
                name = f'{subject}_{kind}_seed{seed}'
                result = save_result(root, name, kind, seed, data, prep, args)
                results.append(result)
                if args.protocol == 'adaptation' and kind in ('lstm', 'mlp'):
                    results.extend(run_adaptation(root, name, kind, target, scales, args, seed))
                write_json(root / 'summary.json', results)
    # Adaptation budgets are distinct experimental conditions, never extra subjects.
    aggregate = aggregate_runs([r for r in results if '_cal' not in r['name']])
    write_json(root / 'aggregate.json', aggregate)
    lines = ['# Retrospective benchmark', '', f'Target: {args.target_mode}; protocol: {args.protocol}.',
             'Scores are internal validation on existing participants. Missing coverage is recorded separately.', '',
             '| Run | Validation mean position RMSE (Nm) | Retest mean position RMSE (Nm) | Training seconds |',
             '| --- | ---: | ---: | ---: |']
    for r in results:
        val = r['splits']['val']['mean_subject_position_rmse']
        test = r['splits'].get('test', {}).get('mean_subject_position_rmse')
        lines.append(f"| {r['name']} | {val:.4f} | {test:.4f} | {r['training_seconds']:.1f} |" if test is not None
                     else f"| {r['name']} | {val:.4f} | not evaluated | {r['training_seconds']:.1f} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Artifacts: {root}', flush=True)
    return root


def run_adaptation(root, source_name, kind, target, source_scales, args, seed):
    import copy
    import json
    import tensorflow as tf
    from ML.preprocessing.pipeline import build_bundle
    from ML.preprocessing.trial_manifest import load_inventory
    package = json.loads((root / source_name / 'package.json').read_text())
    results = []
    for budget in args.calibration_seconds:
        if budget <= 0:
            continue  # zero-calibration result is already saved as the source run
        intervals = calibration_subset(target, budget, target.config.sample_rate)
        manifest = copy.deepcopy(target.manifest)
        for r in manifest['trials']:
            if r['kind'] == 'active' and r['session'] == 'test':
                if str(r['trial']) in intervals:
                    r['valid_intervals'] = intervals[str(r['trial'])]
                else:
                    r['status'] = 'excluded'
        trials, _ = load_inventory(target.manifest['subject'], args.data_dir, args.overrides)
        calibrated = build_bundle(trials, manifest, target.config, args.data_dir)
        converted = {}
        for split, d in calibrated.splits.items():
            X = d.X.copy()
            X[:, :, :4] *= (np.asarray(calibrated.preprocessing['emg_scales']) / source_scales).astype(np.float32)
            converted[split] = omit_channel(SplitData(X, d.y, d.measured, d.metadata), args.omit_muscle)
        if min(len(converted[s].y) for s in ('train', 'val')) == 0:
            write_json(root / f'{source_name}_cal{budget}_unavailable.json',
                       {'reason': 'Insufficient calibration after gap/edge trimming', 'intervals': intervals})
            continue
        tf.keras.utils.set_random_seed(seed)
        model = tf.keras.models.load_model(root / source_name / 'model.keras', compile=False)
        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True
        model.compile(optimizer=tf.keras.optimizers.Nadam(1e-4, clipnorm=1.), loss='mse')
        offset, scale = package['target_offset'], package['target_scale']
        tr, val = converted['train'], converted['val']
        start = time.perf_counter()
        history = model.fit(tr.X, (tr.y-offset)/scale, validation_data=(val.X, (val.y-offset)/scale),
                            epochs=args.epochs, batch_size=args.batch, verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=args.patience,
                                                                         restore_best_weights=True)]).history
        prep = dict(package['preprocessing'], calibration_seconds=budget, calibration_intervals=intervals,
                    actual_calibration_seconds=sum(b-a for spans in intervals.values() for a, b in spans)
                                               / target.config.sample_rate,
                    adaptation='dense_only; raw calibration prefix isolated before filtering')
        name = f'{source_name}_cal{budget}s'
        result = save_result(root, name, kind, seed, converted, prep, args,
                             fitted=(model, offset, scale, history, time.perf_counter()-start))
        results.append(result)
    return results


def parser(default_protocol='within'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', choices=['within', 'loso', 'adaptation'], default=default_protocol)
    p.add_argument('--subjects', '--subject', nargs='+', choices=list(SUBJECTS), default=list(SUBJECTS))
    p.add_argument('--models', nargs='+', choices=['position', 'ridge', 'mlp', 'lstm'], default=['position', 'ridge', 'lstm'])
    p.add_argument('--data-dir', default=DATA_DIR)
    p.add_argument('--output')
    p.add_argument('--overrides', help='JSON of subject -> one-based trial -> evidence-backed overrides')
    p.add_argument('--target-mode', choices=['measured_torque', 'active_torque'], default='active_torque')
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--window', type=int, default=50,
                   help='History length in post-downsampled steps (default: 50 = 500 ms)')
    p.add_argument('--train-stride', type=int, default=5)
    p.add_argument('--edge-trim', type=float, default=1.)
    p.add_argument('--gap', type=float, default=1.)
    p.add_argument('--lp-cutoff', type=float, default=2.)
    p.add_argument('--position-weights', action='store_true')
    p.add_argument('--omit-muscle', choices=['gm', 'gl', 'sol', 'ta'])
    p.add_argument('--calibration-seconds', nargs='+', type=int, default=[30, 60, 120])
    p.add_argument('--evaluate-retest', action='store_true', help='Explicit final/internal retest evaluation')
    p.add_argument('--plots', action='store_true')
    return p


def main(default_protocol='within'):
    args = parser(default_protocol).parse_args()
    if min(args.epochs, args.batch, args.patience) < 1 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid training configuration')
    run(args)


if __name__ == '__main__':
    main()

"""Compare eager and cached-graph inference on the same saved neural package.

This measures computation on synthetic, already preprocessed windows, not raw
signal latency. Run without concurrent training for a useful latency estimate.
"""
import argparse
from pathlib import Path
import platform
import time
import numpy as np

from ML.evaluation.report import write_json
from ML.inference import Predictor


def profile(folder, repeats=200):
    predictor = Predictor(folder)
    if predictor.package['kind'] not in ('lstm', 'mlp') or repeats < 20:
        raise ValueError('Requires a neural package and at least 20 repeats')
    X = np.random.default_rng(11).normal(0, .1, (1, *predictor.package['input_shape'])).astype(np.float32)
    if predictor.package.get('omit_muscle'):
        from ML.preprocessing.trial_manifest import CHANNELS
        X[:, :, CHANNELS.index(predictor.package['omit_muscle'])] = 0
    offset, scale = predictor.package['target_offset'], predictor.package['target_scale']
    def eager():
        return np.asarray(predictor.model(X, training=False)).ravel() * scale + offset
    start = time.perf_counter()
    compiled = predictor.predict_windows(X)
    cold_ms = (time.perf_counter()-start) * 1000
    np.testing.assert_allclose(compiled, eager(), atol=1e-5, rtol=1e-6)
    for _ in range(5):
        eager()
        predictor.predict_windows(X)
    durations = {'eager': [], 'compiled': []}
    for i in range(repeats):
        # Alternate timing order to reduce systematic warm-up/order bias.
        methods = [('eager', eager), ('compiled', lambda: predictor.predict_windows(X))]
        for name, method in methods[::1 if i % 2 else -1]:
            start = time.perf_counter()
            method()
            durations[name].append((time.perf_counter()-start) * 1000)
    return dict(package=str(folder), platform=platform.platform(), repeats=repeats,
                input='synthetic preprocessed single window', cold_graph_ms=cold_ms,
                timing_ms={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)))
                           for k, v in durations.items()},
                scope='CPU computation only; excludes acquisition, filters, resampling, and signal delay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = profile(args.package, args.repeats)
    write_json(output, result)
    print(result)


if __name__ == '__main__':
    main()

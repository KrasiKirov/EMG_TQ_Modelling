"""Rebuild held-out windows and verify saved predictions in a fresh process."""
import argparse
import json
from pathlib import Path
import numpy as np

from ML.config import DATA_DIR, SUBJECTS
from ML.evaluation.report import write_json
from ML.inference import Predictor
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.pipeline import build_bundle
from ML.preprocessing.trial_manifest import file_hash


def verify(run, data_dir, names=None, atol=1e-5):
    root, data_dir = Path(run), Path(data_dir)
    reports, cache = {}, {}
    for path in sorted(root.glob('*/package.json')):
        name = path.parent.name
        if names and name not in names:
            continue
        prediction_file = path.parent / 'test_predictions.npz'
        if not prediction_file.exists():
            continue
        subject = name.split('_')[0]
        predictor = Predictor(path.parent)
        config = predictor.config
        intervals = predictor.preprocess.get('calibration_intervals')
        key = (subject, config, json.dumps(intervals, sort_keys=True))
        if key not in cache:
            filename, subject_id = SUBJECTS[subject]
            manifest = json.loads((root / f'{subject}_manifest.json').read_text())
            if intervals is not None:
                # Reproduce the exact float32 normalization path used when the
                # adaptation package was saved, including its prefix-only scale.
                for record in manifest['trials']:
                    if record['kind'] == 'active' and record['session'] == 'test':
                        if str(record['trial']) in intervals:
                            record['valid_intervals'] = intervals[str(record['trial'])]
                        else:
                            record['status'] = 'excluded'
            if file_hash(data_dir / filename) != manifest['source_sha256']:
                raise ValueError(f'{subject}: raw recording changed')
            trials = read_flb(data_dir / filename, subject_id=subject_id)
            cache[key] = build_bundle(trials, manifest, config, data_dir)
        bundle = cache[key]
        if bundle.preprocessing['passive'] != predictor.preprocess['passive']:
            raise ValueError(f'{name}: calibration changed')
        if bundle.preprocessing['source_sha256'] != predictor.preprocess['source_sha256']:
            raise ValueError(f'{name}: package source mismatch')
        data = bundle['test']
        X = data.X.copy()
        X[:, :, :4] *= (np.asarray(bundle.preprocessing['emg_scales']) /
                        np.asarray(predictor.preprocess['emg_scales'])).astype(np.float32)
        prediction = predictor.predict_windows(X)
        with np.load(prediction_file, allow_pickle=False) as saved:
            for field in ('subject', 'trial', 'sample_index', 'position_id'):
                np.testing.assert_array_equal(data.metadata[field], saved[field])
            np.testing.assert_allclose(data.y, saved['y_true'], atol=atol, rtol=1e-6)
            np.testing.assert_allclose(prediction, saved['y_pred'], atol=atol, rtol=1e-6)
            error = float(np.max(np.abs(prediction-saved['y_pred']), initial=0.))
        reports[name] = dict(windows=len(prediction), max_absolute_difference_nm=error,
                             absolute_tolerance_nm=atol, relative_tolerance=1e-6, passed=True)
        print(f'{name}: {len(prediction)} predictions reproduced; max difference {error:.3g} Nm', flush=True)
    if not reports or names and set(names) != set(reports):
        raise ValueError('Requested packages with saved retest predictions were not all found')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = verify(args.run, args.data_dir, args.names)
    write_json(output, report)


if __name__ == '__main__':
    main()

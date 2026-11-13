"""Evaluate existing checkpoints using the legacy builder; never call this a reproduced run."""
import argparse
from pathlib import Path
import time
import numpy as np
import tensorflow as tf

from ML.config import DATA_DIR, SUBJECTS
from ML.evaluation.metrics import compute_metrics
from ML.evaluation.report import write_json
from ML.preprocessing.dataset_builder import build_dataset
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.trial_manifest import file_hash
from ML.training.run_artifacts import create_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = create_run(args.output, vars(args))
    results = {}
    for subject, (filename, sid) in SUBJECTS.items():
        path = Path(args.checkpoints) / subject / 'best_lstm.keras'
        if not path.exists():
            results[subject] = {'status': 'checkpoint missing'}
            continue
        model = tf.keras.models.load_model(path, compile=False)
        trials = read_flb(Path(args.data_dir) / filename, subject_id=sid)
        dataset = build_dataset(trials)
        X, y = dataset[4:6]
        start = time.perf_counter()
        yp = np.concatenate([np.asarray(model(X[i:i+512], training=False)).ravel()
                             for i in range(0, len(X), 512)])
        results[subject] = dict(status='reconstructed; historical preprocessing configuration unknown',
                               checkpoint_sha256=file_hash(path), source_sha256=file_hash(Path(args.data_dir)/filename),
                               input_shape=list(model.input_shape[1:]), parameter_count=model.count_params(),
                               metrics=compute_metrics(y, yp), inference_seconds=time.perf_counter()-start)
        write_json(root / 'reference.json', results)
    print(root)


if __name__ == '__main__':
    main()
